// src/kernels/cuda/mid_chain_ada.cu - the mid chain on compute capability 8.9 (see include/basalt/kernels/mid_chain.hpp).
//
// One persistent kernel, one CTA of four 128-thread groups per SM, runs the unfused sequence of a GDN layer's middle on
// such a card (native_quantize_q8_1 of the read's rows, dense_tc qkv and z, gdn_ab_multi, gdn_conv_l2_multi,
// gdn_step_norm_multi, native_quantize_q8_1 of its output, dense_tc out) through those kernels' own bodies
// (dense_tc_block, gdn_unfused.cuh), so every output has their bits.  The phases hand over through grid barriers in
// global memory; a wait past its bound sets the abort word and falls through (wait_abort.hpp).  The kernel needs every
// CTA resident, and the output's q8_1 rows (rec.yq) apart from the read's (xq).  Its own translation unit, built
// without fast math like dense_tc.cu and verify_kernels.cu.
#define FZ_DEV_ONLY
#include "dense_tc.cu"
#include "gdn_unfused.cuh"
#include "basalt/kernels/mid_chain.hpp"
#include "basalt/kernels/q8_1_finite.hpp"
#include "basalt/kernels/verify_kernels.hpp"
#include "basalt/kernels/wait_abort.hpp"

#include <cstdlib>

namespace basalt::kernels {
namespace {

constexpr int kAdaThreads = 512;   // four groups of 128 threads
constexpr int kAdaDtc = 3;         // groups running dense_tc blocks: dense_tc's own three blocks per SM
constexpr unsigned long long kAdaWaitNs = 200000000ull;   // a barrier wait's bound
constexpr int kAbNJ = 10;          // uint4 of bf16 per lane in a 2560-wide alpha/beta row

// a group's dense_tc_block<TYPE, 3, 1, 1> shared memory (smem_bytes), 128-byte aligned
template <int TYPE>
__host__ __device__ constexpr int dtc_smem() {
    return (3 * 16 * row_stride(Fmt<TYPE>::SB) + 4 * 16 * kMaxCols * 4 + 127) & ~127;
}

struct AdaArgs {
    int T;
    const float* x;                  // the read's rows
    uint8_t* xq;                     // their q8_1 rows
    const uint8_t *wq, *wz, *wo;
    DecRecArgs rec;
    float *qkv, *z;
    float* bo;
    unsigned* bar;                   // [0] arrivals, [1] generation
    unsigned long long* word;        // the abort word
    int pair_only;
};

struct BarGroup {   // the barrier of a 128-thread group (ids 1..4; __syncthreads is 0)
    int id;
    __device__ __forceinline__ void operator()() const { asm volatile("bar.sync %0, 128;" ::"r"(id) : "memory"); }
};

__device__ __forceinline__ unsigned ada_ld_acquire(const unsigned* p) {
    unsigned v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ unsigned long long ada_now() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
// native_mmvq.cu's arithmetic (built with fast math): flush-to-zero absolute value, maximum, sum and division
__device__ __forceinline__ float abs_ftz(float a) { float r; asm("abs.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }
__device__ __forceinline__ float max_ftz(float a, float b) { float r; asm("max.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float add_ftz(float a, float b) { float r; asm("add.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float div_ftz(float a, float b) { float r; asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }

// the q8_1 block of 32 values (one warp): native_quantize_q8_1_kernel's
__device__ __forceinline__ void q81_block(const float* src, uint8_t* dst, int lane) {
    const float xi = src[lane];
    float amax = abs_ftz(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = max_ftz(amax, __shfl_xor_sync(0xffffffffu, amax, o, 32));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum = add_ftz(sum, __shfl_xor_sync(0xffffffffu, sum, o, 32));
    const float d = q8_1_finite(div_ftz(amax, 127.0f));
    int8_t q = 0;
    if (amax != 0.0f) {
        const float v = roundf(div_ftz(xi, d));
        q = (int8_t) (v > 127.0f ? 127.0f : v < -127.0f ? -127.0f : v);
    }
    dst[4 + lane] = (uint8_t) q;
    if (lane == 0) *reinterpret_cast<half2*>(dst) = q8_1_ds(d, sum);
}

// every CTA's writes before it are visible to every CTA after it
__device__ __noinline__ void grid_sync(unsigned* bar, unsigned long long* word, int line) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned gen = ada_ld_acquire(bar + 1);
        __threadfence();
        if (atomicInc(bar, gridDim.x - 1) == gridDim.x - 1) {   // the last arrival: the count wraps to 0
            __threadfence();
            atomicAdd(bar + 1, 1u);
        } else {
            const unsigned long long t0 = ada_now();
            while (ada_ld_acquire(bar + 1) == gen) {
                __nanosleep(32);
                const unsigned long long e = ada_now() - t0;
                if (e > kWaitPollNs &&
                    wait_give_up(word, e, kAdaWaitNs, wait_abort_code_of(kWaitMidChain, 2, (unsigned) line, blockIdx.x)))
                    break;
            }
        }
        __threadfence();
    }
    __syncthreads();
}

template <int TQ, int TZ, int TO>
__global__ void __launch_bounds__(kAdaThreads, 1) k_mid_ada(const __grid_constant__ AdaArgs p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    namespace gu = gdn_unfused;
    extern __shared__ __align__(128) uint8_t sm_ada[];
    __shared__ float conv_part[4][gu::S / 32];
    constexpr int GS = dtc_smem<TQ>() > dtc_smem<TZ>() ? (dtc_smem<TQ>() > dtc_smem<TO>() ? dtc_smem<TQ>() : dtc_smem<TO>())
                                                       : (dtc_smem<TZ>() > dtc_smem<TO>() ? dtc_smem<TZ>() : dtc_smem<TO>());
    const int tid = threadIdx.x, b = blockIdx.x, G = gridDim.x, grp = tid >> 7, gt = tid & 127;
    const int T = p.T;
    const DecRecArgs& a = p.rec;
    const int ZV = gu::S * a.h_v;
    const BarGroup gsync{1 + grp};
    uint8_t* gsm = sm_ada + (size_t) (grp < kAdaDtc ? grp : 0) * GS;
    // every phase deals its units over the CTAs first (unit k on CTA k % G), so a phase of few units spreads over the SMs
    // ======== the read's rows to q8_1, one warp per 32-value block
    {
        const int nblk = T * (a.n_embd / 32), lane = tid & 31;
        for (int k = (tid >> 5) * G + b; k < nblk; k += G * 16) q81_block(p.x + (size_t) k * 32, p.xq + (size_t) k * kQ81, lane);
    }
    grid_sync(p.bar, p.word, __LINE__);
    // ======== qkv and z (dense_tc's persistent blocks, kAdaDtc per CTA); alpha/beta on the fourth group
    if (grp < kAdaDtc) {
        const int vb = grp * G + b, nv = G * kAdaDtc;
        dense_tc_block<TQ, 3, 1, 1>(vb, nv, gsm, p.wq, p.xq, p.qkv, a.n_embd, a.C, T, gt, gsync);
        dense_tc_block<TZ, 3, 1, 1>(vb, nv, gsm, p.wz, p.xq, p.z, a.n_embd, ZV, T, gt, gsync);
    } else {
        const int nab = (2 * a.h_v + 1) / 2;
        for (int k = (gt >> 6) * G + b; k < nab; k += 2 * G)
            gu::ab_rows_unrolled<kVerifyMaxT, kAbNJ>(k, gt & 63, a.x, a.w_alpha, a.w_beta, a.dt, a.ssm_a, a.gate, a.beta,
                                                     a.n_embd, a.h_v, T);
    }
    grid_sync(p.bar, p.word, __LINE__);
    if (p.pair_only != 0) return;
    // ======== conv + L2 norm: (channel block, token) units of 128 threads
    {
        const int ncb = a.C / gu::S, n = ncb * T;
        const float* hist = a.state + (size_t) gu::S * a.h_v * gu::S;
        for (int k = grp * G + b; k < n; k += 4 * G)
            gu::conv_l2_unit(k % ncb, k / ncb, gt, conv_part[grp], gsync, hist, p.qkv, a.conv_w, a.h, a.C, 2 * a.h_k, a.eps);
    }
    grid_sync(p.bar, p.word, __LINE__);
    // ======== recurrence + gated norm, one value head per CTA, then the head's q8_1 blocks
    for (int head = b; head < a.h_v; head += G) {
        gu::step_norm_head(head, tid & (gu::S - 1), tid / gu::S, const_cast<float*>(a.state), a.h, a.C, a.gate, a.beta,
                           a.z, a.gamma, a.eps, a.y, a.h_k, a.h_v, T, false, 0);
        __syncthreads();   // the head's output rows are written
        const int lane = tid & 31;
        for (int k = tid >> 5; k < T * (gu::S / 32); k += kAdaThreads / 32) {
            const size_t off = (size_t) (k / (gu::S / 32)) * ZV + (size_t) head * gu::S + (size_t) (k % (gu::S / 32)) * 32;
            q81_block(a.y + off, static_cast<uint8_t*>(a.yq) + off / 32 * kQ81, lane);
        }
    }
    grid_sync(p.bar, p.word, __LINE__);
    // ======== the output projection
    if (grp < kAdaDtc)
        dense_tc_block<TO, 3, 1, 1>(grp * G + b, G * kAdaDtc, gsm, p.wo, static_cast<const uint8_t*>(a.yq), p.bo, ZV,
                                    a.n_embd, T, gt, gsync);
#elif defined(__CUDA_ARCH__)
    __trap();
#endif
}

using AFn = void (*)(AdaArgs);
// the type triples with an instance (all three projections of one type)
AFn ada_pick(int q, int z, int o) {
    if (q == 8 && z == 8 && o == 8) return k_mid_ada<8, 8, 8>;
    if (q == 14 && z == 14 && o == 14) return k_mid_ada<14, 14, 14>;
    return nullptr;
}
int ada_smem(int q) { return kAdaDtc * (q == 8 ? dtc_smem<8>() : dtc_smem<14>()); }

struct AdaDev {
    bool checked = false, ok = false, ada = false;
    int ctas = 0;
    unsigned* bar = nullptr;
};
AdaDev g_ada[16];
bool g_ada_on = false;
AdaDev& ada_dev() {
    int dev = 0;
    cudaGetDevice(&dev);
    static AdaDev none;
    if (dev < 0 || dev >= 16) return none;
    AdaDev& d = g_ada[dev];
    if (!d.checked) {
        d.checked = true;
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
        d.ada = major == 8 && minor == 9;
    }
    return d;
}

}  // namespace

void mid_chain_ada_enable(bool on) { g_ada_on = on; }

bool mid_ada_device() { return g_ada_on && ada_dev().ada; }

void mid_ada_init() {
    AdaDev& d = ada_dev();
    if (d.ok || d.bar != nullptr) return;
    int dev = 0, sms = 0, optin = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    if (smem_bytes<8, 3, 1, 1>() > (size_t) dtc_smem<8>() || smem_bytes<14, 3, 1, 1>() > (size_t) dtc_smem<14>()) {
        basalt::log::error("basalt", "mid_chain_ada: dense_tc block shared memory mismatch");
        std::exit(1);
    }
    wait_abort_init();
    bool ok = cudaMalloc((void**) &d.bar, 8) == cudaSuccess && cudaMemset(d.bar, 0, 8) == cudaSuccess;
    for (const int t : {8, 14}) {
        if (!ok) break;
        const AFn f = ada_pick(t, t, t);
        int occ = 0;
        ok = optin >= ada_smem(t) &&
             cudaFuncSetAttribute((const void*) f, cudaFuncAttributeMaxDynamicSharedMemorySize, ada_smem(t)) == cudaSuccess &&
             cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, f, kAdaThreads, (size_t) ada_smem(t)) == cudaSuccess && occ >= 1;
    }
    cudaDeviceSynchronize();
    d.ok = ok && cudaGetLastError() == cudaSuccess;
    d.ctas = sms;
    if (!d.ok) {
        basalt::log::error("basalt", "mid_chain_ada: FATAL: kernel initialisation failed (%s)", cudaGetErrorString(cudaGetLastError()));
        std::exit(1);
    }
    basalt::log::info("basalt", "mid_chain: sm_89 kernel, %d CTAs of %d threads", d.ctas, kAdaThreads);
}

bool mid_ada_supported(const MidChainArgs& a) {
    const AdaDev& d = ada_dev();
    const DecRecArgs& r = a.rec;
    auto al = [](const void* p, uintptr_t m) { return p != nullptr && ((uintptr_t) p & (m - 1)) == 0; };
    // yq apart from xq: the output projection's non-coherent loads of yq would hit lines the pair's loads of xq cached
    auto apart = [](const void* p, size_t n, const void* q, size_t m) {
        return (uintptr_t) p + n <= (uintptr_t) q || (uintptr_t) q + m <= (uintptr_t) p;
    };
    // dense_tc's own choice for these shapes is the <3, 1, 1> persistent block: the output projection's 160 tiles
    // are not fewer than the SMs
    return d.ok && ada_pick(a.type_q, a.type_z, a.type_o) != nullptr && a.stamps == nullptr && a.T >= 1 &&
           a.T <= kMidChainMaxT && a.T <= kVerifyMaxT && a.ctas >= 0 && a.ctas <= d.ctas && d.ctas <= 160 &&
           al(a.wq, 4) && al(a.wz, 4) && al(a.wo, 4) && al(a.xq, 4) && al(a.bo, 4) && al(r.x, 16) && al(r.qkv, 4) &&
           al(r.z, 4) && al(r.h, 4) && al(r.gate, 4) && al(r.beta, 4) && al(r.y, 4) && al(r.yq, 4) && al(r.state, 4) &&
           al(r.conv_w, 4) && al(r.w_alpha, 16) && al(r.w_beta, 16) && al(r.dt, 4) && al(r.ssm_a, 4) && al(r.gamma, 4) &&
           r.n_embd == 32 * 8 * kAbNJ && r.h_k > 0 && r.h_v % r.h_k == 0 && r.C == (2 * r.h_k + r.h_v) * gdn_unfused::S &&
           r.nk == 0 && apart(a.xq, (size_t) a.T * (r.n_embd / 32) * kQ81, r.yq,
                              (size_t) a.T * (r.h_v * gdn_unfused::S / 32) * kQ81);
}

void mid_ada(const MidChainArgs& a, void* stream) {
    AdaDev& d = ada_dev();
    if (!mid_ada_supported(a)) {
        basalt::log::error("basalt", "mid_chain_ada: unsupported arguments or device");
        std::exit(1);
    }
    AdaArgs p{};
    p.T = a.T;
    p.x = a.rec.x;
    p.xq = static_cast<uint8_t*>(const_cast<void*>(a.xq));
    p.wq = static_cast<const uint8_t*>(a.wq);
    p.wz = static_cast<const uint8_t*>(a.wz);
    p.wo = static_cast<const uint8_t*>(a.wo);
    p.rec = a.rec;
    p.qkv = const_cast<float*>(a.rec.qkv);
    p.z = const_cast<float*>(a.rec.z);
    p.bo = a.bo;
    p.bar = d.bar;
    p.word = wait_abort_device_word();
    p.pair_only = a.pair_only ? 1 : 0;
    // the first launch of a graph clears the barrier (a launch that gave up a wait leaves it mid-count)
    if (a.ord == 0) cudaMemsetAsync(d.bar, 0, 8, (cudaStream_t) stream);
    const int g = a.ctas > 0 ? a.ctas : d.ctas;
    ada_pick(a.type_q, a.type_z, a.type_o)<<<g, kAdaThreads, (size_t) ada_smem(a.type_q), (cudaStream_t) stream>>>(p);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "mid_chain_ada: %s", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace basalt::kernels
