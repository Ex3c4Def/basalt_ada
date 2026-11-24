# Ada (RTX 4090, sm_89) port of Basalt: briefing for the agent that builds and tests it

## Goal
Make Basalt build for compute capability 8.9 (RTX 4090) and run the NVFP4 pack on that card through fallback kernels.
Upstream Basalt is Blackwell-only (sm_120). The user asked for this port in their **local fork only**: AGENTS.md /
CONTRIBUTING.md rules against other architectures and the parity-test requirements do not apply here, but sm_120 behaviour
must stay exactly as it was. Do not git commit, push or delete anything unless the user asks.

## State
Builds and runs on the 4090. The first port changed 14 files (CMakeLists.txt, src/core/device.cu, qwen4exp's dec_gdn,
dec_gdn_rec, dec_moe, dense_tc, hc_fused, verify_kernels, prefill experts and layout, src/kernels/cuda/gu2.cuh,
src/prefill/mmq/mmq_common.cuh, src/program/gpu_select.cpp); each upstream merge below says what 8.9 needed since.

## Build (Windows, docs/platforms.md has the details)
Needs Visual Studio 2022 C++ tools, CUDA 13.x, CMake 3.24+, Ninja. The desktop app does not load vcvars, so use
`ada_build.bat nopause` in this folder (it finds Visual Studio, calls vcvars64.bat, configures and builds with
`-DCMAKE_CUDA_ARCHITECTURES=89 -G Ninja` into `build-ada`, keeps going past errors with `ninja -k 0`, and writes
everything to `ada_build.log`; the last lines say BUILD OK / BUILD FAILED / CONFIGURE FAILED).
Read the first errors in `ada_build.log`, fix, rerun.

## What sm_89 does not have (so what any compile error will be about)
Thread-block clusters and distributed shared memory, TMA / `cp.async.bulk` and mbarrier `expect_tx`/`try_wait`,
`griddepcontrol` (PDL), `fence.proxy.async`, `mul.bf16x2` (all sm_90+), and the block-scaled MMAs, `cvt e2m1` and the
direct `cvt.rn.bf16x2.e4m3x2` (sm_120a / sm_100+). It does have int8, bf16, f16 and plain e4m3 `mma.sync` (the e4m3 one
sums with too few bits for a W8A8 product: see the dense stack below), `cvt.rn.f16x2.e4m3x2`, `cp.async`, and 99 KB
opt-in shared memory per block.
nvcc here is whole-program (no rdc), so unreachable `__device__` helpers are dropped; only code reachable from a
compiled kernel must compile for sm_89.
Pattern used: wrap sm_90+ instructions in `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900` (or a local macro
`HC_SM90_ONLY` / `DM_SM90_ONLY` / `REC_NO_CLUSTER`), let the kernel trap below that, and make a **runtime** predicate pick
the fallback so the trap is never launched.

## How the fallbacks are chosen on the 4090
- GDN recurrence: `dec_gdnf_mode()` (src/kernels/cuda/gdn_window.cu since 0.4.0, a runtime function) returns 0 below sm_90, so
  `record_gdn.cpp` takes the unfused sequence. On 8.9 it runs as one persistent kernel, `k_mid_ada` (mid_chain_ada.cu),
  bitwise equal to the separate kernels, after a join on the L2 read-ahead. `--exp mid_ada=0` runs the separate kernels
  (gdn_conv_l2_multi, gdn_ab_multi, gdn_step_norm_multi, z on native_mmvq), `--exp mid_ada=2` the kernel with z on
  dense_tc (other bits). Development A/B at 450 W: +2.2% decode (token-weighted) over `mid_ada=0`, faster on conv,
  struct and prose (ADA_MID_CHAIN_PLAN.md).
- Dense projections: the persistent `dense_tc` kernel (cp.async, int8 `mma.sync`) runs on sm_89; its bulk-copy variants
  (tile, one-tile, pair) stay off through `tc_bulk_ok()` (dense_tc.cu), and so does `qsa_qproj_half` (qsa_mid.cu).
- Lazy GDN commit: off below sm_90 (lifecycle.cpp): its read-ahead blocks wait beside the MoE grid and deadlock without
  PDL. The read-ahead itself (the next layer's projections and FFN read matrices into L2) runs alone instead, launched
  after each layer's MoE grid beside the combine (`gdn_l2_ahead`, `l2_ahead` in record_moe.cpp); `--exp l2_ahead=0`
  turns it off. Development A/Bs at 450 W: +0.7% and +1.5% decode (token-weighted), 21 of 24 paired runs faster;
  greedy output unchanged.
- Fused HC read and the sm_120 mid chain: already off unless the device major is 12 (hc_fused.cu `hc_fused_supported`).
- HC read: on 8.9 one persistent kernel, `k_hc_ada` (hc_read_ada.cu), bitwise equal to hc8_gr's three kernels, when no
  side branch is open, after a join on the L2 read-ahead. `--exp hc_ada=0` runs the three kernels. The router, the q8_1
  rows and the draft graphs stay outside it. Development A/B at 450 W, per-window log on, a download running: +3.17%
  decode (token-weighted) over `hc_ada=0`, faster on conv, struct and prose (HC_READ_ADA_PLAN.md).
- Route: without a peer, staged share or shared gate, `dec_moe_route` on 8.9 takes `dm_plan_ada_kernel` (block 0 plans, the
  others go straight to the q8_1 rows) and `dm_ring_ada_kernel` (one block publishes the header and x in 16-byte stores and
  rings; blocks for the warm-up's rows do not hold the doorbell back, so those rows may land after it, which changes no
  output). `--exp dm_route_ada=0` runs `dm_route_kernel`, which is unchanged. `dm_route_ada_parity`: 0 differing bytes in
  2560 cases against `dm_route_kernel`; greedy identical on conv, struct and prose. 450 W, development only, three A/Bs
  pooled, 20 pairs each, paired median: conv +0.3 %, struct +0.5 %, prose +2.8 % (its route busy time 7.6 -> 4.3 % of the
  cycle); only prose gains beyond the spread, and expert hit rates are unchanged. A video playing during an A/B once cost
  a conv run 21.5 % across every window width: log `utilization.decoder` beside an A/B.
- HC read and route together, against the build without them (`local_bench/engine_bench.py --tests community`, a fresh
  server, 450 W, DDR5-4400, 4 rotated pairs): conv / struct / prose / decode_tps 111.2 / 200.3 / 120.4 / 170.2 -> 118.7 /
  212.3 / 127.3 / 180.1, +5.8 %, faster in every pair (basalt-bench: the HC read alone +3.53 %, the route on top +1.40 %).
  The gate's rows (100 chunks, T 3) are byte-identical; greedy, and sampled with `--adapt-swaps 0`, are identical.
- Measuring: plain basalt-bench asks for the per-window log unless `--no-windows`, about 2 % slower, so compare its tok/s
  only with its own. It keeps the last of a repeated engine flag, so two experiment gates go in one `--exp a=0,b=0`.
- Comparing the quality gate across builds: its rows follow the GPU slots, and a bigger binary leaves less VRAM (6491 slots
  against 6515 changed nearly every row, KLD 0.0620 against 0.0619). Fix `--expert-cache` when comparing; with it equal the
  rows are byte-identical.
- DeepSeek-V4.1 on 8.9: its code uses no bulk copies, griddepcontrol or FP4 tensor cores; DSpark decodes MXFP4 in
  software. Since upstream 0.3.1+13 one kernel uses a cluster: the index key (`index_key_kernel`, attn_index.cu, eight
  blocks a token). Below 9.0 it compiles without its cluster and traps, and `index_keys()` runs `index_key_block_kernel`
  instead: a block a token, the same row sums (`index_key_row`) and the same finish (`index_key_finish`), so the keys keep
  the cluster form's bits (`block_sum`'s extra warps add zeros). sm_120a's SASS of attn_index.cu is unchanged apart from the
  added kernel. Its `ds41_*` parity tests (`-DBUILD_TESTING=ON`) pass on a 4090 at 450 W, the peer test only with one card.
  No pack has been run, so end-to-end output, quality and speed are unmeasured.
- Frozen hashes: `ds41_router_parity`, `ds41_attn_index_parity` (gemv_bf16, compressor rows and keys) and
  `ds41_dspark_parity` hash a kernel's output against the bits of the kernel it replaced, frozen upstream on Blackwell.
  On the 4090 the replaced kernels (fork_4090's sources before 0.3.1+13) give other bits, and their replacements give
  exactly those: router f4ec7c76f483174c, gemv 258e734e874cae57, compressor d2ec8ec03dfea810, DSpark MoE c7b1acee87ad0e77.
  `frozen_for_device` (kernels/test_util.hpp) uses these below 9.0, upstream's values from 9.0 up.
- Since 0.3.2+80 DSpark's MoE rows take q8_1 activations and dp4a (upstream 30ad59a6), which changes their bits by design;
  upstream froze the new kernel's own bits (3be9ade2c87c72d7). dspark_kernels.cu has no architecture-dependent code, and
  every other check in `ds41_dspark_parity` passes on the 4090 (against the double reference, the fp32 hidden rows, live
  rows bitwise), so the 4090's bits of the same kernel are its frozen value below 9.0: 9c13389dbf259e5f. Unlike the values
  above, this one was not checked against a replaced kernel on the 4090: it pins the dp4a kernel's bits for later changes.
- Draft reads: on 8.9 the draft graphs' hyper-connection reads, their router and q8_1 rows run as one kernel,
  `hc_fused_ada` (hc_fused_ada.cu), where `hc_fused` does not run. Verify stays on `k_hc_ada`, and `--exp hc_fused_ada=0`
  keeps the separate kernels. A one-row read now takes fused_gr_read_multi's bits instead of gr_read's, which only the
  drafts see: greedy and sampled output with swaps off, and the gate's rows at equal slots, are identical.
  `hc_fused_ada_parity`: 0 differing cases at T 1..8, 0 of 208 stress reads, 0 stuck with 100 SMs held. Microbench at
  450 W, T = 1, hot / cold: read 24.4 / 32.7 -> 12.5 / 26.3 us; with the router 38.1 / 44.2 -> 17.0 / 33.4 us. The same
  binary on against off at 450 W, development only: struct +0.15 % (10 pairs), conv +0.7 % and prose +1.8 % pooled over 8
  seeds; on identical text conv +0.5 %, prose about +1.3 %. The traces suggested about 1.4 % on struct; the GDN commit
  running beside the draft may be why it does not show. Struct runs near 199 tok/s were seen with it, but in the same-binary
  A/B the worst dip (203) was with it off.
- Measuring conv and prose: their text path moves them more than most changes do, so use several seeds. Seed 7 alone
  showed this kernel at +4.4 % on conv.
- NVFP4 prefill: `fp4_tensor_cores()` (layout.cpp) is true only for major >= 12. `fp4_pack()`, `fp4_every_layer()` and
  `fp4_l` (experts.hpp) depend on it, so NVFP4 experts take the MMQ path (src/prefill/mmq, mmq_nvfp4) and the MMQ
  buffers are sized for every chunk length. If a buffer overflow or wrong-size error shows up in prefill, look at
  `moe_bufs` / `group_rows_cap` in layout.cpp first.
- Sampler / qsa_select cluster kernels check the device major. PDL launches (`launch_pdl`, `launch_dependent`) check
  `pdl_kernel_ok` and launch plain below 9.0: a programmatic edge without griddepcontrol only adds latency, 25 to 42 us
  once a fork/join has moved the chain to another of the graph's internal streams.
- The VRAM TMA expert kernel (`dec_moe_vram` kind 1) is never selected by default.

## Multi-GPU and the other upstream features on 8.9 (checked 2026-11-08, upstream 0.3.0+110)
- Cards exchange data without GPU-to-GPU access (GeForce 40 has none), through pinned host buffers (`StagedCopy`) and
  `cudaMemcpyPeer`, as upstream does. Every kernel choice (`dec_gdnf_mode`, `tc_bulk_ok`, `hc_fused_supported`,
  `pdl_kernel_ok`) is made per card, so on an all-Ada setup each card takes the same 8.9 paths as one card does;
  `fp4_tensor_cores` covers the whole card set (below). `dm_->vram` is always -1, so the TMA expert kernel is never chosen
  on any card.
- GDN split by key head (`gs_split`; taken when the peer has at least a quarter of the two cards' bandwidth, so also with
  two equal cards): CUDA0's half called `dense_tc_pair`, which exits on 8.9, and the cluster `dec_gdn_rec`. On 8.9 it now
  runs qkv and z as two `dense_proj` and `dec_gdn_rec_split` + `dec_gdn_rec_norm` (the same bits as `dec_gdn_rec`,
  gdn_window.hpp); sm_120 is unchanged. The peer's mixer item already used the split form.
- `gdn_head_split_parity` runs on 8.9 (split recurrence only, both ranks on the one card): the slices, the in-place head map
  and the peer item's q8_1 rows are bitwise. It fails one tolerance: the combined out projection's max error against the
  CPU must stay within 2x the whole layer's, and is 2.3-3.0x in 11 of 60 cases (RMS within 1.2x). That is the rounding of
  the two-part sum, the same arithmetic as on sm_120; the threshold was left as it is.
- `dec_gdn_parity` runs on 8.9 (modes 2 and 3 and the segment kernels the batch window uses there): PASS, bitwise.
  `peer_share_parity` runs with the share on the one card: ok. `ds41_peer_parity` passes with one card.
- Not run end to end, for want of a second card: the expert tier, the layer split and its overlapped windows, the prompt
  split, the GDN and head splits, and DeepSeek's second-card tier.
- Mixed Blackwell + Ada setups (a 5090 main card with a 4090 tier, or the other way round): build with
  `cmake -S . -B build-mixed -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="120;89"` (ada_build.bat builds
  89 only). The engine refuses at start a card its build has no device code for (`build_archs.hpp`, checked in
  cli/gpu.cpp and device.cu), since a 5090 would otherwise run the 8.9 PTX with its sm_90 paths trapping. FP4 prompt
  kernels run only when every engine card is 12.x (`fp4_tensor_cores`; the process sees only its own cards), so on a mixed
  set every card reads NVFP4 prompts through MMQ: the buffers, chunk plan, peer share and row rounding are the same on
  every card, and `fp4_pack()` / `fp4_every_layer()`, decided once on the main card, hold on all of them. The price is
  the 5090's FP4 prompt speed; decode does not use FP4 tensor cores. Everything else already chooses per card (decode's
  expert rows use the same arithmetic on both, so a tier's rows still do not depend on where an expert sits). Checked on
  the 4090 with the 120;89 binary only (it holds 99 sm_89 and 99 sm_120 images plus the sm_120a FP4 module): GPU suite
  116 of 124 pass, 6 skip, 2 fail (`file_expert_repin_test`, `gdn_head_split_parity`'s tolerance, both as above); output
  identical to fork_4090's 89-only build. Never run with a Blackwell card: its half of the binary is compiled, not run.
- DeepSeek-V4.1: every `ds41_*` parity test passes on 8.9; no pack has been run (none downloaded).
- The other Flash-Next formats (IQ1_S to Q8_0, the K-quants, Q2_0): their kernel parity tests pass on 8.9 (`iq_*`,
  `dec_moe_parity_kq`, `prefill_mmq_*`, `native_expert_parity_*`); only the NVFP4 pack has been run end to end.
- End to end on the NVFP4 pack, 450 W: `generate_prompt_test`, `stdin_stop_test`, `grammar_engine_test` and
  `grammar_batch_test` (`--parallel 4`) pass; vision `dynamic`, the sidecar on GPU 0 and on the CPU each read the wordmark
  in assets/basalt-lockup-light.png as "Basalt" (1.1 / 1.3 / 4.4 s); the router swaps two models both ways (about 25 s a
  swap). Greedy and sampled output with swaps off is identical to fork_4090 at cab6afa3.
- Fixed on the way, both not Ada-specific: the sidecar kept the image open until after its OK line, so the server's
  delete failed on Windows (WinError 32) and every sidecar image request was dropped; and the model-pack ctest entries were
  never registered, because python3 was looked up after them.
- Left as they are, Windows only: `file_expert_repin_test` assumes a registered copy's device alias is its host address
  (true under Linux's unified addressing), and `iq_avx512_test`'s rows differ from its frozen hashes under MSVC on Zen 5
  (max rel error against double 4e-6).
- sm_120-only tests skip below their compute capability: `hc_fused_parity`, `prefill_fp4_test`, `mid_chain_hold*`,
  `qsa_mid_parity`. The MoE experts' FP4 kernels (`src/prefill/moe_fp4.cu`, library `prefill_fp4`) are always built for
  `120a-real` and never launched on 8.9. The server does not parse gpu_select.cpp's error strings ("not sm_120 or sm_89").

## The dense stack (Qwen3.8-27B) on 8.9 (upstream 0.4.0)
- No 27B pack has been run (none downloaded; `qwen35_checkpoint_round_trip` skips without one). The pack takes 16.0 GiB
  on the card with its drafter, so it would fit a 4090. Every kernel the stack calls has an 8.9 path, checked by
  upstream's kernel tests on the 4090.
- Upstream builds `window_gemm`, `prefill_fp8`, the dense FP4 product and two tests (`narrow_bf16_parity`, `silu_parity`)
  for `120a-real` only. They build for `BASALT_CUDA_ARCHS_A` (CMakeLists.txt): `120a-real` for a 120, `89-real` for an
  89, so a 120-only build is as upstream's. The dense FP4 product is its own library, `prefill_fp4_dense`, which
  `prefill_fp4` links.
- Decode windows: `window_gemm` (cp.async, bf16 mma) runs as on 12.0. On 8.9 its NVFP4 codes decode by a byte permute
  from the eight magnitudes' patterns, and its scale multiply is `fma.rn.bf16x2` with a -0 addend (narrow_bf16.cuh): the
  same bits as 12.0a's conversions and `mul.rn.bf16x2` (`narrow_bf16_parity`: every input against the host values).
  `dec_gdn_rec` runs as `dec_gdn_rec_split` + `dec_gdn_rec_norm` there (the same bits, `state_out` included; the raw
  output in `a.o`, or else `a.y`); batch windows use `dec_gdn_rec_seg_split` on every card; `dec_gdn_rec_seg` refuses
  below 9.0.
- Prompt chunks. `gdn_gating_proj`: a flat form, a block of 8 warps per 16 tokens, warp q the cluster form's rank q (a
  copy of its loop, so the cluster kernel stays upstream's source), the partials added in rank order. FP8 W8A8
  (`fp8::gemm`): 8.9's FP8 mma keeps too few bits in its sums (1.2e-3 of sum |w a| at K 6144, 1.5e-4 with each k32
  step's sum moved into fp32, against the test's 1e-5), so the codes decode to f16 (exact) and the product runs on f16
  mma with fp32 accumulation: 1.6e-6, 134-161 TFLOP/s at 450 W. On the FP8 tensor cores the same kernel reaches 220-268
  TFLOP/s (1.65-1.85x) at 1.5e-4 even with the sums in fp32; open as item 6 of ADA_NOT_IMPLEMENTED.md (main checkout,
  untracked), with the measurements. The cause, measured with single instructions: 8.9's e4m3 mma aligns its products
  to the largest in a 14-bit window that also holds the accumulator (f16 mma: 25 bits). Only packs with F8_E4M3
  projections (NVIDIA's 27B checkpoint) call it.
- FP4 W4A4 (`fp4_dense`, 12.0's FP4 tensor cores). On 8.9 `kernel_ada` runs one e4m3 mma m16n8k16 per 16-value scale
  group from a zero accumulator, then adds the sum times the token's and the row's group scales into fp32 with one fma.
  - Why it is exact: every E2M1 code is an E4M3 value. A group's code products are multiples of 1/4 of at most 36, so
    all their bits lie within 8 bits of the largest product, inside the 14-bit window, and the sum (at most 576) is
    exact in fp32.
  - Codes become E4M3 bytes by a byte permute; each warp turns its scales into fp32 at each stage. The epilogue is the
    TMA kernel's arithmetic, and the quantizer rounds to E2M1 in software (nvfp4_planes.cu's rounding).
  - Error in fp4_dense_tma_test, of sum |w x|: plain 6.6e-8, add 6.7e-8 (the f16 form it replaced: 6.4e-8, 8.4e-8).
    At the 27B's shapes, gate (K 5120) 5.4e-8 and down (K 17408) 1.7e-8, against 3.0e-7 and 8.3e-8 before. Rows are
    bitwise at any chunk length.
  - Speed at 450 W, fp4_dense_tma_test --bench, alternated with the f16 form (ABBA, twice), TFLOP/s new (old):
    gate+up SwiGLU 213-215 (128) at 2048 tokens and 204-205 (127-128) at 4096; down 172-175 (107) and 207 (128); down
    with Add 172-176 (104-105) and 199-200 (124-125). That is 1.60-1.68x.
  - The 12.0 SASS of the file is unchanged. Without a 27B pack, prefill end to end and quality are unmeasured.
  - The sums run in another order than 12.0's block-scaled mma, so a 4090's prompt rows are not a 5090's bits.
- sm_120 code: the SASS of every upstream kernel in the touched files (dense_ops, gdn_window, window_gemm*, fp8_gemm,
  fp4_dense) is identical to upstream 0.4.0's; the 8.9 kernels are trap stubs there.
- `dense_ops_parity` fails only its old baseline, `bf16_rows_dot` + `gdn_gates` at T = 2048 (3.17e-4 against 3e-4, beta
  3.37e-5 against 3e-5): its tf32 mma accumulates below fp32 (the test's own comment), and the engine no longer calls it.
  `bf16_rows_dot` itself passes. The thresholds were left as they are.

## Upstream's own Ada peer and the rest of 0.4.1 (merged 2026-11-18)
- Upstream added a narrower Ada mode (ba7d5d72, #343: `-DPEER_ADA=ON` compiled a list of peer-tier files for sm_89).
  Here `PEER_ADA=ON` adds 89 to `CMAKE_CUDA_ARCHITECTURES` instead (the start refuses a card whose code the build lacks,
  `build_archs.hpp`). gdn_window.cu keeps the fork's `REC_NO_CLUSTER` stubs (they also cover an 89-only build's host
  pass); gu2.cuh and mmq_common.cuh take upstream's `pdl_launch_dependents` (nothing below 9.0); dec_moe.cuh's TMA kernel
  is compiled out below 9.0 by upstream's guard.
- The rest of upstream's 79 commits add no instruction 8.9 lacks: the new fp16 KV prompt attention is f16 mma m16n8k16
  and ldmatrix, the GQA decode's next-chunk fetch plain loads, the 27B's layer split and `--kv-resident` host code and
  flag kernels. The FP8 GEMM's per-device shared-memory opt-in sits beside the fork's `gemm_ada`, which already had one.
- `gqa_attn_parity` freezes the decode kernel's bits (upstream measured them on Blackwell). The 4090 gives other bits,
  the same before and after upstream's next-chunk fetch (checked by building the test with that change undone), so the
  test takes those below 9.0.
- Output: with `--kv int8` (with and without prompt lookup) 3 greedy and 2 sampled cases are identical to fork_4090's.
  With the default fp16 KV they differ from the first token on, by upstream's design: its fp16 prompt attention now rounds
  q and p to fp16 and runs one fp16 mma instead of hi + lo parts (c8f54503; upstream's gate: lossless).
- sm_120 code: in the 21 files the fork changes, every upstream kernel's SASS equals upstream's, apart from two kernels
  in verify_kernels.cu whose constant-bank address load reads offset 8 instead of 0 (the fork's `g_l2_ahead_sink` comes
  first in the module's address table; the instructions are the same).
- Speed (2 rotated pairs, engine_bench community, int8 KV, 450 W, DDR5-4800): decode_tps 197.4 against 197.9, prefill
  16K +1.8 %, 64K -3.6 %: where the adaptive swaps leave the experts after the varying sampled prose, not the prefill code
  (alone 6131 against 6103 t/s; with `--adapt-swaps 0` the same text and 6260 against 6264).

## Upstream 0.5.0: Hy4-flash (merged 2026-11-20)
- Hy4-flash and the code it shares with DeepSeek build for 8.9 as they are (f16 mma with ldmatrix and cp.async, the e4m3
  cvt, NVFP4 experts through MMQ; `attend_mma_kernel` opts in to 85,760 of 101,376 bytes). Its parity tests pass; no pack
  (158.7 GB) has been run. `file_expert_repin_test` fails on Windows (it assumes Linux's unpinned RAM copy). Output
  identical, SASS of 33 files as upstream's, A/B a tie (202.2 against 203.0, every run `ram-experts=59.5GB`).

## PCIe share of solo decode windows (experiment, 2026-11-20)
- `--exp solo_stage=1` stages a share of each layer's CPU misses over PCIe in solo windows too (`stage_frac=F`,
  `stage_min=N`; [experiment keys](docs/configuration/diagnostics.md)); off is the old path. NVFP4, 450 W, int8 KV, 8
  seeds: no share wins (0.1 to 0.35: conv / prose -2.9 to +0.8 %, struct up to +2.2 %; 0.5: -13.7 / -9.2 %); the stage
  kernel, launch and join cost 0.2-0.5 ms a pass. Staging changes the text, so retest over seeds.

## Upstream 0.5.0+58: the shared routing flags (merged 2026-11-20)
- Nothing to port (no sm_90+ feature or capability check); upstream's FP4-Local reader (#444) replaces the fork's
  (dbb54997). CPU 94/94, GPU the 3 known fails; output identical; SASS of 26 files as upstream's; A/B 201.3 against 201.3.

## Upstream 0.5.5: Inkling 2 Mini (merged 2026-11-23)
- Inkling 2 (src/models/inkling2) checks no compute capability. On 8.9 its fused window attention (a cluster of eight
  blocks) compiles out and traps (`INK_NO_CLUSTER`); `attention_window` takes the split and the combine: upstream's own
  path for longer windows, the same bits. Its prompt NVFP4 experts take MMQ below 12.x (`fp4_cores()`; moe_fp4 is
  sm_120a only). The bulk L2 prefetches in its glue and ops and in Hy4-flash's ops run a line at a time below 9.0.
- The drafter's `head_final_kernel` read the pick back from the restrict-qualified `*prev` that thread 0 stored; the
  sm_89 compiler hoists that load above the barrier (sm_120's does not), so warps copied the previous draft's w1 row.
  Below 9.0 the pick comes from shared memory; `inkling2_dspark_parity` checks the row.
- 450 W, DDR5-4800: CPU 100/100, GPU the 3 known fails; Flash-Next output identical (fp16 and int8 KV); sm_120 SASS
  of 31 files as upstream's. A/B, 2 rotated pairs: decode_tps 201.2 against 202.3 (-0.5 %, one slower run).

## Upstream 0.5.6+32: parallel slots, Inkling's CPU experts and MX8 pack (merged 2026-11-23)
- Inkling 2's two new prompt GEMMs (#575) were built for sm_120a alone: every Inkling prompt stopped at "no kernel
  image". The Q8_0 projections take `gemm_ada_kernel` below 9.0 (the TMA kernel's tiles, fragments and int32 block
  dots, loaded by cp.async; `q8_gemm_test` replays it bit for bit). The MX8 pack's MXFP8 experts (layers 20-23) run
  each K=32 block as two f16 mma.sync on the exactly decoded E4M3 codes, the UE8M0 scales on the block's sum (not the
  block-scaled mma's bits; `moe_mxfp8_test` holds). `fp4_cores()` answers per device: the new layer split (#527) may
  pair a Blackwell and an Ada card.
- CPU 104/104; GPU the 3 known fails and the new `ds41_vision_reserve_test` (Windows: WDDM lets an allocation past the
  free VRAM succeed in system memory); Flash-Next output identical (fp16 and int8 KV); sm_120 SASS as upstream's. A/B,
  2 rotated pairs, every run `ram-experts=59.5GB` and the same text: decode_tps 200.4 against 200.3, prefill a tie.
- Below 9.0 a decode window reads no experts into L2 beside the attention (6d7dc337): the guessed experts' prefetch held
  the SMs at the wrong L1/shared split and took the attention's bandwidth. Plain decoding +12 %, MX8 conv / prose +2.9 /
  +4.7 %, same text (both packs alike). Drafts accepted 56 / 97 / 40 % (5090: 68 / 97.5 / 51): deeper drafts pay here.
- Inkling 2 Mini, every expert on the card, 450 W, DDR5-4800. Gate: upstream's `inkling2-bf16-base.kld`, 100 chunks,
  T=1, scored-row base PPL 7.0633. Speed: basalt-bench seed 1, two runs (fp16, int8, int8, fp16), tok/s; ms/tok is plain
  decoding (`--spec 1`) after the 16K prompt. Context: the largest `--max-context` that keeps every expert on the card
  (CLI, one session, the card otherwise empty; it moves by about 100 between starts).

  | Pack, KV | KLD     | top-1 % | PPL    | conv | struct | prose | 16K   | 64K   | ms/tok | context |
  |----------|--------:|--------:|-------:|-----:|-------:|------:|------:|------:|-------:|--------:|
  | MX8 fp16 | 0.00297 |   97.69 | 7.0727 |  219 |    470 |   184 | 8,597 | 7,444 |   7.17 |  84,572 |
  | MX8 int8 | 0.00346 |   97.51 | 7.0742 |  223 |    476 |   194 | 8,566 | 7,309 |   6.86 | 141,673 |
  | DQ8 fp16 | 0.00298 |   97.70 | 7.0733 |  228 |    450 |   186 | 8,778 | 7,587 |   7.17 |  81,158 |
  | DQ8 int8 | 0.00346 |   97.47 | 7.0746 |  222 |    479 |   186 | 8,739 | 7,442 |   6.87 | 136,085 |

- A 5090 (400 W, MX8): conv 347, struct 628, prose 296, 16K 15,820; gate at T=3 (base 7.0644) MX8 0.00297 /
  97.83 %, DQ8 0.00304 / 97.81 % / 7.0737, int8 0.00343 / 97.47 % / 7.0788. int8 reads 288 against 512 MiB of cache a
  token at 16K and fits 1.68x the context; a short-context window costs the same (DQ8), so conv / struct / prose follow
  int8's other texts. DQ8 reads prompts 2 % faster here (upstream: MX8 5 %; likely 8.9's MXFP8 on f16 mma). Just above
  each context limit about 35 MiB refused to start instead of moving experts to RAM (#542; fixed by #643, below).

## Upstream 0.5.7+39: gpu-mode for several cards, Inkling's tensor-core attention (merged 2026-11-24)
- `gpu-mode` (peer, split) and `gpu-share` replace each model's multi-card logic; upstream takes an sm_89 card only as a
  Flash-Next peer (`-DPEER_ADA=ON`), here every card may be sm_120 or sm_89 in every mode (gpu_select.cpp).
- Inkling's tensor-core attention (#626, #606) is f16 mma and ldmatrix, sized by the card's opt-in limits; the cluster
  attention's `cluster_device()`, `fp4_cores()` and `side_reads` are kept. No new arch guard, bulk copy or 120a library.
- Tests frozen on Blackwell take 8.9 values (ad46f9ef): the INT8 KV graph's PDL launches, `gqa_window_parity` (the
  one-tile kernel's own 8.9 bits) and Inkling's full-attention hashes. `prefill_mmq_mxfp4_test` (new) fails IQ2_XS /
  IQ2_S gate/up (2.96e-4 against 1e-4) the same at every planning SM count: not 8.9's.
- CPU 111/111, GPU the 4 known fails; Flash-Next output identical (fp16, int8 KV); sm_120 SASS of 35 files as upstream's.
  A/B 4 pairs, `ram-experts=59.5GB`, same text: decode_tps 200.2 against 200.1, conv / struct / prose +0.3 / -0.1 /
  -0.4 %, prefill a tie (prose read -3.6 % in the first 2 pairs, +3.1 % in the next 2: a fixture swings ~4 % a run).
- Inkling (#579 changes its bits by design): gate MX8 0.00296 / 97.73 %, DQ8 0.00296 / 97.73 %, int8 0.00343 / 97.45
  and 0.00345 / 97.40 %; per-window costs level or lower, MX8 struct -6.6 % from another greedy text. With #643 every
  expert stays on the card up to MX8 80,918 / 135,718, DQ8 77,500 / 130,129 (fp16 / int8); above, experts move to RAM.

## When a new build fails on the card
An illegal-instruction or launch failure means a `__trap()` stub was launched: rerun with `CUDA_LAUNCH_BLOCKING=1`
(and `compute-sanitizer`) to find the kernel, then the runtime predicate that should have excluded it.
