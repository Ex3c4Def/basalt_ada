# Ada (RTX 4090, sm_89) port of Basalt: briefing for the agent that builds and tests it

## Goal
Make Basalt build for compute capability 8.9 (RTX 4090) and run the NVFP4 pack on that card through fallback kernels.
Upstream Basalt is Blackwell-only (sm_120). The user asked for this port in their **local fork only**: AGENTS.md /
CONTRIBUTING.md rules against other architectures and the parity-test requirements do not apply here, but sm_120 behaviour
must stay exactly as it was. Do not git commit, push or delete anything unless the user asks.

## State
The source edits are already applied (14 files, listed below). They were written **without a compiler or a GPU**, so
nothing has been built or run yet. The first job is to get it to compile and then to run.

Changed files:
CMakeLists.txt, include/basalt/models/qwen4exp/kernels/dec_gdn.hpp, src/core/device.cu, src/models/qwen4exp/kernels/dec_gdn_rec.cu,
src/models/qwen4exp/kernels/dec_moe.cuh, src/models/qwen4exp/kernels/dense_tc.cu, src/kernels/cuda/gu2.cuh, src/models/qwen4exp/kernels/hc_fused.cu,
src/models/qwen4exp/kernels/verify_kernels.cu, src/models/qwen4exp/prefill/experts.hpp, src/models/qwen4exp/prefill/layout.cpp, src/models/qwen4exp/prefill/layout.hpp,
src/prefill/mmq/mmq_common.cuh, src/program/gpu_select.cpp. (`git diff` shows all of them.)

## Build (Windows, docs/platforms.md has the details)
Needs Visual Studio 2022 C++ tools, CUDA 13.x, CMake 3.24+, Ninja. The desktop app does not load vcvars, so use
`ada_build.bat nopause` in this folder (it finds Visual Studio, calls vcvars64.bat, configures and builds with
`-DCMAKE_CUDA_ARCHITECTURES=89 -G Ninja` into `build-ada`, keeps going past errors with `ninja -k 0`, and writes
everything to `ada_build.log`; the last lines say BUILD OK / BUILD FAILED / CONFIGURE FAILED).
Read the first errors in `ada_build.log`, fix, rerun.

## What sm_89 does not have (so what any compile error will be about)
Thread-block clusters and distributed shared memory, TMA / `cp.async.bulk` and mbarrier `expect_tx`/`try_wait`,
`griddepcontrol` (PDL), `fence.proxy.async` (all sm_90+), and FP4 block-scaled MMA / `cvt e2m1` (sm_120a). It does have
int8 `mma.sync`, `cp.async`, and 99 KB opt-in shared memory per block.
nvcc here is whole-program (no rdc), so unreachable `__device__` helpers are dropped; only code reachable from a
compiled kernel must compile for sm_89.
Pattern used: wrap sm_90+ instructions in `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900` (or a local macro
`HC_SM90_ONLY` / `DM_SM90_ONLY` / `REC_NO_CLUSTER`), let the kernel trap below that, and make a **runtime** predicate pick
the fallback so the trap is never launched.

## How the fallbacks are chosen on the 4090
- GDN recurrence: `dec_gdnf_mode()` (dec_gdn_rec.cu, now a runtime function) returns 0 below sm_90, so
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
  2560 cases against `dm_route_kernel`; greedy identical on conv, struct and prose. All numbers at 450 W, development only.
  Traces: chain 8-10 -> 5 us at T=1, 18-21 -> 7-9 us at T=8; ring 12-14 -> 8 us at T=1, 17-21 -> 12-14 us at T=8; prose's
  route busy time 7.6 -> 4.3 % of the cycle. Microbench, to the doorbell: 4.7 -> 3.3 us at T=1, 15.4 -> 6.2 us at T=8.
  A/B, 4 rounds x 2 repetitions with the video decoder idle throughout: conv -0.1 %, struct -0.7 %, prose +2.5 % (7 of 8
  faster), token-weighted +0.05 %. All three A/Bs pooled, 20 pairs each, paired median: conv +0.3 %, struct +0.5 %, prose
  +2.8 %, with prose's window time -3.3 %. So only prose gains beyond the spread; expert hit rates are unchanged. Other
  load on the machine moves these numbers: in an A/B during which a video played at some time, one conv run lost 21.5 %
  across every window width. Log `utilization.decoder` beside an A/B to see this.
- HC read and route together, against the build without them: `local_bench/engine_bench.py --tests community`, a fresh
  server from `basalt_config.json`, 450 W, DDR5-4400, 4 rotated pairs, development only. conv / struct / prose /
  decode_tps 111.2 / 200.3 / 120.4 / 170.2 -> 118.7 / 212.3 / 127.3 / 180.1, +5.8 %, faster in every pair. decode_tps
  ranges 163.1 to 175.2 and 178.0 to 182.5. Same builds, plain basalt-bench with the log on, 14 pairs: +5.53 %; the hc
  read alone +3.53 % and the route on top +1.40 %, 6 pairs each. Text differs between runs on conv and prose (adaptive
  swaps), so struct, greedy, is the cleanest figure.
- Those builds' output: the quality gate's rows (100 chunks, T 3) are byte-identical; greedy, and with `--adapt-swaps 0`
  sampled conv and prose, are identical.
- Measuring: plain basalt-bench asks for the per-window log unless `--no-windows`, about 2 % slower, so compare its tok/s
  only with its own. It keeps the last of a repeated engine flag, so two experiment gates go in one `--exp a=0,b=0`.
- Comparing the quality gate across builds: its rows follow the GPU slots, and a bigger binary leaves less VRAM (6491 slots
  against 6515 changed nearly every row, KLD 0.0620 against 0.0619). Fix `--expert-cache` when comparing; with it equal the
  rows are byte-identical.
- DeepSeek-V4.1 on 8.9: its code has no architecture guards, and it uses no clusters, bulk copies, griddepcontrol or FP4
  tensor cores; DSpark decodes MXFP4 in software. Its `ds41_*` parity tests (`-DBUILD_TESTING=ON`) pass on a 4090 at 450 W,
  the peer test only with one card. No pack has been run, so end-to-end output, quality and speed are unmeasured.
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
  dec_gdn.hpp); sm_120 is unchanged. The peer's mixer item already used the split form.
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
  `qsa_mid_parity`. `src/models/qwen4exp/prefill/fp4/moe_fp4.cu` is always built for `120a-real` and never launched on
  8.9. The server does not parse gpu_select.cpp's error strings ("not sm_120 or sm_89").

## Test once it builds
1. Run the built `basalt` (see README.md for the command line; the binaries land in `build-ada`) on the user's NVFP4
   pack. Ask the user for the model path and any flags. Start with a short prompt, greedy.
2. Pass criteria: the GPU is accepted as sm_89, the model loads, prefill and decode finish, the text is coherent, and
   there is no CUDA error. An illegal-instruction / launch-failure error means a `__trap()` stub was launched: rerun
   with `CUDA_LAUNCH_BLOCKING=1` (and `compute-sanitizer` if available) to find the kernel, then find which runtime
   predicate should have excluded it.
3. Cross-check against the other engine in `C:\dev\llm_engine\strata-nvfp4` (same model, Ada-capable) for sanity,
   not for bit equality. Its non-RTX-50 prefill mode is `STRATA_PREFILL_NVFP4=w4a8`.
4. Report: what built, which fallbacks ran, tokens per second for prefill and decode, and any errors.
