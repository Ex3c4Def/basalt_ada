# Plan: the mid chain on sm_89 (RTX 4090), local fork only

Status: the kernel is the default on 8.9 and bitwise equal to the separate kernels (`--exp mid_ada=0`). Development
A/B at 450 W: +2.2 % decode (token-weighted) over `mid_ada=0`, faster on all three prompts. `--exp mid_ada=2` runs it
with z on dense_tc (its first version, other bits than `mid_ada=0`).

## The kernel
`src/kernels/cuda/mid_chain_ada.cu`: one persistent kernel, one CTA of four 128-thread groups per SM (128 CTAs on
the 4090), runs record_gdn.cpp's unfused sequence of a GDN layer's middle with those kernels' bodies or arithmetic:
- q8_1 of the read's rows (native_quantize_q8_1's arithmetic);
- qkv through `dense_tc_block` (three groups), alpha/beta on the fourth;
- z in native_mmvq's per-row arithmetic on every warp after that: a warp takes two rows, each lane doing the work of
  the four native threads it stands for in turn, the two rows sharing the activation loads (`mid_ada=2`: z through
  `dense_tc_block` after qkv on the three groups);
- conv + L2 norm;
- recurrence + gated norm per value head, then the head's q8_1 blocks;
- the output projection through `dense_tc_block`.

Four grid barriers with bounded waits and the abort word. The bodies are in `gdn_unfused.cuh`; `dense_tc_block` takes a
thread index and a group barrier (the existing kernels' SASS byte-identical on sm_89 and sm_120). native_mmvq.cu is
built with fast math, so the parts taken from it use its flush-to-zero operations explicitly. Only the all-Q8_0 and
all-Q6_K triples (the pack's GDN projections are Q6_K); `rec.yq` must lie apart from `xq`.

`mid_chain_parity` on 8.9: bitwise against the engine's unfused sequence, every output, T 1..8, grids 128 / 64 / 20,
both triples; the z-on-dense_tc form likewise against that sequence with z on dense_tc. Microbench, graph of 48
layers, us per layer, 450 W (development A/B, not a headline number); "z dtc" is the kernel with z on dense_tc:

| Q6_K | engine seq. cold | mid chain cold | z dtc cold | engine seq. hot | mid chain hot | z dtc hot |
|---|---|---|---|---|---|---|
| T=1 | 70.4 | 64.8 (92.1 %) | 65.6 | 42.0 | 35.9 (85.4 %) | 37.9 |
| T=2 | 71.8 | 66.9 (93.1 %) | 67.2 | 45.6 | 38.1 (83.4 %) | 39.0 |
| T=4 | 75.5 | 70.0 (92.7 %) | 69.0 | 54.3 | 42.7 (78.6 %) | 42.1 |
| T=8 | 91.2 | 79.2 (86.8 %) | 75.1 | 74.0 | 56.4 (76.2 %) | 51.1 |

## In the engine
record_gdn.cpp's unfused path takes the kernel on 8.9 (single GPU, not a batch window, no PLE or lazy branch open, not
profiling), with its own `mid_yq_` buffer, after a join on the L2 read-ahead (the kernel needs every SM). `--exp
mid_ada=0` runs the separate kernels, `--exp mid_ada=2` the kernel with z on dense_tc. Greedy, 400 tokens, conv /
struct / prose: identical to `mid_ada=0`; `mid_ada=2` differs on conv and prose at the tokens where the first version
did.

basalt-bench A/B, 450 W, 4 rounds x 2 repetitions, modes rotated in one run (development A/B, tok/s; "n of 8": paired
runs where the left mode is faster). "z on dense_tc" is `mid_ada=2`, measured before it was a flag through a
temporary switch with the same code:

| prompt | mid_ada=0 | kernel | z on dense_tc | kernel vs 0 | z on dense_tc vs 0 |
|---|---|---|---|---|---|
| conv | 114.9 | 117.8 | 111.7 | +2.5 %, 7 of 8 | -2.8 %, 3 of 8 |
| struct | 204.2 | 209.7 | 209.9 | +2.7 %, 8 of 8 | +2.8 %, 7 of 8 |
| prose | 118.4 | 121.8 | 121.6 | +2.9 %, 7 of 8 | +2.7 %, 7 of 8 |
| token-weighted | 173.4 | 177.2 | 176.4 | +2.2 % | +1.7 % |

Window time at `mid_ada=0`'s mix of window widths (the bench's `verify_ms_by_t`, independent of acceptance): kernel
-0.6 / -2.9 / -2.3 % on conv / struct / prose, z on dense_tc +3.3 / -2.9 / -1.5 %. A z phase of one row per warp (no
shared activation loads) measured -0.4 % token-weighted in its own A/B. The two z forms are equally fast in the
engine (struct -0.1 %, prose +0.2 %); conv's gap is another text: z on dense_tc changes conv's output from the first
repetition, and that text's experts hit VRAM less (83.3 % against 85.7 %), so windows wait longer on the CPU tier.

The join on the read-ahead costs no wait: in Nsight Systems traces the read-ahead ends a median 31 us (prose) and 285
us (struct) before the kernel starts. Two A/Bs of the first version with and without the join gave +2.1 % and +1.8 %
token-weighted with it; without it each A/B had a slow run.

Found on the way: `launch_dependent` gave the grouped chain's SwiGLU and down the programmatic attribute although sm_89
has no griddepcontrol; each edge waited 3 us, 25 to 42 us after a fork/join (every QSA layer). It launches plain below
9.0 now (41ba76cc): +3.25 % decode with `mid_ada=0`.

## Measuring on this card
A large part of each layer is the combine waiting for the CPU expert tier (a median 150 to 300 us per layer in traces),
so the GDN middle's savings (7 to 12 us per GDN layer in place) show as a few percent. Expert swaps (`--adapt-swaps`)
land at timing-dependent moments and CPU and GPU experts are not bitwise equal, so two runs' token streams differ even
with identical kernels: compare modes within one run, per window width where acceptance differs. With swaps off decode
is CPU-bound (struct 63 tok/s) and hides GPU changes.

## The hc read's methods, tried
The hc read kernel (HC_READ_ADA_PLAN.md) gained from staging, tagged hand-offs and fewer barriers. On this kernel they did
not pay, so it stays as it is. Numbers at 450 W, development only.

Where the time goes (Q6_K, T = 2, us after the first CTA's start, median CTA; timer ticks 1 us):

| | q81 + barrier | qkv + z | barrier | conv + barrier | rec + barrier | out |
|---|---|---|---|---|---|---|
| hot, 38 us | 2 to 3 | to 17 | to 20 | to 22.5 | to 27.5 | to 37 |
| cold, 67 us | 2 | to 39 | to 42 | to 45 | to 52 | to 66.6 |

The weights are 47 MB, and they set the time. Hot, qkv + z move 34 MB at about 2 TB/s, near the card's read limit. Cold, the
whole kernel runs at the DRAM's limit of about 0.93 TB/s. The output projection is the slowest part when hot, about 1.3 TB/s,
since only 160 of its 384 groups have tiles. The barriers cost 1 to 2 us each.

Each candidate on its own, keeping the bits, us per layer (T = 2, cold / hot; base 67.1 / 38.0):

| candidate | result | |
|---|---|---|
| barrier as a count alone, no generation word | 65.4 / 36.4 | -2 us at every T |
| + qkv's first slots issued before q81 | 66.4 / 36.9 | worse: threads stall on `cp.async`, so q81 ends later |
| + out's first slots issued before conv | 65.9 / 36.8 | no gain |
| + out's ring of 6 slots | 66.9 / 36.1 | cold worse, hot about equal |
| + out's ring of 6 slots issued before conv | 67.5 / 36.8 | out 4 us faster cold, conv 6 us slower |

Issuing early moves the stall onto the critical path. A `cp.async` stalls its thread while the queue is full, and every
thread is needed next. This agrees with the earlier in-layer prefetch measurement under Risks.

The count barrier passed `mid_chain_parity` (0 differing bytes) but gave no gain in the engine. basalt-bench A/B against the
current barrier, 4 rounds x 2 repetitions, video decoder idle throughout: conv -0.6 %, struct -0.8 %, prose -0.3 %,
token-weighted -0.6 %, 8 of 24 paired runs faster, window time -0.5 / +0.8 / -0.2 %. That is within the spread, and 2 us
per layer is at most a few tenths of a percent of a window. So it was not kept.

What matters more in the engine is whether the read-ahead finished. Traces, kernel median by T = 1 / 2 / 4 / 8: 39.5 / 42.5 /
47 / 60 us after a completed read-ahead, 69 / 72 / 77 / 87 us after none. Layers with less than 40 us of read-ahead: prose 48
%, conv 24 %, struct 12 %. Their time over the hot median is 2.3 / 0.8 / 0.2 % of the cycle, which bounds what a perfect
read-ahead could gain. Most of them have little or no CPU wait, so the read-ahead has no slack to run in, and the 47 MB then
pass through DRAM on the critical path.

## Next steps
1. Remove the `--exp mid_ada` switch once A/Bs against the separate kernels are no longer needed.

## Risks
A CTA that is not resident stalls the chain until its waits give up (200 ms bound, abort word, the window fails). The
earlier in-layer prefetch measurement (beside conv/ab/rec, every variant 0.03 to 0.14 ms per window slower) still argues
against overlapping the output projection's weight reads with the recurrence.
