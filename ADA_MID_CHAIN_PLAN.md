# Plan: the mid chain on sm_89 (RTX 4090), local fork only

Status: step 1 done, step 2 first cut in the engine behind `--exp mid_ada=1` (+1.3 % decode vs its reference).
The user decides how to go on. Each step below is one session that ends in a commit with its test.

## Step 1, done: the kernel and its parity arm
`src/kernels/cuda/mid_chain_ada.cu`: one persistent kernel, one CTA of four 128-thread groups per SM (128 CTAs on
the 4090), runs the unfused Ada sequence of a GDN layer's middle with those kernels' own bodies:
- q8_1 of the read's rows;
- qkv + z through `dense_tc_block` (three groups), alpha/beta on the fourth;
- conv + L2 norm;
- recurrence + gated norm per value head, then the head's q8_1 blocks;
- the output projection through `dense_tc_block`.

The phases hand over through four grid barriers (bounded waits, abort word). The bodies moved into
`gdn_unfused.cuh` (conv, alpha/beta, recurrence), and `dense_tc_block` takes a thread index and a group barrier. The
existing kernels' SASS is byte-identical to before on sm_89 and sm_120 (verify_kernels, dense_tc, dec_gdn_rec,
mid_chain). `mid_chain_supported` is true on 8.9 only after `mid_chain_ada_enable(true)`; lifecycle.cpp sets it from
`--exp mid_ada=1`. Only the all-Q8_0 and all-Q6_K triples; `rec.yq` must lie apart from `xq` (the output projection's
non-coherent loads would hit L1 lines the pair cached).

`mid_chain_parity` on 8.9: bitwise against the unfused sequence (z on dense_tc), every output, T 1..8, grids 128 / 64 /
20, both triples. Microbench, graph of 48 layers, us per layer, 450 W (development A/B, not a headline number):

| Q8_0 | engine seq. cold | mid chain cold | engine seq. hot | mid chain hot | pair: 4 kernels / mid, cold |
|---|---|---|---|---|---|
| T=1 | 83.9 | 79.9 (95.3%) | 37.1 | 34.8 (93.7%) | 56.0 / 52.2 |
| T=2 | 83.8 | 81.1 (96.8%) | 40.2 | 38.0 (94.7%) | 56.3 / 52.3 |
| T=4 | 87.7 | 83.2 (94.9%) | 48.0 | 42.0 (87.5%) | 57.5 / 52.3 |
| T=8 | 96.5 | 89.2 (92.4%) | 66.0 | 50.4 (76.4%) | 61.3 / 52.3 |

"Engine seq." is what the engine runs today (its z projection on native_mmvq). Over 36 GDN layers that is 0.1 to 0.27
ms per window cold, up to 0.56 ms hot, about 0.3 to 1 % of a window. The old step 2 gate (the pair phase faster than its
launches) is met: 85 to 93 % cold.

## Step 2, first cut: in the engine behind `--exp mid_ada=1`
record_gdn.cpp's unfused path takes the kernel on 8.9 (single GPU, not a batch window, no PLE or lazy branch open),
with its own `mid_yq_` buffer. `--exp mid_ada=2` is its reference in the engine: the unfused sequence with z in
dense_tc's order. Greedy, 400 tokens, conv / struct / prose: mid_ada=1 identical to 2 on all three; mid_ada=0 (z on
native_mmvq) diverges at tokens 238 and 157 on conv and prose.

The L2 read-ahead is joined before the kernel, so the kernel has every SM. In Nsight Systems traces the read-ahead ends
a median 31 us (prose) and 285 us (struct) before the kernel starts and never later, so the join adds no wait; the
gate/up to SwiGLU gap is 0.06 us with and without it.

basalt-bench A/B, 450 W, 3 rounds x 2 repetitions, modes rotated (development A/B, tok/s; measured while the grouped
chain still had programmatic edges, see below):

| prompt | mid_ada=0 (today) | mid_ada=2 (reference) | mid_ada=1 (kernel) | 1 vs 2 | 1 vs 0 |
|---|---|---|---|---|---|
| conv | 110.2 | 105.1 | 106.2 | +1.0 %, 4 of 6 faster | -3.6 %, 1 of 6 |
| struct | 198.6 | 200.9 | 204.4 | +1.7 %, 4 of 6 | +2.9 %, 5 of 6 |
| prose | 114.2 | 113.9 | 115.0 | +1.0 %, 4 of 6 | +0.7 %, 4 of 6 |
| token-weighted | 167.3 | 167.0 | 169.2 | +1.3 % | +1.1 % |

Acceptance varies between repetitions of one mode (sampling at temperature 0.6), so single prompts are noisy; struct
(98 % acceptance) is the steady one. Conv is faster on today's path in both A/Bs; whether that is z's numerics
(acceptance) or the z kernel's speed is open.

Found on the way: `launch_dependent` gave the grouped chain's SwiGLU and down the programmatic attribute although sm_89
has no griddepcontrol. Each edge waited 3 us, and 25 to 42 us once a fork/join had moved the chain to another of the
graph's internal streams (every QSA layer, layer 1, and the read-ahead join above). It now launches plain below 9.0:
+3.25 % decode with mid_ada=0 (450 W, development A/B). The mid_ada A/B above is not re-run since.

## Numerics on integration
The engine's Ada path runs the z projection on native_mmvq (dense_tc_mask has no bit 2); the mid chain runs it in
dense_tc's order. Taking the kernel changes z's bits: the quality gate (`basalt-bench --quality`) and
`docs/precision/` apply then.

## Next steps (revised order: measure in the engine before refining)
2. Integration: done behind `--exp mid_ada=1` (above). Left: the quality gate (`basalt-bench --quality`, z's
   numerics change), then making it the default on 8.9 and removing the gate.
3. Only if step 2 shows a gain worth more: finer hand-offs instead of grid barriers (per-head counters; each rec CTA
   computing its heads' conv itself; out tiles starting as heads finish).
4. ADA_PORT_NOTES.md and the precision page; remove the `--exp` gate once it resolves.

## Risks
A CTA that is not resident stalls the chain until its waits give up (200 ms bound, abort word, the window fails). The
earlier in-layer prefetch measurement (beside conv/ab/rec, every variant 0.03 to 0.14 ms per window slower) still argues
against overlapping the output projection's weight reads with the recurrence.
