# Plan: the hc read on sm_89 (RTX 4090), local fork only

Status: the kernel is the default on 8.9 and bitwise equal to hc8_gr's three kernels (`--exp hc_ada=0`). Development A/B
at 450 W, per-window log on, a download running: +3.17 % decode (token-weighted) over `hc_ada=0`, faster on all three
prompts. Together with the route kernels on `engine_bench.py`: ADA_PORT_NOTES.md.

## Where the time went
Numbers at 450 W, GPU otherwise idle, development only.

- hc8_down read all of w_norm and every token's R on every SM: 128 x 40 KB x (T + 1), 47 MB at T = 8, for 3.3 MB of
  weights. Reads into the SMs top out near 1 to 1.5 TB/s on this card, so down took about 29 us at T = 8.
- hc8_up was limited by instructions: 2560 warp columns, each running its epilogues on one lane at a time.
- A kernel launch costs about 1.4 us, so merging the three launches alone would gain almost nothing.
- A floor remains: the read's weights are 7.1 MB (codes and scales of down, inject and up), about 5 to 7 us.

## The kernel
`src/kernels/cuda/hc_read_ada.cu`: one persistent kernel, one CTA of 512 threads per SM, 128 CTAs on the 4090. The first
256 threads serve the down rows, the other 256 the norm, the staging and the sums. hc8_gr's arithmetic and reduction tree
are kept exactly.

- **Classes of columns.** A warp of hc8_down covers the scale groups g and g + 160 of its lanes, and its butterfly merges
  lanes by their high bits first, so lanes with the same low bits form a subtree. Five warps x four subtrees give 20
  classes. A CTA serves one class: it stages that class's part of w_norm, R and bo with `cp.async` (2 KB per array), forms
  R' in shared memory, and each lane of a warp takes a row pair. An SM reads a twentieth of those arrays, however many
  rows it serves. Splits of 2 and 8 measured slower than 4.
- **R stays old until the barrier.** Every reader forms R' itself, so the norm runs beside the staging on the same CTAs,
  without a barrier before down. A CTA writes its columns' R' (float4 stores) after the barrier.
- **Tagged hand-offs.** rs and the subtrees' sums are 64-bit words carrying the read's number (the barrier count over the
  grid), usable the moment they land. A row's sum lane loads all 20 words in parallel and waits alone on a missing one;
  polling them in turn cost 5 to 9 us.
- **One grid barrier**, before up: every CTA's lo is written and no CTA reads R any more. An atomic count, in its own
  allocation.
- **Up.** A CTA takes a contiguous run of columns. Its weights and the epilogues' R, bo and w_norm are staged from the
  CTA's start, so they overlap down. Dots run per warp as in hc8_up; then the epilogues run a lane per (column, token)
  from shared memory, which removes 140 single-lane instructions per column.

The wait bound is 200 ms with the abort word (`kWaitHcRead`). The barrier's count stays a multiple of the grid, so a test
that changes the grid goes through `hc_read_ada_test_grid`, which clears the count and the tags.

## Microbench
`hc_read_ada_parity`: 0 differing bytes against `hc8_gr_read_multi`, every output, T 1..8 x 20, grids 128 / 64 / 20, with
and without the pending write and the inject rows, R in place and apart. Time per read, graph of 48 reads, us, 450 W
(development A/B, not a headline number). Cold: 24 sets of matrices, about 170 MB, more than L2 holds. Hot: one set.

| T | three kernels cold | kernel cold | three kernels hot | kernel hot |
|---|---|---|---|---|
| 1 | 21.8 | 16.1 (74 %) | 19.5 | 12.4 (64 %) |
| 2 | 25.2 | 17.7 (70 %) | 23.4 | 13.4 (57 %) |
| 3 | 28.4 | 18.0 (63 %) | 27.4 | 14.7 (54 %) |
| 4 | 32.9 | 19.4 (59 %) | 31.6 | 16.5 (52 %) |
| 5 | 37.1 | 20.7 (56 %) | 35.4 | 18.5 (52 %) |
| 6 | 41.5 | 21.8 (53 %) | 40.0 | 19.9 (50 %) |
| 7 | 45.8 | 24.5 (54 %) | 44.4 | 22.5 (51 %) |
| 8 | 49.9 | 26.2 (52 %) | 48.8 | 24.7 (51 %) |

Phases at T = 1, hot, us after the first CTA's start (timer ticks 1 us, stamps add fences): tile 2, R' 3, down 5, sums 6,
barrier 8, lo 9, up 10, epilogues 11.

## In the engine
`record_layer.cpp`'s `gr_read_group` and `record_head.cpp` take the kernel on 8.9 when no side branch is open (not a batch
window, not profiling, no PLE or lazy branch), after a join on the L2 read-ahead. So layer 0's FFN read, beside the PLE
branch, stays on the three kernels, and so do the draft graphs. `--exp hc_ada=0` runs the three kernels everywhere.

Greedy, 400 tokens, conv / struct / prose: identical to `hc_ada=0`.

basalt-bench A/B, 450 W, 4 rounds x 2 repetitions, modes rotated in one run (development A/B, tok/s; "n of 8": paired runs
where the kernel is faster):

| prompt | hc_ada=0 | kernel | gain |
|---|---|---|---|
| conv | 110.2 | 112.2 | +1.8 %, 6 of 8 |
| struct | 196.3 | 204.6 | +4.2 %, 7 of 8 |
| prose | 115.1 | 117.6 | +2.2 %, 5 of 8 |
| token-weighted | 166.9 | 172.2 | +3.17 % |

Window time at `hc_ada=0`'s mix of window widths: -2.3 / -5.1 / -3.3 % on conv / struct / prose. Expert hit rates are
equal within 0.4 points. An earlier A/B of the kernel, with up's weights staged later: +3.88 % (2 rounds).

Nsight traces at 450 W: a read takes a median 15.6 us at T = 1, 18.7 at T = 3, 24 to 34 at T = 8 (before: 21 and 53). The
reads are 7.0 / 7.0 / 9.4 % of the cycle on conv / struct / prose (before: 11.7 / 12.3 / 15.3 % of a verify window). Reads
over 40 us occur only in a run's first windows (the first holds prefill) with no kernel beside them.

## Tried and left out
- **The router in the kernel** (the MMVF GEMV and the top-10 after a second barrier, weights staged during the read,
  bitwise, 0 differing bytes). In the microbench the two kernels add about 3.9 us at T = 1 hot and the kernel's router about
  6 us. Engine A/B at 450 W, 2 rounds x 2 repetitions: +0.28 % token-weighted, window time -1.6 / +0.2 / +0.6 %. Within
  the spread, so the extra 200 lines are not kept. In traces the GEMV takes 4 to 7 us and the top-10 about 4.5 us;
  `dm_route_kernel` (10 to 20 us per launch) is the larger part of what follows the read.
- **The q8_1 rows in the read.** A CTA's run of columns does not hold whole q8_1 blocks, and runs of whole blocks would
  leave SMs idle in up. The mid chain's first phase already does them.
- **The attention read's matrices in the L2 read-ahead.** A/B at 450 W, 3 rounds x 2: +0.22 % token-weighted, window time
  -0.4 / -0.6 / +0.6 %. Left out.
- **The draft graphs.** Their reads are 1.1 to 1.5 % of the cycle, one row at a time, with a prefetch branch open beside
  the attention read. At most about 0.5 %, so they keep the three kernels.

## Next steps
1. Remove the `--exp hc_ada` switch once A/Bs against the three kernels are no longer needed.

The route after the read has its own kernels now (`dm_plan_ada_kernel`, `dm_ring_ada_kernel`; see ADA_PORT_NOTES.md). These
methods on `k_mid_ada` gave no gain in the engine; see ADA_MID_CHAIN_PLAN.md.

## Risks
Like the mid chain, a CTA that is not resident stalls the kernel until the wait bound (200 ms, abort word, the window
fails), so no caller may run it beside a side branch. After an abort the barrier's count is off; the host runs no further
window then.
