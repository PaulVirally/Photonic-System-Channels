#!/usr/bin/env julia
"""
    bench/parity_cost_model.jl

Dump every prediction `bench/cost_model.jl` makes, over a grid of points, as plain
text on stdout. Diff two dumps to see whether a change to the model moved anything.

    julia bench/parity_cost_model.jl > after.txt
    git stash && julia bench/parity_cost_model.jl > before.txt && git stash pop
    diff before.txt after.txt

# Why this exists

The model has two kinds of change. One deliberately moves a prediction: a refit, a
new coefficient, a corrected count. The other is supposed to move nothing: adding a
*mode* whose default reproduces the old behaviour, threading a keyword through,
splitting a function in two. The second kind is the dangerous one, because there is
nothing to look at afterwards -- a coefficient that silently stopped being applied
looks exactly like a coefficient that was never wrong.

So: 36k predictions across every cluster, geometry, scale, separation, rank,
positive count, card capacity and job kind that the real sweeps use, printed with
enough digits that a change in the tenth significant figure shows up. `diff` is the
assertion.

Nothing here is a *correctness* check. It says only that two versions agree, which
is the whole point: run it before and after a change that is meant to be neutral,
and the diff has to be empty.

# What is in the grid

Everything that selects a branch of the model, on purpose:

  * three calibrated clusters, so a coefficient file that stopped loading shows up
  * the six production geometries, isotropic and anisotropic
  * contact and eight separations out to 50000 wavelengths, which is where the old
    estimator's bounding-box bug lived
  * four ranks and three `num_pos` (including `nothing`, the estimated case)
  * four device capacities, which is what makes `rsvd_mode` / `bounds_mode` return
    `:dense_exact`, `:in_memory` and `:panel`
  * all three job kinds, padded, with the mode printed alongside

Padded rather than raw, because the padded number is the one that reaches `sbatch`.
"""

include(joinpath(@__DIR__, "cost_model.jl"))
using .CostModel
using Printf

const CLUSTERS = ("narval", "molering", "fir")
const GEOMETRIES = ((8, 8, 8), (16, 16, 16), (24, 24, 24), (32, 32, 32),
                    (64, 32, 32), (128, 32, 32))
const SCALES = (1 // 32, -1 // 8)
const SEPARATIONS = (0 // 1, 1 // 32, 1 // 2, 1 // 1, 10 // 1, 1000 // 1, 50000 // 1)
const RANKS = (400, 800, 1350, 4000)
const NUM_POS = (nothing, 900, 1832)
const CAPACITIES = (nothing, 20 * 2^30, 40 * 2^30, 80 * 2^30)

function main()
    load_coefficients!(@__DIR__)
    n = 0
    for cluster in CLUSTERS
        coeffs = coefficients_for(cluster)
        for cells in GEOMETRIES, scale in SCALES, sep in SEPARATIONS,
            rank in RANKS, num_pos in NUM_POS

            scl = scale < 0 ? (1 // 32, abs(scale), abs(scale)) : (scale, scale, scale)
            pt = SRPoint(cells, cells; scale=scl, separation=sep, rank=rank,
                         oversamples=50, power_iters=14, threads=4, num_pos=num_pos)
            for capacity in CAPACITIES, job in (GenerateGreens, GenerateRSVD, ComputeBounds)
                p = predict(job, pt, coeffs; pad=true, vram_capacity_bytes=capacity)
                @printf("%s %s %s %s %d %s %s %s | %.9g %.9g %.9g %.9g %.9g %s\n",
                        cluster, cells, scale, sep, rank, string(num_pos),
                        string(capacity), string(job),
                        p.time_s, p.device_time_s, p.host_bytes, p.vram_bytes,
                        p.vram_floor_bytes, string(p.mode))
                n += 1
            end
        end
    end
    println(stderr, "$n predictions")
    return n
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
