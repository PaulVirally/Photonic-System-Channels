#!/usr/bin/env julia
# Bounds at any channels from the reduced problem filled_bound.jl saved, on the CPU, no box solves:
#
#     julia --project=. filled_channels.jl results/wg_1ss2_reduced.jld2 1:300 > wg_1ss2_1-300.json
#
# Channels as a range (1:300) or a list (5,7,9). Same numbers filled_bound gives at those n;
# the "n" object has the same keys as in filled_bound.jl's JSON.

using PhotonicSystemChannels, JLD2

red = load(ARGS[1], "reduced")
ns = occursin(':', ARGS[2]) ? (:)(parse.(Int, split(ARGS[2], ':'))...) : parse.(Int, split(ARGS[2], ','))
bounds = filled_channels(red, ns; log=msg -> (println(stderr, msg); flush(stderr)))
num(x) = isfinite(x) ? string(x) : "null"
rows = join(("\"$n\":{" * join(("\"$k\":$(num(getfield(b, k)))" for k in keys(b)), ",") * "}" for (n, b) in sort(collect(bounds))), ",")
println("{\"n\":{$rows}}")
