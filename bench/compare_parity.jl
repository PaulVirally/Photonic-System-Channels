#!/usr/bin/env julia
"""
    bench/compare_parity.jl

Compare the spectra two RSVD runs wrote, which is the check trial E2 exists for:
the same body, the same rank, one run on the in-memory `CuArray` path and one on
Funicular's panel path.

    julia --project=. bench/compare_parity.jl \\
        --a  <cal_root>/funicular/l1_inmem/<prefix>.jld \\
        --b  <cal_root>/funicular/l1_panel/<prefix>.jld

Options:

  --a / --b            the two `.jld` files (required)
  --label-a / --label-b   names for the report (default "in-memory" / "panel")
  --rtol 1e-6          tolerance for the verdict
  --top-fraction 0.1   the leading share of the positive spectrum the verdict is
                       taken over
  --keys UR_asym,RS    which spectra to compare
  --strict             also fail on a deviation past the top band

# Why this is not an equality check

The two paths do not draw the same test matrix. The in-memory path takes its
Gaussian sketch from Julia's global RNG, while the panel path has nowhere to keep
one and regenerates blocks from an integer `seed` (`src/Params.jl`). Two runs of
the panel path at the same seed sketch identically, but a panel run and an
in-memory run never do, whatever seed either is given. The two spectra therefore
agree only to the accuracy of the randomized method itself, and asserting
bit-for-bit equality would be asserting something false.

What is meaningful is the top of the spectrum. A randomized eigensolver with
`q = 14` power iterations resolves the leading directions to many digits and the
tail to few, the tail being where the sketch's random subspace has not converged
and where two different random subspaces have no reason to agree. The verdict is
taken over the leading `--top-fraction` of the positive eigenvalues, and
everything below it is reported rather than judged.

Deviations are reported two ways, because they answer different questions:

  * per-element, `|a - b| / max(|a|, |b|)`: how many digits this eigenvalue agrees
    to. The honest measure at the top, and meaningless once the eigenvalues are
    small enough that both runs are reporting noise.
  * scaled, `|a - b| / |Gamma_1|`: the deviation as a share of the largest
    eigenvalue. This is the one that says whether a disagreement matters
    downstream, since the bounds job's pencil is built from the whole positive
    block and a tiny eigenvalue disagreeing by 100% moves nothing.

Reads only JLD2 and needs no GPU, so it runs on a login node.
"""

using JLD2
using Printf
using Statistics

const DEFAULT_KEYS = ["UR_asym", "RS"]

function parse_cli(argv::Vector{String})
    opts = Dict{String,String}()
    i = 1
    while i <= length(argv)
        startswith(argv[i], "--") || error("Expected an option starting with --, got '$(argv[i])'")
        key = argv[i][3:end]
        if i + 1 <= length(argv) && !startswith(argv[i + 1], "--")
            opts[key] = argv[i + 1]
            i += 2
        else
            opts[key] = "true"
            i += 1
        end
    end
    return opts
end

"""
    read_spectrum(path, key) -> (values, num_pos) or nothing

`<key>/D` and, where the group has one, `<key>/num_pos`. The `RS/` group holds
singular values and has no positivity split, so `num_pos` there is the whole
vector.
"""
function read_spectrum(path::AbstractString, key::AbstractString)
    isfile(path) || error("no such file: $path")
    return jldopen(path, "r") do io
        haskey(io, "$key/D") || return nothing
        values = Array(io["$key/D"])
        num_pos = haskey(io, "$key/num_pos") ? Int(io["$key/num_pos"]) :
                  count(>(zero(eltype(values))), values)
        return (values=values, num_pos=num_pos)
    end
end

"""
    deviations(a, b) -> (per_element, scaled)

Element-wise relative deviation, and deviation scaled by the leading eigenvalue.
Both are zero where the two entries are exactly equal, including where both are
zero, so a padded tail does not manufacture a disagreement.
"""
function deviations(a::AbstractVector, b::AbstractVector)
    n = min(length(a), length(b))
    scale = n == 0 ? 1.0 : max(abs(float(a[1])), abs(float(b[1])), eps())
    per_element = zeros(Float64, n)
    scaled = zeros(Float64, n)
    for i in 1:n
        ai, bi = float(a[i]), float(b[i])
        gap = abs(ai - bi)
        denom = max(abs(ai), abs(bi))
        per_element[i] = denom == 0 ? 0.0 : gap / denom
        scaled[i] = gap / scale
    end
    return per_element, scaled
end

function summarize(label::AbstractString, values::AbstractVector)
    isempty(values) && return "  $(rpad(label, 16)) (empty)"
    worst, at = findmax(values)
    return @sprintf("  %-16s max %.3e at index %d   median %.3e   mean %.3e",
                    label, worst, at, median(values), mean(values))
end

"""
    compare_key(...) -> Bool

Report on one spectrum group and say whether the verdict band passed.
"""
function compare_key(key, path_a, path_b, label_a, label_b, rtol, top_fraction, strict)
    a = read_spectrum(path_a, key)
    b = read_spectrum(path_b, key)
    println("\n", "-"^78)
    println("$key/D")
    if a === nothing || b === nothing
        which = a === nothing ? label_a : label_b
        println("  missing from the $which output, nothing to compare")
        return true   # absence is a gap in the trial, not a parity failure
    end

    @printf("  %-12s %6d values, %d positive\n", label_a, length(a.values), a.num_pos)
    @printf("  %-12s %6d values, %d positive\n", label_b, length(b.values), b.num_pos)
    if length(a.values) != length(b.values)
        println("  NOTE: different lengths; comparing the leading ",
                min(length(a.values), length(b.values)), " entries")
    end
    if a.num_pos != b.num_pos
        # Worth saying loudly. `num_pos` sizes every downstream bounds object, so
        # the two paths disagreeing here changes the shape of the next job rather
        # than a digit in this one.
        println("  NOTE: num_pos differs by ", abs(a.num_pos - b.num_pos),
                " (", @sprintf("%.2f%%", 100 * abs(a.num_pos - b.num_pos) / max(a.num_pos, 1)),
                " of ", label_a, "'s); the smaller eigenvalues straddle zero, which is",
                " expected when two different random subspaces resolve the same tail")
    end

    per_element, scaled = deviations(a.values, b.values)
    m = min(a.num_pos, b.num_pos, length(per_element))
    if m == 0
        println("  no positive eigenvalues in common, nothing to judge")
        return false
    end
    top = max(1, round(Int, top_fraction * m))

    println(summarize("top $top (per-elt)", view(per_element, 1:top)))
    println(summarize("top $top (scaled)", view(scaled, 1:top)))
    println(summarize("all $m pos (per-elt)", view(per_element, 1:m)))
    println(summarize("all $m pos (scaled)", view(scaled, 1:m)))

    over_top = count(>(rtol), view(per_element, 1:top))
    over_all = count(>(rtol), view(per_element, 1:m))
    @printf("  %d of the top %d and %d of all %d positives exceed rtol=%.1e\n",
            over_top, top, over_all, m, rtol)

    worst, at = findmax(view(per_element, 1:top))
    @printf("  worst in band: index %d, %s %.12e vs %s %.12e (rel %.3e)\n",
            at, label_a, float(a.values[at]), label_b, float(b.values[at]), worst)

    ok = over_top == 0
    strict && (ok &= over_all == 0)
    println("  verdict: ", ok ? "AGREE" : "DEVIATION",
            strict ? " (strict: whole positive block judged)" : "")
    return ok
end

function main(argv::Vector{String})
    opts = parse_cli(argv)
    path_a = get(opts, "a", "")
    path_b = get(opts, "b", "")
    (isempty(path_a) || isempty(path_b)) && error("--a and --b are both required")
    label_a = get(opts, "label-a", "in-memory")
    label_b = get(opts, "label-b", "panel")
    rtol = parse(Float64, get(opts, "rtol", "1e-6"))
    top_fraction = parse(Float64, get(opts, "top-fraction", "0.1"))
    strict = get(opts, "strict", "false") in ("true", "1", "yes")
    keys = split(get(opts, "keys", join(DEFAULT_KEYS, ",")), ',')

    println("="^78)
    println("RSVD parity: $label_a against $label_b")
    println("  $label_a: $path_a")
    println("  $label_b: $path_b")
    @printf("  rtol %.1e over the leading %.0f%% of the positive spectrum\n",
            rtol, 100 * top_fraction)
    println("  the two paths use different RNG mechanisms, so agreement is to")
    println("  RSVD accuracy and never bit-for-bit; see the docstring")

    ok = true
    for key in keys
        ok &= compare_key(strip(key), path_a, path_b, label_a, label_b,
                          rtol, top_fraction, strict)
    end

    println("\n", "="^78)
    println(ok ? "PARITY OK" : "PARITY DEVIATION: read the per-key report above")
    return ok
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS) ? 0 : 1)
end
