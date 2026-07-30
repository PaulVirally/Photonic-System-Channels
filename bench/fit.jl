#!/usr/bin/env julia
"""
    bench/fit.jl

Turn calibration CSVs into `CostModel.Coefficients`, one file per cluster.

    julia bench/fit.jl                       # everything in bench/data/*.csv
    julia bench/fit.jl bench/data/fir.csv    # specific files
    julia bench/fit.jl --report-only         # fit and report, write nothing

For each cluster found in the input it writes `bench/coeffs_<cluster>.jl`, which
`CostModel.load_coefficients!` picks up automatically, and prints a report of
what was fitted, from how many points, and how far the model lands from the
measurements it has.

# What identifies what

Nothing here is a free-for-all regression: each coefficient is pinned by the
points designed for it in `bench/plan.jl`.

  * `g0_thread_scaling` from the thread scan, by grid search over `s` with a
    two-parameter linear fit at each value.
  * The Green-block time coefficients from the `g0_self` / `g0_ext` points, by
    non-negative least squares on
    `[W_self, M_self, W_ext, M_ext, M_contact, 1_self, 1_ext]`.
    `W` and `M` are correlated (`W = M log2 M` over a 1.75x range of `log2 M`),
    so NNLS pushing one of a pair to zero is an expected, honest outcome rather
    than a failure -- it means the data cannot tell the two apart, and whichever
    survives reproduces the measurements.
  * Green-block memory from the `g0_multiregion` points against the analytic peak.
  * `mv_*` from the matvec points, two parameters per operator kind.
  * The dense rates from the `dense` points, by least squares through the origin
    weighted by flops, so the large shapes that the real jobs live at dominate.
  * `bandwidth` and `launch_latency` from the bounds Gram-Schmidt timings, which
    is the BLAS-1 loop whose cost the model actually needs.
  * `geigh_rate`, `sync_latency` and `host_root_find` from the sampled bounds
    outer-loop iterations, by separating the per-`n` constant from the
    per-inner-iteration slope.
  * Startup costs from the `startup_s` column.
  * The padding factors from the distribution of measured/predicted, so they
    reflect the model's actual spread instead of a guess.
"""

include(joinpath(@__DIR__, "cost_model.jl"))
include(joinpath(@__DIR__, "measure.jl"))
using .CostModel
using Dates
using LinearAlgebra
using Printf
using Statistics

# --------------------------------------------------------------------------- #
# Row access helpers
# --------------------------------------------------------------------------- #

const Row = Dict{String,String}

"Numeric field, or `nothing` when absent, unparseable, or not finite (a failed
sub-measurement records NaN rather than dropping the row)."
function num(row::Row, key::AbstractString)
    s = get(row, key, "")
    isempty(strip(s)) && return nothing
    v = tryparse(Float64, s)
    (v === nothing || !isfinite(v)) && return nothing
    return v
end
int(row::Row, key::AbstractString) = begin
    v = num(row, key)
    v === nothing ? nothing : round(Int, v)
end

"""
    extras(row) -> Dict{String,Float64}

Parse the `extra` column's `key=value;key=value` payload, keeping only the
numeric values (which is everything the fit cares about: sub-stage times,
`num_pos`, `N_u`, per-index outer-loop timings).
"""
function extras(row::Row)
    out = Dict{String,Float64}()
    for item in split(get(row, "extra", ""), ';')
        isempty(strip(item)) && continue
        parts = split(item, '='; limit=2)
        length(parts) == 2 || continue
        value = tryparse(Float64, strip(parts[2]))
        value === nothing || (out[strip(parts[1])] = value)
    end
    return out
end

cells(row::Row) = (int(row, "n_x"), int(row, "n_y"), int(row, "n_z"))

"Circulant cells for a block between two equal bodies: `prod(2C)`."
function block_M(row::Row)
    c = cells(row)
    any(isnothing, c) && return nothing
    return prod(2 .* c)
end

is_contact(row::Row) = int(row, "contact") == 1

# --------------------------------------------------------------------------- #
# Least squares with a non-negativity constraint
# --------------------------------------------------------------------------- #

"""
    nnls(A, b) -> Vector{Float64}

Least squares subject to `x >= 0`, by greedily zeroing the most negative
coefficient and refitting. Adequate here because the design matrices are tiny
(at most six columns) and the constraint is a physical one -- a negative cost per
cell is meaningless, and letting one appear would make the model produce negative
estimates outside the fitted range.
"""
function nnls(A::AbstractMatrix, b::AbstractVector)
    n = size(A, 2)
    active = trues(n)
    x = zeros(n)
    while any(active)
        idx = findall(active)
        sub = @view A[:, idx]
        sol = try
            sub \ b
        catch
            pinv(Matrix(sub)) * b
        end
        if all(>=(-1e-12), sol)
            x .= 0
            x[idx] .= max.(sol, 0)
            return x
        end
        # Drop the most negative coefficient and try again.
        active[idx[argmin(sol)]] = false
    end
    return x
end

"""
    nnls_relative(A, b) -> Vector{Float64}

Non-negative least squares on relative residuals, by scaling each equation by
`1/b_i`. The cost model spans three decades in every direction, so minimizing
absolute residuals would fit the largest point and ignore the rest; what we want
is a model that is within a few percent everywhere.
"""
function nnls_relative(A::AbstractMatrix, b::AbstractVector)
    keep = findall(>(0), b)
    isempty(keep) && return zeros(size(A, 2))
    scale = 1 ./ b[keep]
    return nnls(A[keep, :] .* scale, b[keep] .* scale)
end

"Least squares through the origin, weighted so that large regressors dominate."
function rate_through_origin(flops::Vector{Float64}, times::Vector{Float64})
    keep = findall(i -> flops[i] > 0 && times[i] > 0, eachindex(flops))
    isempty(keep) && return nothing
    f, t = flops[keep], times[keep]
    denom = sum(f .* t)
    denom > 0 || return nothing
    return sum(f .^ 2) / denom
end

function summarize(label::AbstractString, predicted::Vector{Float64},
                   measured::Vector{Float64})
    isempty(predicted) && return "$label: no points"
    ratios = [m / p for (p, m) in zip(predicted, measured) if p > 0 && m > 0]
    isempty(ratios) && return "$label: no usable points"
    return @sprintf("%-26s n=%-3d measured/predicted  median %.2f  min %.2f  max %.2f",
                    label, length(ratios), median(ratios), minimum(ratios), maximum(ratios))
end

# --------------------------------------------------------------------------- #
# Individual fits
# --------------------------------------------------------------------------- #

"""
    has_thread_variation(rows) -> Bool

Whether any Green-block points were taken at different thread counts. Without
that variation `eta(T)` is a constant absorbed into the per-cell coefficients and
`s` is not identifiable at all, so it must keep its default rather than be
"fitted" to whatever the grid happens to land on first.
"""
function has_thread_variation(rows::Vector{Row})
    threads = Set{Int}()
    for row in rows
        row["kind"] in ("g0_ext", "g0_self") || continue
        t = int(row, "threads")
        t === nothing || push!(threads, t)
    end
    return length(threads) >= 2
end

"""
    thread_scan_report(rows) -> Vector{String}

The raw scan: measured time and speedup versus one thread, per body.

Printed rather than only fitted because `eta(T) = 1 + s(T - 1)` cannot represent a
plateau, and on a machine with SMT the scan will very likely flatten (or reverse)
once it passes the physical core count. If it does, the single fitted `s` is a
compromise across the whole range, and the right response is to stop asking for
threads past the knee -- which you can only see by looking at the numbers.
"""
function thread_scan_report(rows::Vector{Row})
    groups = Dict{Any,Vector{Row}}()
    for row in rows
        row["kind"] in ("g0_ext", "g0_self") || continue
        key = (row["kind"], row["n_x"], row["n_y"], row["n_z"], row["sep_num"], row["sep_den"])
        push!(get!(groups, key, Row[]), row)
    end
    lines = String[]
    for (key, group) in sort(collect(groups); by=first)
        points = [(int(r, "threads"), num(r, "time_s")) for r in group]
        points = [(t, s) for (t, s) in points if t !== nothing && s !== nothing]
        length(unique(first, points)) >= 3 || continue
        sort!(points; by=first)
        base = last(points[1])
        push!(lines, "thread scan, $(key[1]) $(key[2])x$(key[3])x$(key[4]):")
        for (threads, seconds) in points
            push!(lines, @sprintf("    t=%-4d %8.2f s   speedup %5.2fx   efficiency %4.0f%%",
                                  threads, seconds, base / seconds,
                                  100 * (base / seconds) / threads))
        end
        best_threads = first(argmin(last, points))
        max_threads = first(points[end])
        push!(lines, best_threads < max_threads ?
              @sprintf("    fastest at t=%d, slower by t=%d: do not ask for more than %d",
                       best_threads, max_threads, best_threads) :
              @sprintf("    still improving at t=%d (the largest sampled)", max_threads))
    end
    return lines
end

"""
    fit_greens_time(rows, s) -> NamedTuple

Non-negative least squares for the Green-block construction coefficients. The
per-cell (quadrature) columns are divided by the thread efficiency so that points
taken at different core counts can be pooled.
"""
function fit_greens_time(rows::Vector{Row}, s::Float64)
    # Nine columns: an (fft, cell, fixed) triple for each of the three block
    # kinds. Contact is a kind, not a surcharge -- see `Coefficients`.
    A = Matrix{Float64}(undef, 0, 9)
    b = Float64[]
    used = Row[]
    for row in rows
        row["kind"] in ("g0_self", "g0_ext") || continue
        M = block_M(row)
        t = num(row, "time_s")
        threads = int(row, "threads")
        (M === nothing || t === nothing || threads === nothing || t <= 0) && continue
        W = fft_work(M)
        eta = 1 + s * (threads - 1)
        kind = row["kind"] == "g0_self" ? 1 : (is_contact(row) ? 3 : 2)
        push!(used, row)
        cols = zeros(9)
        cols[3 * (kind - 1) + 1] = W
        cols[3 * (kind - 1) + 2] = M / eta
        cols[3 * (kind - 1) + 3] = 1.0
        A = vcat(A, cols')
        push!(b, t)
    end
    isempty(b) && return nothing
    x = nnls_relative(A, b)
    predicted = A * x
    return (self_fft=x[1], self_cell=x[2], self_fixed=x[3],
            ext_fft=x[4], ext_cell=x[5], ext_fixed=x[6],
            contact_fft=x[7], contact_cell=x[8], contact_fixed=x[9],
            n=length(b), predicted=predicted, measured=b, rows=used,
            rel_sse=sum(abs2, (predicted .- b) ./ b))
end

"""
    fit_greens_joint(rows) -> (fit, s)

Fit the Green-block coefficients and the thread-scaling exponent together, by
grid search over `s` with the full non-negative fit inside. Fitting `s` from the
thread scan alone leaves it trading off against the size-dependent terms; doing
both at once lets the size sweep pin the shape and the thread scan pin `s`.
"""
function fit_greens_joint(rows::Vector{Row})
    identifiable = has_thread_variation(rows)
    if !identifiable
        default_s = Coefficients().g0_thread_scaling
        fit = fit_greens_time(rows, default_s)
        return fit, (fit === nothing ? nothing : default_s), false
    end
    best_fit, best_s = nothing, nothing
    for s in 0.0:0.02:1.0
        fit = fit_greens_time(rows, s)
        fit === nothing && return nothing, nothing, identifiable
        if best_fit === nothing || fit.rel_sse < best_fit.rel_sse
            best_fit, best_s = fit, s
        end
    end
    return best_fit, best_s, identifiable
end

"""
    fit_linear_memory(pairs) -> (factor, base, n)

Two-parameter non-negative fit of `peak = factor * analytic + base` over
`(analytic, measured)` pairs.

Requires at least `MIN_MEMORY_POINTS` distinct analytic values. With one or two
points, or with several points at the same size, the slope and the intercept are
not separable, and the fit will happily return an absurd multiplier attached to a
zero base -- which would then be extrapolated to every job. Returning `nothing`
keeps the default and says so in the report instead.
"""
const MIN_MEMORY_POINTS = 3

function fit_linear_memory(pairs::Vector{Tuple{Float64,Float64}})
    length(unique(p -> round(p[1]; sigdigits=6), pairs)) >= MIN_MEMORY_POINTS ||
        return (nothing, nothing, length(pairs))
    A = hcat([p[1] for p in pairs], ones(length(pairs)))
    b = [p[2] for p in pairs]
    x = nnls_relative(A, b)
    return (x[1], x[2], length(pairs))
end

"""
    fit_matvec(rows, kind) -> NamedTuple or nothing

Two-parameter fit `t = a * W + b` for one matvec kind, refusing to fit fewer than
`MIN_MATVEC_POINTS` distinct sizes.

The guard is not defensive padding. With one point the slope and intercept are
perfectly degenerate, and NNLS resolves it by putting everything on the slope --
which then gets multiplied by an `M log M` two orders of magnitude larger at
production sizes. On the partial fir data that single `matvec_self` point implied
1.9 s per matvec at 96x32x32, i.e. a 27-hour RSVD estimate, versus 27 ms from
narval's four-point fit. A silently absurd number in the requests is worse than a
loudly missing one.
"""
const MIN_MATVEC_POINTS = 3

function fit_matvec(rows::Vector{Row}, kind::AbstractString)
    A = Matrix{Float64}(undef, 0, 2)
    b = Float64[]
    sizes = Set{Int}()
    for row in rows
        row["kind"] == kind || continue
        M = block_M(row)
        t = num(row, "time_s")
        (M === nothing || t === nothing || t <= 0) && continue
        push!(sizes, M)
        A = vcat(A, [fft_work(M), 1.0]')
        push!(b, t)
    end
    length(sizes) >= MIN_MATVEC_POINTS || return (insufficient=true, n=length(b))
    x = nnls_relative(A, b)
    return (insufficient=false, fft=x[1], fixed=x[2], n=length(b),
            predicted=A * x, measured=b)
end

"""
    fit_dense_rates(rows) -> Dict

Notional flops per second for each dense primitive, plus a bandwidth and launch
latency from the BLAS-1 timings. Uses `CostModel`'s flop counts so the fitted
rates are exactly what `predict` divides by.
"""
function fit_dense_rates(rows::Vector{Row})
    dense = filter(r -> r["kind"] == "dense", rows)
    out = Dict{String,Any}()
    isempty(dense) && return out

    collect_pairs(flopf, timekey) = begin
        flops, times = Float64[], Float64[]
        for row in dense
            m, c = int(row, "dense_m"), int(row, "dense_c")
            (m === nothing || c === nothing) && continue
            ex = extras(row)
            # The headline column holds the QR time; everything else is in extra.
            t = timekey == "qr" ? num(row, "time_s") : get(ex, "t_$timekey", nothing)
            (t === nothing || t <= 0) && continue
            push!(flops, flopf(m, c))
            push!(times, t)
        end
        return flops, times
    end

    for (name, flopf, key) in (
        ("qr_rate", (m, c) -> CostModel.flops_qr(m, c), "qr"),
        ("gemm_rate", (m, c) -> CostModel.flops_gemm(m, c, c), "gemm_TN"),
        ("eigh_rate", (m, c) -> CostModel.flops_eigh(c), "eigh"),
        ("geigh_rate", (m, c) -> CostModel.flops_geigh(c), "geigh"),
        ("svdvals_rate", (m, c) -> CostModel.flops_svdvals(c), "svdvals"),
    )
        flops, times = collect_pairs(flopf, key)
        rate = rate_through_origin(flops, times)
        rate === nothing || (out[name] = (rate=rate, n=length(flops)))
    end

    # A second gemm shape, reported so a badly anisotropic gemm rate is visible.
    flops, times = collect_pairs((m, c) -> CostModel.flops_gemm(m, c, c), "gemm_NN")
    rate = rate_through_origin(flops, times)
    rate === nothing || (out["gemm_rate_NN"] = (rate=rate, n=length(flops)))

    # BLAS-1: dot reads two length-m complex vectors.
    A = Matrix{Float64}(undef, 0, 2)
    b = Float64[]
    for row in dense
        m = int(row, "dense_m")
        m === nothing && continue
        t = get(extras(row), "t_dot", nothing)
        (t === nothing || t <= 0) && continue
        A = vcat(A, [2 * m * 16.0, 1.0]')
        push!(b, t)
    end
    if !isempty(b)
        x = nnls(A, b)
        x[1] > 0 && (out["bandwidth_dot"] = (rate=1 / x[1], n=length(b)))
        out["launch_latency_dot"] = (rate=x[2], n=length(b))
    end

    syncs = Float64[]
    for row in dense
        t = get(extras(row), "t_sync", nothing)
        (t === nothing || t <= 0) && continue
        push!(syncs, t)
    end
    isempty(syncs) || (out["sync_latency"] = (rate=median(syncs), n=length(syncs)))
    return out
end

"""
    fit_bounds(rows, gemm_rate) -> NamedTuple

Split the bounds measurements into the coefficients the model needs.

Gram-Schmidt gives `bandwidth` and `launch_latency` directly, from
`t = bytes/BW + launches*L`.

The sampled outer-loop iterations give the rest. Iteration `n` costs a constant
(one `k x k` generalized eigendecomposition plus two Hermitian copies) plus
`num_pos - n + 1` inner iterations, each a `k x k` gemv, a device synchronisation
and a host-side root find. Fitting `t_outer(n)` against `[1, num_pos - n + 1]`
per row group separates the two, and regressing the per-inner slope against
`[1, k]` across groups separates `sync_latency` from `host_root_find`.
"""
function fit_bounds(rows::Vector{Row}, gemm_rate::Union{Nothing,Float64})
    bounds = filter(r -> r["kind"] in ("bounds_core", "stage_bounds"), rows)
    result = Dict{String,Any}()
    isempty(bounds) && return result

    # ---- Gram-Schmidt: bandwidth and launch latency ------------------------
    A = Matrix{Float64}(undef, 0, 2)
    b = Float64[]
    for row in bounds
        ex = extras(row)
        t = get(ex, "t_gram_schmidt", nothing)
        num_pos = get(ex, "num_pos", nothing)
        N_u = get(ex, "N_u", nothing)
        if N_u === nothing
            c = cells(row)
            any(isnothing, c) || (N_u = 6.0 * prod(c))
        end
        (t === nothing || num_pos === nothing || N_u === nothing || t <= 0) && continue
        pairs = num_pos * (num_pos - 1) / 2
        pairs > 0 || continue
        A = vcat(A, [pairs * 3 * N_u * 16.0, 2 * pairs]')
        push!(b, t)
    end
    if !isempty(b)
        x = nnls(A, b)
        x[1] > 0 && (result["bandwidth"] = (rate=1 / x[1], n=length(b)))
        result["launch_latency"] = (rate=x[2], n=length(b))
        result["gs_predicted"] = A * x
        result["gs_measured"] = b
    end

    # ---- Outer loop: geigh rate, sync latency, host root find --------------
    per_inner = Tuple{Float64,Float64}[]  # (k, seconds per inner iteration)
    geigh_pairs = Tuple{Float64,Float64}[] # (flops, seconds of the constant part)
    for row in bounds
        ex = extras(row)
        k = int(row, "rank")
        num_pos = get(ex, "num_pos", nothing)
        (k === nothing || num_pos === nothing) && continue
        samples = Tuple{Float64,Float64}[]
        for (key, value) in ex
            m = match(r"^outer\[(\d+)\]$", key)
            m === nothing && continue
            push!(samples, (parse(Float64, m.captures[1]), value))
        end
        length(samples) >= 2 || continue
        # t(n) = const + (num_pos - n + 1) * per_inner
        A2 = hcat(ones(length(samples)), [num_pos - s[1] + 1 for s in samples])
        b2 = [s[2] for s in samples]
        x2 = nnls(A2, b2)
        push!(geigh_pairs, (CostModel.flops_geigh(k), x2[1]))
        push!(per_inner, (Float64(k), x2[2]))
    end

    if !isempty(geigh_pairs)
        rate = rate_through_origin([p[1] for p in geigh_pairs], [p[2] for p in geigh_pairs])
        rate === nothing || (result["geigh_rate"] = (rate=rate, n=length(geigh_pairs)))
    end
    if !isempty(per_inner)
        # per-inner = sync_latency + k * host_root_find + 8k^2 / gemm_rate
        A3 = Matrix{Float64}(undef, 0, 2)
        b3 = Float64[]
        for (k, seconds) in per_inner
            gemv = gemm_rate === nothing ? 0.0 : CostModel.flops_gemm(k, k, 1) / gemm_rate
            A3 = vcat(A3, [1.0, k]')
            push!(b3, max(seconds - gemv, 0.0))
        end
        x3 = nnls(A3, b3)
        result["sync_latency"] = (rate=x3[1], n=length(b3))
        result["host_root_find"] = (rate=x3[2], n=length(b3))
    end
    return result
end

"""
    MAX_PLAUSIBLE_STARTUP_S

Ceiling on a believable process startup. Julia's boot plus loading precompiled
images and creating a CUDA context off a cold shared filesystem is minutes at
worst; anything beyond this is not startup.

The guard exists because it caught a real bug: `bench/plan.jl` was emitting an
unescaped `date` command substitution into an unquoted heredoc, so the
*submitting* shell expanded it and `startup_s` recorded queue wait. On fir that read as
9.8 hours, and since startup is added directly to the predicted time, writing it
into the coefficients would have put a 10-hour floor under every GPU request --
on the cluster where queue priority was already the problem.
"""
const MAX_PLAUSIBLE_STARTUP_S = 1800.0

"Median startup seconds for host and device points, refusing implausible values."
function fit_startup(rows::Vector{Row})
    host, device = Float64[], Float64[]
    for row in rows
        t = num(row, "startup_s")
        (t === nothing || t <= 0) && continue
        push!(row["device"] == "gpu" ? device : host, t)
    end
    summarise(v) = begin
        isempty(v) && return (value=nothing, n=0, rejected=false)
        m = median(v)
        m > MAX_PLAUSIBLE_STARTUP_S ? (value=nothing, n=length(v), rejected=true) :
                                      (value=m, n=length(v), rejected=false)
    end
    h, d = summarise(host), summarise(device)
    return (host=h.value, device=d.value, n_host=h.n, n_device=d.n,
            host_rejected=h.rejected, device_rejected=d.rejected,
            host_median=isempty(host) ? nothing : median(host),
            device_median=isempty(device) ? nothing : median(device))
end

# --------------------------------------------------------------------------- #
# Assembling a coefficient set
# --------------------------------------------------------------------------- #

"""
    fit_cluster(cluster, rows) -> (Coefficients, report)

Fit everything for one cluster. Any coefficient with no supporting points keeps
its default and is listed in the report as uncalibrated, so a thin data set is
visible rather than quietly baked in.
"""
function fit_cluster(cluster::AbstractString, rows::Vector{Row})
    report = String[]
    missing_fits = String[]
    fields = Dict{Symbol,Any}(:name => cluster, :calibrated => true)

    # ---- Green block times and thread scaling (fitted jointly) -------------
    greens, s, thread_identifiable = fit_greens_joint(rows)
    if s === nothing
        s = Coefficients().g0_thread_scaling
    elseif thread_identifiable
        fields[:g0_thread_scaling] = s
        push!(report, @sprintf("%-26s %.2f  (jointly with the block-time fit)",
                               "g0_thread_scaling", s))
        append!(report, thread_scan_report(rows))
    else
        push!(missing_fits, "g0_thread_scaling (all g0 points ran at one thread count)")
    end

    if greens === nothing
        push!(missing_fits, "g0 time coefficients (no g0_self/g0_ext points)")
    else
        fields[:g0_self_fft] = greens.self_fft
        fields[:g0_self_cell] = greens.self_cell
        fields[:g0_ext_fft] = greens.ext_fft
        fields[:g0_ext_cell] = greens.ext_cell
        fields[:g0_contact_fft] = greens.contact_fft
        fields[:g0_contact_cell] = greens.contact_cell
        fields[:g0_contact_fixed] = greens.contact_fixed
        fields[:g0_self_fixed] = greens.self_fixed
        fields[:g0_ext_fixed] = greens.ext_fixed
        for (label, fft, cell, fixed) in (
            ("g0 block time  self", greens.self_fft, greens.self_cell, greens.self_fixed),
            ("               ext", greens.ext_fft, greens.ext_cell, greens.ext_fixed),
            ("               contact", greens.contact_fft, greens.contact_cell, greens.contact_fixed))
            push!(report, @sprintf("%-26s %.3g s/(M log M), %.3g s/cell, %.3g s fixed",
                                   label, fft, cell, fixed))
        end
        push!(report, "  " * summarize("g0 block time", greens.predicted, greens.measured))
    end

    # ---- Green block memory -----------------------------------------------
    mem_pairs = Tuple{Float64,Float64}[]
    for row in rows
        row["kind"] in ("g0_multiregion", "stage_greens") || continue
        c = cells(row)
        any(isnothing, c) && continue
        peak = num(row, "peak_rss_bytes")
        peak === nothing && continue
        pt = row_to_srpoint(row)
        pt === nothing && continue
        push!(mem_pairs, (Float64(greens_counts(pt).peak_bytes), peak))
    end
    factor, base, n_mem = fit_linear_memory(mem_pairs)
    if factor === nothing
        push!(missing_fits, n_mem == 0 ?
              "greens memory (no g0_multiregion/stage_greens points)" :
              "greens memory (only $n_mem point(s); need $MIN_MEMORY_POINTS distinct sizes)")
    else
        fields[:greens_mem_factor] = factor
        fields[:greens_mem_base] = base
        push!(report, @sprintf("%-26s %.2f x analytic + %s  (from %d points)",
                               "greens memory", factor, human_bytes(base), n_mem))
    end

    # ---- matvecs ----------------------------------------------------------
    for (kind, fftkey, fixedkey, label) in (("matvec_self", :mv_self_fft, :mv_self_fixed, "matvec self"),
                                            ("matvec_ext", :mv_ext_fft, :mv_ext_fixed, "matvec ext"))
        fit = fit_matvec(rows, kind)
        if fit.insufficient
            push!(missing_fits, fit.n == 0 ? "$label (no $kind points)" :
                  "$label (only $(fit.n) point(s); need $MIN_MATVEC_POINTS distinct sizes)")
            continue
        end
        fields[fftkey] = fit.fft
        fields[fixedkey] = fit.fixed
        push!(report, @sprintf("%-26s %.3g s/(M log M) + %.3g s  (from %d points)",
                               label, fit.fft, fit.fixed, fit.n))
        push!(report, "  " * summarize(label, fit.predicted, fit.measured))
    end

    # ---- dense rates ------------------------------------------------------
    dense = fit_dense_rates(rows)
    for (key, field) in (("qr_rate", :qr_rate), ("gemm_rate", :gemm_rate),
                         ("eigh_rate", :eigh_rate), ("geigh_rate", :geigh_rate),
                         ("svdvals_rate", :svdvals_rate), ("sync_latency", :sync_latency))
        if haskey(dense, key)
            fields[field] = dense[key].rate
            unit = endswith(key, "_rate") ? "flop/s" : "s"
            push!(report, @sprintf("%-26s %.3g %s  (from %d points)", key,
                                   dense[key].rate, unit, dense[key].n))
        else
            push!(missing_fits, "$key (no dense points)")
        end
    end
    if haskey(dense, "gemm_rate_NN") && haskey(dense, "gemm_rate")
        ratio = dense["gemm_rate_NN"].rate / dense["gemm_rate"].rate
        push!(report, @sprintf("%-26s %.2f  (NN vs TN; far from 1 means one shape is slow)",
                               "gemm shape ratio", ratio))
    end

    # ---- bounds -----------------------------------------------------------
    gemm_rate = get(fields, :gemm_rate, Coefficients().gemm_rate)
    bounds = fit_bounds(rows, Float64(gemm_rate))
    for (key, field, unit) in (("bandwidth", :bandwidth, "B/s"),
                               ("launch_latency", :launch_latency, "s"),
                               ("geigh_rate", :geigh_rate, "flop/s"),
                               ("sync_latency", :sync_latency, "s"),
                               ("host_root_find", :host_root_find, "s per (num_pos*k)"))
        if haskey(bounds, key)
            fields[field] = bounds[key].rate
            push!(report, @sprintf("%-26s %.3g %s  (from %d bounds points)",
                                   key, bounds[key].rate, unit, bounds[key].n))
        elseif !haskey(fields, field)
            push!(missing_fits, "$key (no bounds points)")
        end
    end
    if haskey(bounds, "gs_predicted")
        push!(report, "  " * summarize("bounds gram-schmidt", bounds["gs_predicted"],
                                       bounds["gs_measured"]))
    end

    # ---- memory for the device jobs ---------------------------------------
    for (kinds, analytic, factor_field, base_field, label) in (
        (("stage_rsvd",), pt -> Float64(rsvd_counts(pt).host_dense_bytes),
         :rsvd_host_mem_factor, :rsvd_host_mem_base, "rsvd host memory"),
        # `bounds_core` is deliberately excluded: it synthesises its eigenvector
        # block directly on the device with `CUDA.randn`, so it never makes the
        # host-side copy that the real job's JLD2 read does. Pairing the analytic
        # host model with that measurement produced a 0.02x factor, i.e. the fit
        # concluding the host needs almost nothing.
        (("stage_bounds",), pt -> Float64(bounds_counts(pt).host_bytes),
         :bounds_host_mem_factor, :bounds_host_mem_base, "bounds host memory"))
        pairs = Tuple{Float64,Float64}[]
        for row in rows
            row["kind"] in kinds || continue
            pt = row_to_srpoint(row)
            pt === nothing && continue
            peak = num(row, "peak_rss_bytes")
            peak === nothing && continue
            push!(pairs, (analytic(pt), peak))
        end
        f, bs, n = fit_linear_memory(pairs)
        if f === nothing
            push!(missing_fits, n == 0 ? "$label (no points)" :
                  "$label (only $n point(s); need $MIN_MEMORY_POINTS distinct sizes)")
        else
            fields[factor_field] = f
            fields[base_field] = bs
            push!(report, @sprintf("%-26s %.2f x analytic + %s  (from %d points)",
                                   label, f, human_bytes(bs), n))
        end
    end
    for (kinds, analytic, factor_field, base_field, label) in (
        (("stage_rsvd",), pt -> Float64(rsvd_counts(pt).vram_bytes),
         :rsvd_vram_factor, :rsvd_vram_base, "rsvd VRAM"),
        (("stage_bounds", "bounds_core"), pt -> Float64(bounds_counts(pt).vram_bytes),
         :bounds_vram_factor, :bounds_vram_base, "bounds VRAM"))
        pairs = Tuple{Float64,Float64}[]
        for row in rows
            row["kind"] in kinds || continue
            pt = row_to_srpoint(row)
            pt === nothing && continue
            # Live pool high-water only. It is the one device number that scales
            # with the problem; the pool's reserved backing never shrinks, so a
            # size model fitted to it attributes allocation churn to the size term
            # (measured on the first run: slopes of 4-33x the analytic count,
            # anti-correlated with size, which produced 280 GB bounds estimates).
            # Rows from before that column existed are skipped rather than
            # substituted, so a stale run leaves the coefficient uncalibrated
            # instead of confidently wrong.
            peak = num(row, "peak_vram_live_bytes")
            (peak === nothing || peak <= 0) && continue
            push!(pairs, (analytic(pt), peak))
        end
        f, bs, n = fit_linear_memory(pairs)
        if f === nothing
            push!(missing_fits, n == 0 ?
                  "$label (no points with a live-pool measurement; rerun with the current bench/point.jl)" :
                  "$label (only $n point(s); need $MIN_MEMORY_POINTS distinct sizes)")
        else
            fields[factor_field] = f
            fields[base_field] = bs
            push!(report, @sprintf("%-26s %.2f x analytic + %s  (from %d points)",
                                   label, f, human_bytes(bs), n))
        end
    end

    # ---- pool overhead ----------------------------------------------------
    #=
    How much more the CUDA.jl pool reserves than the job actually has live. The
    VRAM model is fitted against live bytes, so this ratio is what turns a demand
    estimate into a number the device must actually have free.
    =#
    overheads = Float64[]
    for row in rows
        live = num(row, "peak_vram_live_bytes")
        reserved = num(row, "peak_vram_reserved_bytes")
        (live === nothing || reserved === nothing || live <= 0) && continue
        push!(overheads, reserved / live)
    end
    if isempty(overheads)
        push!(missing_fits, "pool overhead (no peak_vram_live_bytes; rerun with the " *
                            "current bench/point.jl)")
    else
        push!(report, @sprintf("%-26s median %.2fx, p95 %.2fx  (pool reserved / live, %d points)",
                               "device pool overhead", median(overheads),
                               quantile(overheads, 0.95), length(overheads)))
    end

    # ---- startup ----------------------------------------------------------
    startup = fit_startup(rows)
    #=
    The heredoc-expansion bug was per-launcher, not per-point: if one of the two
    startup numbers is implausible then every startup row from that run recorded
    queue wait, including the ones that happen to look believable. CPU-only points
    queue quickly, so their contamination is small enough to pass a ceiling test
    while still being wrong. Refuse both.
    =#
    contaminated = startup.host_rejected || startup.device_rejected
    for (value, rejected, seen, n, field, label) in (
        (startup.host, startup.host_rejected, startup.host_median, startup.n_host,
         :g0_startup_s, "host startup"),
        (startup.device, startup.device_rejected, startup.device_median, startup.n_device,
         :gpu_startup_s, "device startup"))
        if rejected
            push!(missing_fits,
                  @sprintf("%s (median %.0f s exceeds the %.0f s plausibility ceiling: almost certainly queue wait, not startup. Regenerate the launcher and rerun a few points.)",
                           label, seen, MAX_PLAUSIBLE_STARTUP_S))
        elseif contaminated && seen !== nothing
            push!(missing_fits,
                  @sprintf("%s (measured %.0f s, discarded: another startup number from this run was implausible, so all of them include queue wait)",
                           label, seen))
        elseif value === nothing
            push!(missing_fits, "$label (no startup_s; is PSC_T0 exported?)")
        else
            fields[field] = value
            push!(report, @sprintf("%-26s %.1f s  (from %d points)", label, value, n))
        end
    end

    coeffs = Coefficients(; (k => v for (k, v) in fields)...)

    # ---- padding from the validation rows ---------------------------------
    coeffs, pad_report = fit_padding(coeffs, rows)
    append!(report, pad_report)

    return coeffs, report, missing_fits
end

"""
    row_to_srpoint(row) -> SRPoint or nothing

Reconstruct the point a row was measured at. `num_pos` is taken from the row when
the measurement reported it, so that bounds fits use the real value instead of
the `NUM_POS_FRACTION` guess.
"""
function row_to_srpoint(row::Row)
    c = cells(row)
    any(isnothing, c) && return nothing
    sep_num, sep_den = int(row, "sep_num"), int(row, "sep_den")
    scale_num, scale_den = int(row, "scale_num"), int(row, "scale_den")
    (sep_num === nothing || sep_den === nothing) && return nothing
    scale = if scale_num === nothing || scale_den === nothing
        (1 // 32, 1 // 32, 1 // 32)
    elseif scale_num < 0
        # The anisotropic convention from SMRSystem: a negative scale means
        # (1/32, |scale|, |scale|).
        (1 // 32, abs(scale_num // scale_den), abs(scale_num // scale_den))
    else
        (scale_num // scale_den, scale_num // scale_den, scale_num // scale_den)
    end
    ex = extras(row)
    num_pos = haskey(ex, "num_pos") ? round(Int, ex["num_pos"]) : nothing
    return SRPoint(Tuple(c), Tuple(c); scale=scale,
                   separation=sep_num // sep_den,
                   rank=something(int(row, "rank"), 256),
                   oversamples=something(int(row, "oversamples"), 50),
                   power_iters=something(int(row, "power_iters"), 14),
                   threads=something(int(row, "threads"), 4),
                   num_pos=num_pos)
end

"""
    fit_padding(coeffs, rows) -> (coeffs, report)

Set the safety factors from the observed spread of measured/predicted on the
end-to-end points, at the 95th percentile and never below 1.05. A padding factor
chosen from data is the difference between "we asked for 20% more than we think we
need" and "we asked for 20% more than a model we have not checked thinks we need".
"""
function fit_padding(coeffs::Coefficients, rows::Vector{Row})
    report = String[]
    stage_kinds = Dict("stage_greens" => GenerateGreens, "stage_rsvd" => GenerateRSVD,
                       "stage_bounds" => ComputeBounds)
    time_ratios, host_ratios, vram_ratios = Float64[], Float64[], Float64[]
    for row in rows
        haskey(stage_kinds, row["kind"]) || continue
        job = stage_kinds[row["kind"]]
        pt = row_to_srpoint(row)
        pt === nothing && continue
        p = predict(job, pt, coeffs; pad=false)
        t = num(row, "time_s")
        (t === nothing || p.time_s <= 0) || push!(time_ratios, t / p.time_s)
        host = num(row, "peak_rss_bytes")
        (host === nothing || p.host_bytes <= 0) || push!(host_ratios, host / p.host_bytes)
        vram = num(row, "peak_vram_bytes")
        (vram === nothing || vram <= 0 || p.vram_bytes <= 0) || push!(vram_ratios, vram / p.vram_bytes)
    end

    q95(v) = isempty(v) ? nothing : quantile(v, 0.95)
    fields = Dict{Symbol,Any}()
    for (ratios, field, label) in ((time_ratios, :time_pad, "time"),
                                   (host_ratios, :host_mem_pad, "host memory"),
                                   (vram_ratios, :vram_pad, "VRAM"))
        q = q95(ratios)
        if q === nothing
            push!(report, "  $label padding: kept default (no end-to-end points; " *
                          "run --tier validate)")
            continue
        end
        pad = max(1.05, q)
        fields[field] = pad
        push!(report, @sprintf("  %-14s padding %.2f  (p95 over %d runs, median %.2f)",
                               label, pad, length(ratios), median(ratios)))
    end
    isempty(fields) && return coeffs, report
    updated = Dict{Symbol,Any}()
    for field in fieldnames(Coefficients)
        updated[field] = get(fields, field, getfield(coeffs, field))
    end
    return Coefficients(; (k => v for (k, v) in updated)...), report
end

# --------------------------------------------------------------------------- #
# Emission
# --------------------------------------------------------------------------- #

function write_coefficients(path::AbstractString, coeffs::Coefficients,
                           report::Vector{String}, missing_fits::Vector{String},
                           sources::Vector{String})
    open(path, "w") do io
        println(io, "# Generated by bench/fit.jl on $(Dates.now()). Do not edit by hand:")
        println(io, "# rerun the fit against the calibration CSVs instead.")
        println(io, "#")
        println(io, "# Sources:")
        for source in sources
            println(io, "#   $source")
        end
        if !isempty(missing_fits)
            println(io, "#")
            println(io, "# NOT calibrated (defaults kept):")
            for item in missing_fits
                println(io, "#   $item")
            end
        end
        println(io, "#")
        println(io, "# Fit summary:")
        for line in report
            println(io, "#   $line")
        end
        println(io)
        println(io, "CostModel.Coefficients(")
        for field in fieldnames(Coefficients)
            value = getfield(coeffs, field)
            literal = value isa String ? "\"$value\"" :
                      value isa Bool ? string(value) : @sprintf("%.10g", value)
            println(io, "    $field = $literal,")
        end
        println(io, ")")
    end
    return path
end

# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

function main(argv::Vector{String})
    report_only = "--report-only" in argv
    paths = filter(a -> !startswith(a, "--"), argv)
    if isempty(paths)
        data_dir = joinpath(@__DIR__, "data")
        isdir(data_dir) ||
            error("No input given and $data_dir does not exist. Put the calibration " *
                  "CSVs there, or pass them as arguments.")
        paths = [joinpath(data_dir, f) for f in sort(readdir(data_dir)) if endswith(f, ".csv")]
        isempty(paths) && error("No CSV files in $data_dir")
    end

    all_rows = Row[]
    for path in paths
        rows = read_csv_rows(path)
        println("Read $(length(rows)) rows from $path")
        append!(all_rows, rows)
    end
    isempty(all_rows) && error("No rows to fit")

    by_cluster = Dict{String,Vector{Row}}()
    for row in all_rows
        push!(get!(by_cluster, get(row, "cluster", "unknown"), Row[]), row)
    end

    # for (cluster, rows) in sort(collect(by_cluster))
    for (cluster, rows) in collect(by_cluster)
        println("\n" * "="^78)
        println("Cluster: $cluster  ($(length(rows)) rows)")
        kinds = Dict{String,Int}()
        for row in rows
            kinds[row["kind"]] = get(kinds, row["kind"], 0) + 1
        end
        println("  points: ", join(["$k=$v" for (k, v) in sort(collect(kinds))], "  "))
        println("-"^78)

        coeffs, report, missing_fits = fit_cluster(cluster, rows)
        for line in report
            println("  ", line)
        end
        if !isempty(missing_fits)
            println("\n  Not calibrated (kept defaults):")
            for item in missing_fits
                println("    - ", item)
            end
        end

        if !report_only
            path = joinpath(@__DIR__, "coeffs_$(cluster).jl")
            write_coefficients(path, coeffs, report, missing_fits, paths)
            println("\n  Wrote $path")
        end
    end
    println("\n" * "="^78)
    report_only && println("--report-only: no files written")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
