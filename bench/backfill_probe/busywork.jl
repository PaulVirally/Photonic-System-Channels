#!/usr/bin/env julia
"""
    bench/backfill_probe/busywork.jl

Genuine GPU (CPU-fallback) compute for a fixed wall-clock budget. This is the
payload of the narval backfill-queue-latency probe (see this directory's
README.md for the full picture).

# Why do real work instead of `sleep`

SLURM's backfill scheduler plans around the walltime a job *requests*, not what
it uses. The probe's question (how long does the scheduler make a low-priority
job wait before it starts) is answered the instant the job starts running:
`Start - Submit`. Nothing this script does after that changes the measurement.

A `sleep` for the requested duration would answer the same question, but it (a) shows up in
`sacct`/`seff` as a job that used none of its requested resources, which is
exactly the used/requested walltime pattern Alliance's fairshare and any
human auditor would flag, and (b) tells you nothing about whether the GPU you
were handed actually works. So instead this runs repeated synchronized
ComplexF32 gemms for `BUSY_MINUTES` (default 10) and logs an achieved-TFLOPS
line every ~30 s, then exits 0, deliberately well inside the requested
walltime, so every probe job finishes on its own instead of getting killed at
the time limit.

# Usage

    BUSY_MINUTES=10 julia --project=. bench/backfill_probe/busywork.jl
    julia --project=. bench/backfill_probe/busywork.jl --minutes 10

`--minutes` wins over `BUSY_MINUTES` if both are given.
"""

using CUDA
using Dates
using LinearAlgebra
using Printf

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

function parse_cli(argv::Vector{String})
    opts = Dict{String,String}()
    i = 1
    while i <= length(argv)
        arg = argv[i]
        startswith(arg, "--") || error("Expected an option starting with --, got '$arg'")
        key = arg[3:end]
        if i + 1 > length(argv) || startswith(argv[i + 1], "--")
            opts[key] = "true"
            i += 1
        else
            opts[key] = argv[i + 1]
            i += 2
        end
    end
    return opts
end

function busy_minutes(opts::Dict{String,String})
    haskey(opts, "minutes") && return parse(Float64, opts["minutes"])
    return parse(Float64, get(ENV, "BUSY_MINUTES", "10"))
end

# --------------------------------------------------------------------------- #
# The work itself
# --------------------------------------------------------------------------- #

# 8*n^3: 4 real mult-adds per complex mult-add (the usual 3-real-mult Karatsuba
# gemm trick is not what BLAS/cuBLAS use by default), times n^3 mult-adds for an
# n x n x n gemm. Close enough for a log line, not a paper claim.
gemm_flops(n::Integer) = 8.0 * n^3

function run_gpu(deadline::DateTime, n::Int=4096)
    @info "$(now()) [busywork] CUDA is functional on $(CUDA.name(CUDA.device())); running $(n)x$(n) ComplexF32 gemms on GPU"
    A = CUDA.randn(ComplexF32, n, n)
    B = CUDA.randn(ComplexF32, n, n)
    C = CUDA.zeros(ComplexF32, n, n)
    CUDA.synchronize()

    total_gemms = 0
    window_start = time()
    window_flops = 0.0
    while now() < deadline
        mul!(C, A, B)
        CUDA.synchronize()
        total_gemms += 1
        window_flops += gemm_flops(n)
        elapsed = time() - window_start
        if elapsed >= 30
            tflops = window_flops / elapsed / 1e12
            @info @sprintf("%s [busywork] %d gemm(s) so far, %.2f TFLOPS over the last %.1fs (GPU)",
                            string(now()), total_gemms, tflops, elapsed)
            window_start = time()
            window_flops = 0.0
        end
    end

    CUDA.unsafe_free!(A)
    CUDA.unsafe_free!(B)
    CUDA.unsafe_free!(C)
    return total_gemms
end

function run_cpu(deadline::DateTime, n::Int=2048)
    @warn "$(now()) [busywork] CUDA is not functional here; falling back to CPU BLAS gemms ($(n)x$(n) ComplexF32). If this job actually holds a GPU allocation, that is worth a second look, since this is expected only on a CPU-only node."
    A = randn(ComplexF32, n, n)
    B = randn(ComplexF32, n, n)
    C = zeros(ComplexF32, n, n)

    total_gemms = 0
    window_start = time()
    window_flops = 0.0
    while now() < deadline
        mul!(C, A, B)
        total_gemms += 1
        window_flops += gemm_flops(n)
        elapsed = time() - window_start
        if elapsed >= 30
            tflops = window_flops / elapsed / 1e12
            @info @sprintf("%s [busywork] %d gemm(s) so far, %.2f TFLOPS over the last %.1fs (CPU BLAS, %d thread(s))",
                            string(now()), total_gemms, tflops, elapsed, BLAS.get_num_threads())
            window_start = time()
            window_flops = 0.0
        end
    end
    return total_gemms
end

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

function main(argv::Vector{String}=copy(ARGS))
    opts = parse_cli(argv)
    minutes = busy_minutes(opts)
    minutes > 0 || error("--minutes/BUSY_MINUTES must be positive, got $minutes")

    start = now()
    deadline = start + Millisecond(round(Int, minutes * 60_000))
    @info "$(now()) [busywork] Starting $(minutes) minute(s) of busywork, deadline $(deadline)"

    use_gpu = try
        CUDA.functional()
    catch err
        @warn "$(now()) [busywork] CUDA.functional() threw ($(sprint(showerror, err))); treating as no GPU"
        false
    end

    total_gemms = use_gpu ? run_gpu(deadline) : run_cpu(deadline)

    elapsed_min = (now() - start).value / 60_000
    @info @sprintf("%s [busywork] Done: %d gemm(s) in %.2f minute(s). Exiting 0.",
                    string(now()), total_gemms, elapsed_min)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(copy(ARGS))
end
