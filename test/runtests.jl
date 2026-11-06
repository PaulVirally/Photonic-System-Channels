# Each test is a standalone script (CPU only); run them all, each in its own process:
#     julia --project=. test/runtests.jl
const TESTS = filter(f -> endswith(f, ".jl") && f != "runtests.jl", readdir(@__DIR__))
for t in TESTS
    println("== ", t)
    run(`$(Base.julia_cmd()) --project=$(dirname(@__DIR__)) $(joinpath(@__DIR__, t))`)
end
