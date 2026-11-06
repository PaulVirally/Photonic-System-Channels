#!/usr/bin/env julia
# One point of Fig 2, the filled-block mediator bound:
#
#     julia --project=. filled_bound.jl wg 1//2 --k-mb 3072 --k-rs 384 --out results/wg_1ss2.jld2
#
# Writes the `filled_bound` result (bounds per n, tails, ranks) to `--out`, plus a JSON copy.
# On a machine with several GPUs, run one point per GPU (`--gpu 1` for the second).

using PhotonicSystemChannels, ArgParse, CUDA, JLD2, Printf

const GERMANIUM = 18.612 + 1.5502660841406073im

function main()
    s = ArgParseSettings(description="Filled-block mediator bound (Fig 2) for one geometry.")
    @add_arg_table! s begin
        "family"; help = "wg, wgt, ms05, ms1, ms2, ms4 or ms44"; required = true
        "param"; help = "mediator length ℓ (wg) or gap d (ms), in wavelengths, e.g. 1//2"; required = true
        "--scale"; help = "cell size in wavelengths"; default = "1//32"
        "--k-mb"; help = "rank of G_mb"; arg_type = Int; required = true
        "--k-rs"; help = "rank of G_rs"; arg_type = Int; required = true
        "--k-a"; help = "Asym G⁰ₘₘ eigenvectors added to the basis"; arg_type = Int; default = 200
        "--nmax"; help = "largest channel index"; arg_type = Int; default = 40
        "--gpu"; help = "GPU index, or -1 for the CPU"; arg_type = Int; default = 0
        "--out"; help = "output .jld2 file"; required = true
    end
    a = parse_args(s)
    rat(x) = (p = parse.(Int, split(x, "//")); length(p) == 2 ? p[1] // p[2] : p[1] // 1)
    a["gpu"] >= 0 && CUDA.device!(a["gpu"])
    t = time()
    r = filled_bound(a["family"], rat(a["param"]), rat(a["scale"]); χ=GERMANIUM, k_mb=a["k-mb"], k_rs=a["k-rs"],
                     k_a=a["k-a"], nmax=a["nmax"], on_gpu=a["gpu"] >= 0,
                     log=msg -> (println(@sprintf("[%7.0f s] ", time() - t), msg); flush(stdout)))
    mkpath(dirname(abspath(a["out"])))
    jldsave(a["out"]; result=r, family=a["family"], param=a["param"], scale=a["scale"])
    num(x) = isfinite(x) ? string(x) : "null"
    rows = join(("\"$n\":{" * join(("\"$k\":$(num(getfield(b, k)))" for k in keys(b)), ",") * "}" for (n, b) in sort(collect(r.bounds))), ",")
    write(splitext(a["out"])[1] * ".json",
          "{\"family\":\"$(a["family"])\",\"param\":\"$(a["param"])\",\"scale\":\"$(a["scale"])\",\"eps0\":$(r.ε0)," *
          "\"epsF\":$(r.εF),\"epsx\":$(r.εx),\"kU\":$(r.kU),\"kmb\":$(r.k_mb),\"krs\":$(r.k_rs),\"n\":{$rows}}")
end

main()
