#!/bin/bash
# Configure CUDA.jl for narval. Compute nodes have no internet, so CUDA.jl must
# use the cluster's CUDA module instead of downloading its artifact runtime:
# CUDA.set_runtime_version!(v"12.2", local_toolkit=true). The preference lands
# in LocalPreferences.toml next to Project.toml, so every later job inherits it
# as long as the job loads the cuda/12.2 module (the generated launchers do).
#
# Run inside a GPU allocation so the last step can verify CUDA is functional:
#
#   salloc --account=def-smolesky --time=00:30:00 --mem=16G --cpus-per-task=2 \
#          --gpus=a100_1g.5gb:1 srun --pty bash test/setup_cuda_narval.sh
set -e
module load StdEnv/2023 julia/1.12.5 cuda/12.2
cd "$(dirname "$0")/.."

# Write the preference. This session's CUDA keeps its old configuration, so the
# change only takes effect in a fresh process.
julia --project=. -e 'using CUDA; CUDA.set_runtime_version!(v"12.2", local_toolkit=true)'

# Fresh process: precompile CUDA (now against the local toolkit) and the rest.
julia --project=. -e 'using Pkg; Pkg.precompile()'

# Fresh process again: verify the GPU actually works before trusting 14 jobs to it.
julia --project=. -e '
using CUDA
@assert CUDA.functional() "CUDA.functional() is false; check that the cuda module is loaded and a GPU is allocated"
CUDA.versioninfo()
using PhotonicSystemChannels
x = CUDA.rand(Float32, 1024)
println("CUDA smoke: sum of 1024 random floats = ", sum(x))
println("ALL GOOD: CUDA is configured for the local toolkit")
'
