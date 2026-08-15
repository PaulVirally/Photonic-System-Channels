module Params

export RSVDParams, rank, oversamples, power_iter, seed, GPUChoice
export ComputeEnvironment, preload_dir, project_dir, scratch_dir, use_gpu, gpu_device

struct GPUChoice
    use_gpu::Bool
    id::Int
end

"""
    RSVDParams

A structure to hold parameters for Randomized Singular Value Decomposition (RSVD).

# Fields
- `rank::Int`: The number of singular value components to compute.
- `oversamples::Int`: The number of oversamples to use in RSVD.
- `power_iter::Int`: The number of power iterations to perform in RSVD.
- `seed::Int`: Seed for the panel path's regenerated Gaussian test matrix. The
  panel path has no stored test matrix to seed and takes an integer instead, so
  the seed is carried here and written into the output alongside the spectrum.
  The in-memory path draws from the global RNG and ignores it.

The three-argument constructor leaves `seed` at 0.
"""
struct RSVDParams
    rank::Int
    oversamples::Int
    power_iter::Int
    seed::Int
end

RSVDParams(rank::Int, oversamples::Int, power_iter::Int) = RSVDParams(rank, oversamples, power_iter, 0)

rank(params::RSVDParams) = params.rank
oversamples(params::RSVDParams) = params.oversamples
power_iter(params::RSVDParams) = params.power_iter
seed(params::RSVDParams) = params.seed

"""
    ComputeEnvironment(preload_dir::String, project_dir::String, scratch_dir::String, use_gpu::Bool)

A struct to hold paths and settings for the compute environment.

# Fields
- `preload_dir::String`: Directory path for preloaded data.
- `project_dir::String`: Directory path for the project.
- `scratch_dir::String`: Directory path for scratch data.
- `use_gpu::Bool`: Flag indicating whether to use GPU acceleration.
"""
struct ComputeEnvironment
    preload_dir::String
    project_dir::String
    scratch_dir::String
    gpu_choice::GPUChoice
end

preload_dir(env::ComputeEnvironment) = env.preload_dir
project_dir(env::ComputeEnvironment) = env.project_dir
scratch_dir(env::ComputeEnvironment) = env.scratch_dir
use_gpu(env::ComputeEnvironment) = env.gpu_choice.use_gpu
gpu_device(env::ComputeEnvironment) = env.gpu_choice.id

end # module
