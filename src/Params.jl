module Params

export RSVDParams, rank, oversamples, power_iter
export ComputeEnvironment, preload_dir, project_dir, scratch_dir, use_gpu

"""
    RSVDParams

A structure to hold parameters for Randomized Singular Value Decomposition (RSVD).

# Fields
- `rank::Int`: The number of singular value components to compute.
- `oversamples::Int`: The number of oversamples to use in RSVD.
- `power_iter::Int`: The number of power iterations to perform in RSVD.
"""
struct RSVDParams
    rank::Int
    oversamples::Int
    power_iter::Int
end

rank(params::RSVDParams) = params.rank
oversamples(params::RSVDParams) = params.oversamples
power_iter(params::RSVDParams) = params.power_iter

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
    use_gpu::Bool
end

preload_dir(env::ComputeEnvironment) = env.preload_dir
project_dir(env::ComputeEnvironment) = env.project_dir
scratch_dir(env::ComputeEnvironment) = env.scratch_dir
use_gpu(env::ComputeEnvironment) = env.use_gpu

end # module
