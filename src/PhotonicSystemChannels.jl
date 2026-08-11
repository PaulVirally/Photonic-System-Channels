module PhotonicSystemChannels

include("Params.jl")
using .Params
import .Params: rank

export RSVDParams, rank, oversamples, power_iter, GPUChoice
export ComputeEnvironment, preload_dir, project_dir, scratch_dir, use_gpu, gpu_device

include("SMRSystems.jl")
using .SMRSystems

export SMRVolumeSymbol, Sender, Mediator, Receiver, Design, char2volume_symbol, volume_symbol2char
export SMRSystem, sender, mediator, receiver, ms_separation, rm_separation, rs_separation, volume, χ, susceptibility, chi, design_regions, universe_regions, universe, design, volume_pairs
export load_green_function, fix_mask, file_prefix

include("common.jl")

include("generate_green.jl")
export generate_green

include("rsvd.jl")
export generate_rsvd

include("bounds.jl")
export compute_bounds, load_bounds_inputs, bounds_from_spectrum

include("verify_bounds.jl")
export verify_bounds

end # module
