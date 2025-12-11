module PhotonicSystemChannels

include("Params.jl")
using .Params
import .Params: rank

export RSVDParams, rank, oversamples, power_iter
export ComputeEnvironment, preload_dir, project_dir, scratch_dir, use_gpu

include("SMRSystems.jl")
using .SMRSystems

export SMRVolumeSymbol, Sender, Mediator, Receiver, Design, char2volume_symbol, volume_symbol2char
export SMRSystem, sender, mediator, receiver, ms_separation, rm_separation, rs_separation, volume, χ, susceptibility, chi, design_regions, universe_regions, universe, design
export load_greens_function

include("common.jl")

include("generate_greens.jl")
export generate_greens

include("rsvd.jl")
export generate_rsvd

include("bounds.jl")
export compute_bounds

end # module
