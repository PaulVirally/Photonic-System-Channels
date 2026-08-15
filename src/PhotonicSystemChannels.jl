module PhotonicSystemChannels

include("Params.jl")
using .Params
import .Params: rank

export RSVDParams, rank, oversamples, power_iter, seed, GPUChoice
export ComputeEnvironment, preload_dir, project_dir, scratch_dir, use_gpu, gpu_device

include("SMRSystems.jl")
using .SMRSystems

export SMRVolumeSymbol, Sender, Mediator, Receiver, Design, char2volume_symbol, volume_symbol2char
export SMRSystem, sender, mediator, receiver, ms_separation, rm_separation, rs_separation, volume, χ, susceptibility, chi, design_regions, universe_regions, universe, design, volume_pairs
export load_green_function, fix_mask, file_prefix

include("common.jl")
export residency_plan, resolve_seed, device_budget_bytes

include("generate_green.jl")
export generate_green

include("rsvd.jl")
export generate_rsvd
export use_dense_path, use_panel_path, gila_workspace_bytes, materialize_columns, ur_asym_vectors_path
export DENSE_EXACT_MAX_N_U, DENSE_EXACT_MAX_N_R

include("bounds.jl")
export compute_bounds, load_bounds_inputs, bounds_from_spectrum
export use_panel_bounds, bounds_footprint_bytes
export reverse_gram_schmidt!, blocked_reverse_gs_transform

include("verify_bounds.jl")
export verify_bounds

end # module
