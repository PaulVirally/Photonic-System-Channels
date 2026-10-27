module PhotonicSystemChannels

include("Params.jl")
using .Params
import .Params: rank

export RSVDParams, rank, oversamples, power_iter, seed, GPUChoice
export ComputeEnvironment, preload_dir, project_dir, scratch_dir, use_gpu, gpu_device

include("SMRSystems.jl")
using .SMRSystems

export SMRVolumeSymbol, Sender, Mediator, Receiver, Design, char2volume_symbol, volume_symbol2char
export design_regions_from, design_string
export SMRSystem, sender, mediator, receiver, ms_separation, rm_separation, rs_separation, volume, susceptibility, design_regions, design
export load_green_function, load_asym_green_function, green_map, asym_green_map, universe_mesh, file_prefix
export GapRefinement, gap_refinement, refine_body, dof_length, mesh_tag
export mesh, sender_mesh, receiver_mesh, refinement, is_refined, MIN_GAP_CELLS

include("common.jl")
export parse_cli, residency_plan, resolve_seed, device_budget_bytes

include("generate_green.jl")
export generate_green

include("rsvd.jl")
export generate_rsvd, asym_ur, rsvd_output_path, ur_asym_vectors_path
export use_dense_path, use_panel_path, gila_workspace_bytes, DENSE_EXACT_MAX_N_U, DENSE_EXACT_MAX_N_R

include("bounds.jl")
export compute_bounds, load_bounds_inputs, bounds_from_spectrum, bounds_path
export reverse_gram_schmidt!, augmented_basis, FactoredB, factored_pencil_eigen, factored_probe_duals
export psd_pencil_whitener, bracket_root, uu_eigenbasis, fit_augmentation
export DEFAULT_K_UU, DEFAULT_GAMMA_RTOL

end # module
