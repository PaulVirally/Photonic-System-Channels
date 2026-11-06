module PhotonicSystemChannels

include("Params.jl")
using .Params
import .Params: rank

export RSVDParams, rank, oversamples, seed, GPUChoice
export ComputeEnvironment, preload_dir, project_dir, scratch_dir

include("SMRSystems.jl")
using .SMRSystems

export SMRVolumeSymbol, Sender, Mediator, Receiver, Design
export design_regions_from, design_string
export SMRSystem, sender, mediator, receiver, volume, susceptibility, design
export load_green_function, load_asym_green_function, green_map, asym_green_map, universe_mesh, file_prefix
export GapRefinement, gap_refinement, dof_length
export mesh, sender_mesh, receiver_mesh, refinement, is_refined, MIN_GAP_CELLS

include("common.jl")
export golden_min, residency_plan

include("generate_green.jl")
export generate_green

include("rsvd.jl")
export generate_rsvd, asym_ur, ur_asym_vectors_path
export DENSE_EXACT_MAX_N_U

include("bounds.jl")
export compute_bounds, bounds_from_spectrum, bounds_path
export augmented_basis, fit_augmentation, tau_prep, system_dual
export DEFAULT_GAMMA_RTOL

include("mediator.jl")
export mediator_front, mediator_curve, mediator_dual, mediator_primal

include("preconditioner.jl")
include("filled.jl")
include("elimination.jl")
export FilledBox, solo_box, boxsolve, filled_geometry, filled_bound, lanczos_norm

end # module
