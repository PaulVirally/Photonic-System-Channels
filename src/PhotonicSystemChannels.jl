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
# The block-parallel bounds path: where a `--partial-suffix` run writes. Exported
# so test/bounds_blocks.jl names the files the same way the writer does;
# bench/merge_bounds_blocks.jl deliberately re-derives the pattern instead, so
# that a merge job starts in seconds without loading CUDA.
export partial_bounds_path
export use_panel_bounds, bounds_footprint_bytes
export reverse_gram_schmidt!, blocked_reverse_gs_transform
# The Asym(G⁰ᵤᵤ) augmentation of the projection basis. Exported because
# bench/augmented_basis_experiment.jl and test/augmented_basis.jl drive these
# directly -- the experiment is a thin driver over exactly this code, which is what
# makes it evidence about production rather than about a second implementation.
export augmented_basis, qr_thin_rdiag!
export FactoredB, apply_factor, factored_diag, factored_pencil_eigen, factored_probe_duals
export uu_eigenbasis, uu_residuals, plan_uu_solve
export uu_sketch_bytes, augmented_footprint_bytes
# The adaptive --k-uu clip. `max_k_uu_for_budget` is pure arithmetic on
# (N_u, m, budget) and is mirrored by `augment_k_uu_cap` in bench/cost_model.jl;
# test/augmented_basis.jl checks the two against each other, which is why it is
# exported rather than internal.
export max_k_uu_for_budget, clip_k_uu, K_UU_CLIP_FLOOR
export DEFAULT_K_UU, DEFAULT_AUGMENT_THRESHOLD, DEFAULT_GAMMA_RTOL
export AUG_QR_RTOL, AUG_BTOL, UU_OVERSAMPLES, UU_POWER_ITERS, UU_MIN_OVERSAMPLES

include("verify_bounds.jl")
export verify_bounds

end # module
