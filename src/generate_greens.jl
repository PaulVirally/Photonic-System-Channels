using CUDA

function _generate_greens_sr(compute_env::ComputeEnvironment, smr::SMRSystem)
    @info string(now()) * " [generate_greens::_generate_greens_sr] Generating Greens functions for SR system"

    # sender -> receiver
    @info string(now()) * " [generate_greens::_generate_greens_sr] Generating sender -> receiver Greens function"
    load_greens_function(compute_env, smr, Receiver, Sender)
    run_gc() # relieve some memory pressure that CUDA sometimes introduces
    
    # universe -> universe
    @info string(now()) * " [generate_greens::_generate_greens_sr] Generating universe -> universe Greens function"
    load_greens_function(compute_env, smr, Design, Design)
    run_gc()

    @info string(now()) * " [generate_greens::_generate_greens_sr] Completed Greens function generation"
end

function _generate_greens_smr(compute_env::ComputeEnvironment, smr::SMRSystem)
    @info string(now()) * " [generate_greens::_generate_greens_smr] Generating Greens functions for SMR system"

    if prod(smr.mediator_volume.cel) != 0
        # sender -> mediator
        @info string(now()) * " [generate_greens::_generate_greens_smr] Generating sender -> mediator Greens function"
        load_greens_function(compute_env, smr, Mediator, Sender)
        run_gc() # relieve some memory pressure that CUDA sometimes introduces

        # mediator -> mediator
        @info string(now()) * " [generate_greens::_generate_greens_smr] Generating mediator -> mediator Greens function"
        load_greens_function(compute_env, smr, Mediator, Mediator)
        run_gc()

        # mediator -> receiver
        @info string(now()) * " [generate_greens::_generate_greens_smr] Generating mediator -> receiver Greens function"
        load_greens_function(compute_env, smr, Receiver, Mediator)
        run_gc()
    end

    # sender -> receiver
    @info string(now()) * " [generate_greens::_generate_greens_smr] Generating sender -> receiver Greens function"
    load_greens_function(compute_env, smr, Receiver, Sender)
    run_gc()
end

function generate_greens()
    @info string(now()) * " [generate_greens::generate_greens] Starting Greens function generation"
    compute_env, smr, _ = parse_args()

    if use_gpu(compute_env)
        @info string(now()) * " [generate_greens::generate_greens] Using GPU acceleration on device $(gpu_device(compute_env))"
        CUDA.device!(gpu_device(compute_env))
    else
        @info string(now()) * " [generate_greens::generate_greens] Using CPU computation"
    end

    if isnothing(mediator(smr))
        _generate_greens_sr(compute_env, smr)
    else
        _generate_greens_smr(compute_env, smr)
    end
end
