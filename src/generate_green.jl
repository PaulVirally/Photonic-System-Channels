using CUDA

function _generate_green_sr(compute_env::ComputeEnvironment, smr::SMRSystem)
    @info string(now()) * " [generate_green::_generate_green_sr] Generating Green functions for SR system"

    # sender -> receiver
    @info string(now()) * " [generate_green::_generate_green_sr] Generating sender -> receiver Green function"
    load_green_function(compute_env, smr, Receiver, Sender)
    run_gc() # relieve some memory pressure that CUDA sometimes introduces

    # receiver -> receiver
    @info string(now()) * " [generate_green::_generate_green_sr] Generating receiver -> receiver Green function"
    load_green_function(compute_env, smr, Receiver, Receiver)
    run_gc()

    # universe -> universe
    @info string(now()) * " [generate_green::_generate_green_sr] Generating universe -> universe Green function"
    # load_green_function(compute_env, smr, Design, Design)
    load_green_function(compute_env, smr, [Sender, Receiver], [Sender, Receiver]) # universe -> universe
    run_gc()

    @info string(now()) * " [generate_green::_generate_green_sr] Completed Green function generation"
end

function _generate_green_smr(compute_env::ComputeEnvironment, smr::SMRSystem)
    @info string(now()) * " [generate_green::_generate_green_smr] Generating Green functions for SMR system"

    if prod(smr.mediator_volume.cel) != 0
        # sender -> mediator
        @info string(now()) * " [generate_green::_generate_green_smr] Generating sender -> mediator Green function"
        load_green_function(compute_env, smr, Mediator, Sender)
        run_gc() # relieve some memory pressure that CUDA sometimes introduces

        # mediator -> mediator
        @info string(now()) * " [generate_green::_generate_green_smr] Generating mediator -> mediator Green function"
        load_green_function(compute_env, smr, Mediator, Mediator)
        run_gc()

        # mediator -> receiver
        @info string(now()) * " [generate_green::_generate_green_smr] Generating mediator -> receiver Green function"
        load_green_function(compute_env, smr, Receiver, Mediator)
        run_gc()
    end

    # sender -> receiver
    @info string(now()) * " [generate_green::_generate_green_smr] Generating sender -> receiver Green function"
    load_green_function(compute_env, smr, Receiver, Sender)
    run_gc()
end

function generate_green()
    @info string(now()) * " [generate_green::generate_green] Starting Green function generation"
    compute_env, smr, _ = parse_args()

    if use_gpu(compute_env)
        @info string(now()) * " [generate_green::generate_green] Using GPU acceleration on device $(gpu_device(compute_env))"
        if !haskey(ENV, "CC_CLUSTER") # This breaks on compute canada
            CUDA.device!(gpu_device(compute_env))
        end
    else
        @info string(now()) * " [generate_green::generate_green] Using CPU computation"
    end

    if isnothing(mediator(smr))
        _generate_green_sr(compute_env, smr)
    else
        _generate_green_smr(compute_env, smr)
    end
end
