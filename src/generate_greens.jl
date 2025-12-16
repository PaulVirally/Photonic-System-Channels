function generate_greens()
    @info string(now()) * " [generate_greens::generate_greens] Starting Greens function generation"
    compute_env, smr, _ = parse_args()

    @info string(now()) * " [generate_greens::generate_greens] Generating Greens functions for SMR system"

    # sender -> receiver
    @info string(now()) * " [generate_greens::generate_greens] Generating sender -> receiver Greens function"
    load_greens_function(compute_env, smr, Receiver, Sender)
    run_gc() # relieve some memory pressure that CUDA sometimes introduces

    # universe -> receiver
    @info string(now()) * " [generate_greens::generate_greens] Generating universe -> receiver Greens function"
    load_greens_function(compute_env, smr, Receiver, Design)
    run_gc()

    # receiver -> universe
    @info string(now()) * " [generate_greens::generate_greens] Generating receiver -> universe Greens function"
    load_greens_function(compute_env, smr, Design, Receiver)
    run_gc()

    # sender -> universe
    @info string(now()) * " [generate_greens::generate_greens] Generating sender -> universe Greens function"
    load_greens_function(compute_env, smr, Design, Sender)
    run_gc()

    # universe -> sender
    @info string(now()) * " [generate_greens::generate_greens] Generating universe -> sender Greens function"
    load_greens_function(compute_env, smr, Sender, Design)
    run_gc()

    @info string(now()) * " [generate_greens::generate_greens] Completed Greens function generation"
end
