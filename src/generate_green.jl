# Builds every Green operator the later stages load, so that they only deserialize.
function _generate_green_sr(compute_env::ComputeEnvironment, smr::SMRSystem)
    for (what, load) in (("G⁰ᵣₛ", () -> load_green_function(compute_env, smr, Receiver, Sender)),
                         ("G⁰ᵣᵣ", () -> load_green_function(compute_env, smr, Receiver, Receiver)),
                         ("G⁰ᵤᵤ", () -> load_green_function(compute_env, smr, [Sender, Receiver], [Sender, Receiver])),
                         # Asym(G⁰) of the self blocks, built from the volumes: these are what the RSVD and the bounds apply
                         ("Asym(G⁰ᵣᵣ)", () -> load_asym_green_function(compute_env, smr, Receiver)),
                         ("Asym(G⁰ᵤᵤ)", () -> load_asym_green_function(compute_env, smr, Design)))
        @info string(now()) * " [generate_green] Generating $(what)"
        load()
        run_gc()
    end
end

function _generate_green_smr(compute_env::ComputeEnvironment, smr::SMRSystem)
    for (what, load) in (("G⁰ᵣₛ", () -> load_green_function(compute_env, smr, Receiver, Sender)),
                         ("G⁰ₘₛ", () -> load_green_function(compute_env, smr, Mediator, Sender)),
                         ("G⁰ᵣ←[s;m]", () -> load_green_function(compute_env, smr, [Receiver], [Sender, Mediator])),
                         ("Asym(G⁰ₘₘ)", () -> load_asym_green_function(compute_env, smr, Mediator)))
        @info string(now()) * " [generate_green] Generating $(what)"
        load()
        run_gc()
    end
end

function generate_green()
    args = parse_cli()
    select_device(args.compute_env)
    if isnothing(mediator(args.smr))
        _generate_green_sr(args.compute_env, args.smr)
    else
        _generate_green_smr(args.compute_env, args.smr)
    end
    @info string(now()) * " [generate_green] Done"
end
