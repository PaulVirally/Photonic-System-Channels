using Base.Threads
using JLD2
using Plots
using Plots.PlotMeasures

max_threads = 8
nthreads() >= max_threads || error("Please run with at least $max_threads threads: julia -t $max_threads")

nearest_even(r::Real) = iseven(floor(Int, r)) ? floor(Int, r) : ceil(Int, r)

function run_shell(cmd::Base.AbstractCmd, input="")
    inp = Pipe()
    out = Pipe()
    err = Pipe()

    process = run(pipeline(cmd, stdin=inp, stdout=out, stderr=err), wait=false)
    close(out.in)
    close(err.in)

    stdout = @async String(read(out))
    stderr = @async String(read(err))
    write(process, input)
    close(inp)
    wait(process)
    if process isa Base.ProcessChain
        exitcode = maximum([p.exitcode for p in process.processes])
    else
        exitcode = process.exitcode
    end

    return (
        stdout = fetch(stdout),
        stderr = fetch(stderr),
        code = exitcode
    )
end

function sample(num_cells::Int, num_threads::Int)
    command = `/usr/bin/time --format %e,%M julia -t $(num_threads) --project=. bench/generate_single_greens.jl -n $(num_cells)`
    _, err_out, _ = run_shell(command)
    time_prog_out = split(err_out, "\n")[end-1] # second last line contains output from /usr/bin/time

    time_str, ram_str = split(time_prog_out, ",")
    time_s = parse(Float64, strip(time_str))
    max_ram_bytes = parse(Int, strip(ram_str)) * 1024 # Convert from KB to bytes
    return time_s, max_ram_bytes
end

threads_vec = collect(2:2:max_threads)
pushfirst!(threads_vec, 1)
# cells_vec = sort(unique(nearest_even.(10 .^ range(log10(2), 7, 51))))
cells_vec = 2:8:256
ram_data_bytes = Array{Int, 2}(undef, length(cells_vec), length(threads_vec))
time_data_sec = Array{Float64, 2}(undef, length(cells_vec), length(threads_vec))

sample(2, nthreads()) # warmup run to get everything compiled
for (i, cells) in enumerate(cells_vec), (j, threads) in enumerate(threads_vec)
    wall_time_s, max_ram_bytes = sample(cells, threads)
    ram_data_bytes[i, j] = max_ram_bytes
    time_data_sec[i, j] = wall_time_s
    println("Cells: $(cells .^ 3), Threads: $threads => Time: $wall_time_s s, RAM: $(max_ram_bytes / 1024^2) MB")
end
@save "bench/generate_greens_benchmark.jld2" cells_vec threads_vec ram_data_bytes time_data_sec

plt_time = plot(cells_vec .^ 3, time_data_sec, xlabel="Number of cells", ylabel="Wall time [s]", label=permutedims(["$i thread" * (i == 1 ? "" : "s") for i in threads_vec]), title="Green's function generation time", dpi=300)
plt_ram = plot(cells_vec .^ 3, ram_data_bytes ./(1024^2), xlabel="Number of cells", ylabel="Max RAM usage [MB]", label=permutedims(["$i thread" * (i == 1 ? "" : "s") for i in threads_vec]), title="Green's function generation RAM usage", dpi=300)
plt = plot(plt_time, plt_ram, layout=(1, 2), size=(2*1920÷2, 1080÷2), left_margin=5mm, right_margin=5mm, bottom_margin=5mm)
savefig(plt, "bench/generate_greens_benchmark.png")
