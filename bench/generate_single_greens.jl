using PhotonicSystemChannels
using ArgParse

settings = ArgParseSettings()
@add_arg_table settings begin
    "-n", "--num-cells"
        help = "Number of cells in each dimension"
        arg_type = Int
        required = true
end
args = parse_args(settings)
cells = args["num-cells"]

smr = SMRSystem((cells, cells, cells), (8//32, 0//1, 0//1), (cells, cells, cells), [Sender, Receiver], 1//32, complex(0.0, 0.0))
env = ComputeEnvironment("", "", "", false)

G0 = load_greens_function(env, smr, Sender, Sender; force_generate=true, save_to_disk=false)
