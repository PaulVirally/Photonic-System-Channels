module PhotonicSystemChannels

include("Params.jl")
using .Params
import .Params: rank

include("SMRSystems.jl")
using .SMRSystems

include("common.jl")

include("generate_greens.jl")
export generate_greens

include("rsvd.jl")
export generate_rsvd

include("bounds.jl")
export compute_bounds

end # module
