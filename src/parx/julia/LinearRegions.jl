module LinearRegions

using LinearAlgebra
using JuMP
using HiGHS

include(joinpath(@__DIR__, "bridge.jl"))
include(joinpath(@__DIR__, "sparse.jl"))
include(joinpath(@__DIR__, "lp.jl"))
include(joinpath(@__DIR__, "exact.jl"))

end # module
