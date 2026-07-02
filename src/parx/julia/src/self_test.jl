# Standalone smoke tests for `julia --project=. -e "using LinearRegions; LinearRegions.run_tests()"`.
# Exercises the module without going through the Python/juliacall bridge.

function run_tests()
    @assert occursin("LinearRegions.jl is loaded", hello())

    weights = [Matrix{Float64}([1.0 0.0; 0.0 1.0]), Matrix{Float64}([1.0 -1.0])]
    biases  = [Vector{Float64}([0.0, 0.0]), Vector{Float64}([0.0])]

    input_dim, n_layers = network_info(weights, biases)
    @assert input_dim == 2
    @assert n_layers == 2

    points = [1.0 1.0; -1.0 -1.0; 1.0 -1.0]
    patterns, offsets, centroids = find_regions_sparse(weights, biases, points)
    @assert size(patterns, 1) == size(centroids, 1)
    @assert offsets[end] == size(patterns, 2)

    x0 = [0.5, 0.5]
    result = find_regions_exact(weights, biases, x0)
    ex_patterns, ex_offsets, ex_centroids, active_flat, active_offsets, bounded = result
    n_regions = size(ex_patterns, 1)
    @assert n_regions > 0
    @assert length(bounded) == n_regions
    @assert active_offsets[end] == length(active_flat)

    println("LinearRegions self-tests passed ($(n_regions) exact regions found).")
    return true
end
