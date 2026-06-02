# Exact region enumeration via DFS + facet-flipping.
#
# Internal DFS is parameterised by `make_model` so that `find_regions_exact`
# and `find_regions_exact_fast` share all logic and differ only in LP construction.

function _dfs_exact!(
    weights  ::Vector{Matrix{Float64}},
    biases   ::Vector{Vector{Float64}},
    layer    ::Int,
    A_prev   ::Matrix{Float64},   # accumulated linear map: input → pre-ReLU at layer-1
    c_prev   ::Vector{Float64},   # accumulated bias
    D_prev   ::Matrix{Float64},   # stacked halfspace rows from ancestor layers
    g_prev   ::Vector{Float64},
    q_path   ::Vector{BitVector}, # activation patterns for layers 1 … layer-1
    x_parent ::Vector{Float64},   # interior point of parent region
    results  ::Vector{Tuple{Vector{Int8}, Vector{Float64}}},
    make_model,
)
    if layer > length(weights)
        path_flat = reduce(vcat, [Int8.(q) for q in q_path])
        push!(results, (path_flat, copy(x_parent)))
        return
    end

    W, b  = weights[layer], biases[layer]
    W_hat = W * A_prev        # (n_out, input_dim)
    b_hat = W * c_prev + b    # (n_out,)

    q_start = BitVector(W_hat * x_parent + b_hat .> 0)
    queue   = [q_start]
    visited = Set{BitVector}([q_start])

    while !isempty(queue)
        q_curr = popfirst!(queue)

        s       = -2.0 .* Float64.(q_curr) .+ 1.0  # −1 if active, +1 if inactive
        D_local = s .* W_hat
        g_local = -(s .* b_hat)

        D_full = vcat(D_prev, D_local)
        g_full = vcat(g_prev, g_local)

        x_int, r = _chebyshev_center(D_full, g_full, make_model)
        isnothing(x_int) && continue   # empty or degenerate region — skip

        A_next = Float64.(q_curr) .* W_hat
        c_next = Float64.(q_curr) .* b_hat

        _dfs_exact!(
            weights, biases,
            layer + 1,
            A_next, c_next,
            D_full, g_full,
            [q_path; [q_curr]],
            x_int,
            results,
            make_model,
        )

        # Facet-flipping: flip each non-redundant local constraint to reach neighbours.
        active = _active_local_indices(D_full, g_full, size(D_prev, 1), make_model)
        for i in active
            q_n = copy(q_curr)
            q_n[i] = !q_n[i]
            if q_n ∉ visited
                push!(visited, q_n)
                push!(queue, q_n)
            end
        end
    end
end

function _find_regions_exact_impl(py_weights, py_biases, py_x0, make_model)
    weights, biases = _to_weights(py_weights, py_biases)
    x0 = Vector{Float64}(py_x0)

    n_layers    = length(weights)
    layer_sizes = [size(w, 1) for w in weights]
    total_bits  = sum(layer_sizes)
    input_dim   = size(weights[1], 2)

    offsets    = Vector{Int64}(undef, n_layers + 1)
    offsets[1] = 0
    for l in 1:n_layers
        offsets[l + 1] = offsets[l] + layer_sizes[l]
    end

    results = Vector{Tuple{Vector{Int8}, Vector{Float64}}}()
    _dfs_exact!(
        weights, biases,
        1,
        Matrix{Float64}(I, input_dim, input_dim),
        zeros(Float64, input_dim),
        Matrix{Float64}(undef, 0, input_dim),
        Float64[],
        BitVector[],
        x0,
        results,
        make_model,
    )

    n_regions = length(results)
    patterns  = Matrix{Int8}(undef,    n_regions, total_bits)
    centroids = Matrix{Float64}(undef, n_regions, input_dim)

    for (i, (path, centroid)) in enumerate(results)
        patterns[i, :]  = path
        centroids[i, :] = centroid
    end

    return (patterns, offsets, centroids)
end

"""
    find_regions_exact(py_weights, py_biases, py_x0)
        -> (patterns, offsets, centroids)

Exactly enumerate all feasible linear regions via DFS + facet-flipping.

`py_x0` is a starting point in input space, shape `(input_dim,)`.  The DFS
begins from the region containing `x0` and traverses all reachable neighbours
by flipping active halfspace constraints at each layer.

Return format is identical to `find_regions_sparse`.
"""
find_regions_exact(py_weights, py_biases, py_x0) =
    _find_regions_exact_impl(py_weights, py_biases, py_x0, _make_model_standard)

"""
    find_regions_exact_fast(py_weights, py_biases, py_x0)
        -> (patterns, offsets, centroids)

Faster variant: identical algorithm to `find_regions_exact`, but constructs
LPs with `direct_model(HiGHS.Optimizer())` instead of `Model(HiGHS.Optimizer)`,
skipping JuMP's caching/bridge layer (~6× faster on small LPs).
"""
find_regions_exact_fast(py_weights, py_biases, py_x0) =
    _find_regions_exact_impl(py_weights, py_biases, py_x0, _make_model_direct)
