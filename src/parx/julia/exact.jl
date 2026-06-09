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
    r_parent ::Float64,           # Chebyshev radius from the parent's LP
    results  ::Vector{Tuple{Vector{Int8}, Vector{Float64}, Vector{Int32}, Bool}},
    make_model,
)
    if layer > length(weights)
        path_flat     = reduce(vcat, [Int8.(q) for q in q_path])
        # Non-redundant rows of the complete halfspace system (0-based for Python).
        active_1based = _active_local_indices(D_prev, g_prev, 0, make_model)
        active_0based = isempty(active_1based) ? Int32[] : Int32.(active_1based .- 1)
        bounded       = r_parent < 1e3
        push!(results, (path_flat, copy(x_parent), active_0based, bounded))
        return
    end

    W, b  = weights[layer], biases[layer]
    W_hat = W * A_prev        # (n_out, input_dim)
    b_hat = W * c_prev + b    # (n_out,)

    q_start  = BitVector(W_hat * x_parent + b_hat .> 0)
    frontier = [q_start]
    visited  = Set{BitVector}([q_start])

    while !isempty(frontier)
        # Snapshot the frontier and process each item as an independent task.
        # Each task owns its local results and candidate neighbours.
        tasks = map(frontier) do q_curr
            Threads.@spawn begin
                s       = -2.0 .* Float64.(q_curr) .+ 1.0  # −1 if active, +1 if inactive
                D_local = s .* W_hat
                g_local = -(s .* b_hat)
                D_full  = vcat(D_prev, D_local)
                g_full  = vcat(g_prev, g_local)

                x_int, r = _chebyshev_center(D_full, g_full, make_model)
                isnothing(x_int) && return (nothing, BitVector[])

                local_results = Vector{Tuple{Vector{Int8}, Vector{Float64}, Vector{Int32}, Bool}}()
                _dfs_exact!(
                    weights, biases,
                    layer + 1,
                    Float64.(q_curr) .* W_hat,
                    Float64.(q_curr) .* b_hat,
                    D_full, g_full,
                    [q_path; [q_curr]],
                    x_int, r,
                    local_results,
                    make_model,
                )

                # Facet-flipping: collect candidate neighbours for this item.
                active = _active_local_indices(D_full, g_full, size(D_prev, 1), make_model)
                neighbors = [let q_n = copy(q_curr); q_n[i] = !q_n[i]; q_n end
                             for i in active]
                return (local_results, neighbors)
            end
        end

        # Merge results and build next frontier (deduplication is sequential).
        next_candidates = BitVector[]
        for t in tasks
            local_results, neighbors = fetch(t)
            isnothing(local_results) && continue
            append!(results, local_results)
            append!(next_candidates, neighbors)
        end

        frontier = BitVector[]
        for q_n in next_candidates
            if q_n ∉ visited
                push!(visited, q_n)
                push!(frontier, q_n)
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

    results = Vector{Tuple{Vector{Int8}, Vector{Float64}, Vector{Int32}, Bool}}()
    _dfs_exact!(
        weights, biases,
        1,
        Matrix{Float64}(I, input_dim, input_dim),
        zeros(Float64, input_dim),
        Matrix{Float64}(undef, 0, input_dim),
        Float64[],
        BitVector[],
        x0, Inf,
        results,
        make_model,
    )

    n_regions = length(results)
    patterns  = Matrix{Int8}(undef,    n_regions, total_bits)
    centroids = Matrix{Float64}(undef, n_regions, input_dim)
    bounded_arr = Vector{Bool}(undef, n_regions)

    # Build active_indices_flat and active_offsets.
    active_offsets_arr    = Vector{Int64}(undef, n_regions + 1)
    active_offsets_arr[1] = 0
    for i in 1:n_regions
        active_offsets_arr[i + 1] = active_offsets_arr[i] + length(results[i][3])
    end
    total_active        = Int(active_offsets_arr[end])
    active_indices_flat = Vector{Int32}(undef, total_active)

    for (i, (path, centroid, active_idxs, bnd)) in enumerate(results)
        patterns[i, :]  = path
        centroids[i, :] = centroid
        bounded_arr[i]  = bnd
        start = Int(active_offsets_arr[i]) + 1
        stop  = Int(active_offsets_arr[i + 1])
        active_indices_flat[start:stop] = active_idxs
    end

    return (patterns, offsets, centroids, active_indices_flat, active_offsets_arr, bounded_arr)
end

"""
    find_regions_exact(py_weights, py_biases, py_x0)
        -> (patterns, offsets, centroids, active_indices_flat, active_offsets, bounded)

Exactly enumerate all feasible linear regions via DFS + facet-flipping.

`py_x0` is a starting point in input space, shape `(input_dim,)`.  The DFS
begins from the region containing `x0` and traverses all reachable neighbours
by flipping active halfspace constraints at each layer.

Return format (first three fields are identical to `find_regions_sparse`):
- `patterns`:             (n_regions, total_bits)   Int8
- `offsets`:              (n_layers + 1,)            Int64
- `centroids`:            (n_regions, input_dim)     Float64
- `active_indices_flat`:  (k,)                       Int32  — 0-based row indices
- `active_offsets`:       (n_regions + 1,)           Int64  — region i owns [i:i+1]
- `bounded`:              (n_regions,)               Bool
"""
find_regions_exact(py_weights, py_biases, py_x0) =
    _find_regions_exact_impl(py_weights, py_biases, py_x0, _make_model_standard)

"""
    find_regions_exact_fast(py_weights, py_biases, py_x0)
        -> (patterns, offsets, centroids, active_indices_flat, active_offsets, bounded)

Faster variant: identical algorithm to `find_regions_exact`, but constructs
LPs with `direct_model(HiGHS.Optimizer())` instead of `Model(HiGHS.Optimizer)`,
skipping JuMP's caching/bridge layer (~6× faster on small LPs).
"""
find_regions_exact_fast(py_weights, py_biases, py_x0) =
    _find_regions_exact_impl(py_weights, py_biases, py_x0, _make_model_direct)
