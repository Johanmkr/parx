"""
    find_regions_sparse(py_weights, py_biases, py_points)
        -> (patterns, offsets, centroids)

Find all linear regions observed in `py_points` via parallel forward passes.

Each unique activation path corresponds to one linear region.

Returns
-------
patterns  : (n_regions, total_bits) Int8 matrix — flattened activation bits,
            all layers concatenated.
offsets   : (n_layers+1,) Int64 vector — 0-indexed column boundaries such that
            `patterns[i, offsets[l]+1 : offsets[l+1]]` (Julia 1-indexed) or
            `patterns[i, offsets[l] : offsets[l+1]]` (Python 0-indexed) gives
            the activation at layer l for region i.
centroids : (n_regions, input_dim) Float64 matrix — the first data point routed
            to each region, used as a feasible interior point.
"""
function find_regions_sparse(py_weights, py_biases, py_points)
    weights, biases = _to_weights(py_weights, py_biases)
    points = Matrix{Float64}(py_points)   # (N, input_dim)

    N           = size(points, 1)
    n_layers    = length(weights)
    layer_sizes = [size(w, 1) for w in weights]
    total_bits  = sum(layer_sizes)

    # 0-indexed column boundaries (Python-compatible)
    offsets    = Vector{Int64}(undef, n_layers + 1)
    offsets[1] = 0
    for l in 1:n_layers
        offsets[l + 1] = offsets[l] + layer_sizes[l]
    end

    # ── Parallel forward pass ─────────────────────────────────────────────────
    all_paths = Vector{Vector{Int8}}(undef, N)

    Threads.@threads for i in 1:N
        x    = Vector{Float64}(points[i, :])
        path = Vector{Int8}(undef, total_bits)
        a    = x
        for l in 1:n_layers
            z = weights[l] * a + biases[l]
            q = Int8.(z .> 0)
            # offsets are 0-indexed; Julia slices are 1-indexed and inclusive
            path[offsets[l] + 1 : offsets[l + 1]] = q
            a = q .* z   # ReLU: active neurons pass z through, others → 0
        end
        all_paths[i] = path
    end

    # ── Deduplication (sequential) ────────────────────────────────────────────
    seen           = Dict{Vector{Int8}, Int}()
    unique_indices = Int[]

    for i in 1:N
        if !haskey(seen, all_paths[i])
            seen[all_paths[i]] = i
            push!(unique_indices, i)
        end
    end

    n_unique  = length(unique_indices)
    input_dim = size(points, 2)
    patterns  = Matrix{Int8}(undef,    n_unique, total_bits)
    centroids = Matrix{Float64}(undef, n_unique, input_dim)

    for (i, idx) in enumerate(unique_indices)
        patterns[i, :]  = all_paths[idx]
        centroids[i, :] = points[idx, :]
    end

    return (patterns, offsets, centroids)
end
