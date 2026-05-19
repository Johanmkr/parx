module LinearRegions

using LinearAlgebra

# ── Internal helpers ──────────────────────────────────────────────────────────

# Numpy arrays passed via juliacall implement AbstractArray; constructing
# Matrix{Float64}/Vector{Float64} copies into column-major Julia storage
# while preserving the (out_features, in_features) indexing convention.
function _to_weights(py_weights, py_biases)
    weights = [Matrix{Float64}(w) for w in py_weights]
    biases  = [Vector{Float64}(b) for b in py_biases]
    return weights, biases
end

# ── Phase 1: smoke tests & bridge verification ────────────────────────────────

function hello()
    return "LinearRegions.jl is loaded and running on $(Threads.nthreads()) thread(s)."
end

"""
    network_info(py_weights, py_biases) -> (input_dim, n_layers)

Accepts weight/bias lists from Python and returns basic architecture info.
Serves as the Phase 1 smoke test for the Python↔Julia data bridge.
"""
function network_info(py_weights, py_biases)
    weights, biases = _to_weights(py_weights, py_biases)
    input_dim = size(weights[1], 2)
    n_layers  = length(weights)
    return (input_dim, n_layers)
end

# ── Phase 2: sparse region finder ────────────────────────────────────────────

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

    N         = size(points, 1)
    n_layers  = length(weights)
    layer_sizes = [size(w, 1) for w in weights]   # neurons at each layer
    total_bits  = sum(layer_sizes)

    # 0-indexed column boundaries (Python-compatible)
    offsets = Vector{Int64}(undef, n_layers + 1)
    offsets[1] = 0
    for l in 1:n_layers
        offsets[l + 1] = offsets[l] + layer_sizes[l]
    end

    # ── Parallel forward pass ─────────────────────────────────────────────
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

    # ── Deduplication (sequential) ────────────────────────────────────────
    # Map each unique path to the index of the first point that produced it.
    seen          = Dict{Vector{Int8}, Int}()
    unique_indices = Int[]

    for i in 1:N
        if !haskey(seen, all_paths[i])
            seen[all_paths[i]] = i
            push!(unique_indices, i)
        end
    end

    n_unique = length(unique_indices)

    # ── Pack outputs ──────────────────────────────────────────────────────
    input_dim = size(points, 2)
    patterns  = Matrix{Int8}(undef,    n_unique, total_bits)
    centroids = Matrix{Float64}(undef, n_unique, input_dim)

    for (i, idx) in enumerate(unique_indices)
        patterns[i, :]  = all_paths[idx]
        centroids[i, :] = points[idx, :]
    end

    return (patterns, offsets, centroids)
end

end # module
