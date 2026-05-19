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

end # module
