"""
Utility functions for testing MFG solvers

Provides functions for:
- Comparing objectives and gradients
- Finite difference validation
- Taylor tests
- Adjoint tests
- Mass conservation checks
"""
module TestUtils

using Test
using LinearAlgebra
using Printf

export compare_objectives, compare_gradients, compare_states
export finite_difference_gradient, taylor_test, adjoint_test
export mass_conservation_test, vec_params, reconstruct_params
export TestTolerances, MIGRATION_TOLERANCES, REFERENCE_TOLERANCES

# ============================================================================
# Tolerance Specifications
# ============================================================================

"""
Tolerance levels for different types of comparisons
"""
struct TestTolerances{R<:Real}
    objective_abstol::R
    objective_reltol::R
    gradient_abstol::R
    gradient_reltol::R
    state_abstol::R
    state_reltol::R
    fd_reltol::R
end

# Standard tolerances for migration tests (relaxed for solver differences)
const MIGRATION_TOLERANCES = TestTolerances(
    objective_abstol = 1e-6,
    objective_reltol = 1e-3,
    gradient_abstol = 1e-6,
    gradient_reltol = 1e-3,
    state_abstol = 1e-6,
    state_reltol = 1e-3,
    fd_reltol = 1e-3
)

# Strict tolerances for reference solutions
const REFERENCE_TOLERANCES = TestTolerances(
    objective_abstol = 1e-10,
    objective_reltol = 1e-8,
    gradient_abstol = 1e-8,
    gradient_reltol = 1e-6,
    state_abstol = 1e-8,
    state_reltol = 1e-6,
    fd_reltol = 1e-4
)

# ============================================================================
# Comparison Functions
# ============================================================================

"""
    compare_objectives(Jc1, Jc2; abstol, reltol, name)

Compare two objective function values with specified tolerances.

Returns NamedTuple with abs_error, rel_error, and passed flag.
"""
function compare_objectives(Jc1, Jc2; abstol=1e-6, reltol=1e-3, name="objective")
    abs_err = abs(Jc1 - Jc2)
    rel_err = abs_err / (abs(Jc1) + 1e-10)

    passed = abs_err < abstol || rel_err < reltol

    if !passed
        @warn "$name comparison failed" Jc1 Jc2 abs_err rel_err abstol reltol
    end

    @test passed

    return (abs_error=abs_err, rel_error=rel_err, passed=passed)
end

"""
    compare_gradients(∇1, ∇2; abstol, reltol, name)

Compare two gradient structures with specified tolerances.

Handles nested tuple structures and flattens for comparison.
"""
function compare_gradients(∇1, ∇2; abstol=1e-6, reltol=1e-3, name="gradient")
    v1 = vec_params(∇1)
    v2 = vec_params(∇2)

    abs_err = norm(v1 - v2)
    rel_err = abs_err / (norm(v1) + 1e-10)

    passed = abs_err < abstol || rel_err < reltol

    if !passed
        @warn "$name comparison failed" norm_grad1=norm(v1) norm_grad2=norm(v2) abs_err rel_err
        # Print per-component error for debugging
        comp_err = abs.(v1 - v2)
        max_idx = argmax(comp_err)
        @warn "Max component error at index $max_idx" v1[max_idx] v2[max_idx] comp_err[max_idx]
    end

    @test passed

    return (abs_error=abs_err, rel_error=rel_err, passed=passed)
end

"""
    compare_states(U1, U2; abstol, reltol, name)

Compare two state matrices/arrays.
"""
function compare_states(U1, U2; abstol=1e-6, reltol=1e-3, name="state")
    abs_err = norm(U1 - U2)
    rel_err = abs_err / (norm(U1) + 1e-10)

    passed = abs_err < abstol || rel_err < reltol

    if !passed
        @warn "$name comparison failed" size(U1) size(U2) abs_err rel_err
    end

    @test passed

    return (abs_error=abs_err, rel_error=rel_err, passed=passed)
end

# ============================================================================
# Parameter Flattening/Reconstruction
# ============================================================================

"""
    vec_params(Θ)

Flatten nested parameter structure to vector.

Handles tuples and arrays recursively.
"""
function vec_params(Θ)
    if Θ isa Tuple
        return vcat([vec_params(θ) for θ in Θ]...)
    elseif Θ isa AbstractArray
        return vec(Θ)
    elseif Θ isa Number
        return [Θ]
    else
        error("Unknown parameter type: $(typeof(Θ))")
    end
end

"""
    reconstruct_params(v_flat, template)

Reconstruct parameter structure from flat vector using template.
"""
function reconstruct_params(v_flat::AbstractVector, template)
    idx_ref = Ref(1)

    function reconstruct_level(t)
        if t isa Tuple
            return tuple([reconstruct_level(ti) for ti in t]...)
        elseif t isa AbstractArray
            n = length(t)
            result = reshape(v_flat[idx_ref[]:idx_ref[]+n-1], size(t))
            idx_ref[] += n
            return result
        elseif t isa Number
            val = v_flat[idx_ref[]]
            idx_ref[] += 1
            return val
        else
            error("Unknown type in template: $(typeof(t))")
        end
    end

    return reconstruct_level(template)
end

"""
    count_params(Θ)

Count total number of scalar parameters.
"""
function count_params(Θ)
    return length(vec_params(Θ))
end

# ============================================================================
# Finite Difference Gradient
# ============================================================================

"""
    finite_difference_gradient(f, Θ; h, method)

Compute gradient via finite differences for validation.

# Methods
- `:forward` - Forward difference: (f(θ+h) - f(θ))/h
- `:central` - Central difference: (f(θ+h) - f(θ-h))/(2h) (more accurate)
"""
function finite_difference_gradient(f, Θ; h=1e-5, method=:forward)
    f0 = f(Θ)

    function perturb_param(Θ, idx, delta)
        v = vec_params(Θ)
        v_pert = copy(v)
        v_pert[idx] += delta
        return reconstruct_params(v_pert, Θ)
    end

    n_params = count_params(Θ)
    grad_flat = zeros(n_params)

    for i in 1:n_params
        if method == :forward
            Θ_plus = perturb_param(Θ, i, h)
            f_plus = f(Θ_plus)
            grad_flat[i] = (f_plus - f0) / h

        elseif method == :central
            Θ_plus = perturb_param(Θ, i, h)
            Θ_minus = perturb_param(Θ, i, -h)
            f_plus = f(Θ_plus)
            f_minus = f(Θ_minus)
            grad_flat[i] = (f_plus - f_minus) / (2h)

        else
            error("Unknown method: $method")
        end
    end

    return reconstruct_params(grad_flat, Θ)
end

# ============================================================================
# Taylor Test
# ============================================================================

"""
    taylor_test(f, Θ; verbose, n_tests)

Taylor test for gradient correctness.

Verifies: f(θ+hv) = f(θ) + h⟨∇f(θ),v⟩ + O(h²)

# Returns
- errors_0: Zero-th order errors (should be O(h))
- errors_1: First-order errors (should be O(h²))
- hs: Step sizes tested
"""
function taylor_test(f, ∇f, Θ; verbose=true, n_tests=10)
    # Compute function and gradient at Θ
    f0 = f(Θ)
    g0 = ∇f(Θ)

    # Random direction
    v = random_direction(Θ)

    # Directional derivative ⟨∇f, v⟩
    dv = dot_params(g0, v)

    errors_0 = Float64[]
    errors_1 = Float64[]
    hs = Float64[]

    for k in 1:n_tests
        h = 2.0^(-k)
        push!(hs, h)

        Θ_pert = add_direction(Θ, v, h)
        fh = f(Θ_pert)

        # Zero-th order error (should be O(h))
        err0 = abs(fh - f0)
        push!(errors_0, err0)

        # First-order error (should be O(h²))
        err1 = abs(fh - f0 - h * dv)
        push!(errors_1, err1)
    end

    if verbose
        println("\nTaylor Test Results:")
        println("="^70)
        println("h\t\t|E0|\t\t|E1|\t\tE0/h\t\tE1/h²")
        println("-"^70)
        for i in 1:length(hs)
            @printf("%.2e\t%.2e\t%.2e\t%.4f\t\t%.4f\n",
                    hs[i], errors_0[i], errors_1[i],
                    errors_0[i]/hs[i], errors_1[i]/hs[i]^2)
        end
        println("="^70)
    end

    # Test convergence rates (check last few where numerical errors aren't dominant)
    if n_tests >= 4
        idx = n_tests - 2  # Use second-to-last
        ratio_0 = errors_0[idx] / errors_0[idx+1]
        ratio_1 = errors_1[idx] / errors_1[idx+1]

        # E0 ~ h: ratio should be ~2 when h is halved
        # E1 ~ h²: ratio should be ~4 when h is halved
        @test 1.5 < ratio_0 < 2.5
        @test 3.0 < ratio_1 < 5.0
    end

    return (errors_0=errors_0, errors_1=errors_1, hs=hs)
end

# Helper functions for Taylor test

"""Generate random direction matching parameter structure"""
function random_direction(Θ)
    if Θ isa Tuple
        return tuple([random_direction(θ) for θ in Θ]...)
    elseif Θ isa AbstractArray
        return randn(eltype(Θ), size(Θ)...)
    elseif Θ isa Number
        return randn(typeof(Θ))
    else
        error("Unknown type: $(typeof(Θ))")
    end
end

"""Add scaled direction to parameters: Θ + h*v"""
function add_direction(Θ, v, h)
    if Θ isa Tuple
        return tuple([add_direction(Θ[i], v[i], h) for i in 1:length(Θ)]...)
    elseif Θ isa AbstractArray
        return Θ .+ h .* v
    elseif Θ isa Number
        return Θ + h * v
    else
        error("Unknown type: $(typeof(Θ))")
    end
end

"""Dot product for parameter structures: ⟨Θ1, Θ2⟩"""
function dot_params(Θ1, Θ2)
    if Θ1 isa Tuple
        return sum([dot_params(Θ1[i], Θ2[i]) for i in 1:length(Θ1)])
    elseif Θ1 isa AbstractArray
        return dot(vec(Θ1), vec(Θ2))
    elseif Θ1 isa Number
        return Θ1 * Θ2
    else
        error("Unknown type: $(typeof(Θ1))")
    end
end

# ============================================================================
# Adjoint Test
# ============================================================================

"""
    adjoint_test(J_forward, J_adjoint, v, w)

Test adjoint consistency: ⟨v, J'w⟩ = ⟨Jv, w⟩

# Arguments
- `J_forward(v)`: Forward operator v → Jv
- `J_adjoint(w)`: Adjoint operator w → J'w
- `v`: Input vector
- `w`: Output vector
"""
function adjoint_test(J_forward, J_adjoint, v, w; R=Float64, name="operator")
    Jv = J_forward(v)
    JTw = J_adjoint(w)

    lhs = dot(vec(v), vec(JTw))
    rhs = dot(vec(Jv), vec(w))

    abs_err = abs(lhs - rhs)
    rel_err = abs_err / (abs(lhs) + abs(rhs) + 1e-10)

    tol = sqrt(eps(R)) * 100

    passed = rel_err < tol

    if !passed
        @warn "$name adjoint test failed" lhs rhs abs_err rel_err tol
    end

    @test passed

    return (lhs=lhs, rhs=rhs, abs_error=abs_err, rel_error=rel_err)
end

# ============================================================================
# Physical Property Tests
# ============================================================================

"""
    mass_conservation_test(mfg, Θ; tol)

Test mass conservation: ∫ρ(T) = ∫ρ(0)

Note: This assumes particles carry constant mass and uses Monte Carlo integration.
"""
function mass_conservation_test(mfg, Θ; tol=1e-3)
    # Solve MFG problem
    Jc = mfg(Θ)

    # Initial mass (Monte Carlo estimate)
    mass_initial = sum(mfg.w .* mfg.rho0x)

    # Final mass
    # The log-determinant in UN tracks density evolution
    # ρ(x(T)) = ρ(x(0)) * exp(log_det)
    d = size(mfg.X0, 1)
    log_det_final = mfg.UN[d+1, :]
    rho_final = mfg.rho0x .* exp.(vec(log_det_final))
    mass_final = sum(mfg.w .* rho_final)

    abs_err = abs(mass_final - mass_initial)
    rel_err = abs_err / mass_initial

    passed = rel_err < tol

    if !passed
        @warn "Mass conservation test failed" mass_initial mass_final abs_err rel_err tol
    end

    @test passed

    return (initial=mass_initial, final=mass_final, abs_error=abs_err, rel_error=rel_err)
end

"""
    energy_consistency_test(mfg, Θ; tol)

Test energy/Hamiltonian consistency (if applicable).
"""
function energy_consistency_test(mfg, Θ; tol=1e-3)
    # For MFG, the Hamiltonian is: H = ⟨p, v⟩ - L(x, v)
    # This is problem-dependent
    @warn "Energy consistency test not yet implemented"
    return nothing
end

end # module TestUtils
