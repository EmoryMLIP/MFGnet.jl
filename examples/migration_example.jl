"""
Example: Using DifferentialEquations.jl and Optimization.jl interfaces

This file demonstrates how to use the new Julia ecosystem integrations:
1. DifferentialEquations.jl for ODE solving
2. Optimization.jl for optimization

Both maintain backward compatibility with the original implementation.
"""

using MFGnet
using LinearAlgebra
using Printf
import Flux  # For Flux.params
using DifferentialEquations: Tsit5  # For adaptive solver examples
using Optimization, OptimizationOptimJL  # For optimization examples

#=============================================================================
Setup Problem (Same as before)
=============================================================================#

# Problem dimensions
d = 2           # Spatial dimension
nex = 100       # Number of training examples

# Initial distribution and samples
function rho0(x)
    return exp.(-0.5 * sum(x.^2, dims=1))
end

X0 = randn(d, nex)
w = ones(nex) / nex

# Interaction and terminal costs
F(U, t) = zeros(size(U, 2))
G(U) = sum((U[1:d,:] .- 1.0).^2, dims=1)

# Neural network potential
ΘN = ((randn(20, d+1), randn(20)), (randn(1, 20), zeros(1)))
N = NN([SingleLayer(), SingleLayer()])
Φ = PotentialNN(N)

# Initial parameters
w0 = randn(1, 1)
A0 = randn(d+1, d+1)
b0 = randn(d+1)
z0 = randn(1)
Θ_init = (w0, ΘN, A0, b0, z0)

#=============================================================================
Example 1: Legacy ODE Solver (Baseline)
=============================================================================#

println("\n" * "="^70)
println("Example 1: Legacy ODE Solver (Baseline)")
println("="^70)

# Create MFG problem with legacy solver
J_legacy = MeanFieldGame(F, G, X0, rho0, w;
                         Φ=Φ,
                         α=ones(5),
                         stepper=RK4Step(),
                         tspan=[0.0, 1.0],
                         nt=10)

# Evaluate objective (legacy path)
@time obj_legacy = J_legacy(Θ_init)
@printf("Legacy objective: %.6e\n", obj_legacy)

#=============================================================================
Example 2: DifferentialEquations.jl with Fixed-Step RK4 (Should Match Legacy)
=============================================================================#

println("\n" * "="^70)
println("Example 2: DifferentialEquations.jl with Fixed-Step RK4")
println("="^70)

# Create config matching legacy behavior
config_rk4 = RK4Config(10, [0.0, 1.0])

# Evaluate using DifferentialEquations.jl
@time obj_diffeq_rk4 = J_legacy(Θ_init; use_diffeq=true, diffeq_config=config_rk4)
@printf("DiffEq RK4 objective: %.6e\n", obj_diffeq_rk4)
@printf("Relative difference: %.6e\n", abs(obj_diffeq_rk4 - obj_legacy) / abs(obj_legacy))

#=============================================================================
Example 3: DifferentialEquations.jl with Adaptive Solver
=============================================================================#

println("\n" * "="^70)
println("Example 3: DifferentialEquations.jl with Adaptive Tsit5")
println("="^70)

# Create adaptive config
config_adaptive = AdaptiveConfig(
    alg=Tsit5(),
    reltol=1e-6,
    abstol=1e-8
)

# Evaluate with adaptive solver
@time obj_adaptive = J_legacy(Θ_init; use_diffeq=true, diffeq_config=config_adaptive)
@printf("Adaptive objective: %.6e\n", obj_adaptive)
@printf("Relative difference from legacy: %.6e\n", abs(obj_adaptive - obj_legacy) / abs(obj_legacy))

#=============================================================================
Example 4: Training with Legacy BFGS
=============================================================================#

println("\n" * "="^70)
println("Example 4: Training with Legacy BFGS")
println("="^70)

# Setup for legacy optimization
parms = MFGnet.myMap(x->x, Θ_init)
ps = Flux.params(parms)
Θ0_vec = MFGnet.param2vec(parms)

# Objective and gradient functions
f = (Θ) -> MFGnet.evalObj(J_legacy, Θ, parms, ps)
fdf = (Θ) -> MFGnet.evalObjAndGrad(J_legacy, Θ, parms, ps)

# Run BFGS (just 5 iterations for demonstration)
@time Θ_opt_legacy, flag, his, X, H = MFGnet.bfgs(
    f, fdf, Θ0_vec;
    maxIter=5,
    atol=1e-10,
    out=1
)

println("\nLegacy BFGS converged with flag = $flag")

#=============================================================================
Example 5: Training with Optimization.jl L-BFGS
=============================================================================#

println("\n" * "="^70)
println("Example 5: Training with Optimization.jl L-BFGS")
println("="^70)

using Optimization, OptimizationOptimJL

# Create optimization problem
prob, parms_opt = create_optimization_problem(J_legacy, Θ_init, parms, ps)

# Solve with L-BFGS
@time sol = solve(prob, LBFGS(); maxiters=5, show_trace=true)

println("\nOptimization.jl converged with status: $(sol.retcode)")
@printf("Final objective: %.6e\n", sol.objective)

#=============================================================================
Example 6: Training with Optimization.jl + DifferentialEquations.jl
=============================================================================#

println("\n" * "="^70)
println("Example 6: Training with Optimization.jl + Adaptive ODE Solver")
println("="^70)

# Create optimization problem with DiffEq backend
prob_adaptive, parms_adaptive = create_optimization_problem(
    J_legacy, Θ_init, parms, ps;
    use_diffeq=true,
    diffeq_config=config_adaptive
)

# Solve with L-BFGS and adaptive ODE solver
@time sol_adaptive = solve(prob_adaptive, LBFGS(); maxiters=5, show_trace=true)

println("\nHybrid approach converged with status: $(sol_adaptive.retcode)")
@printf("Final objective: %.6e\n", sol_adaptive.objective)

#=============================================================================
Example 7: High-Level train_mfg Interface
=============================================================================#

println("\n" * "="^70)
println("Example 7: High-Level train_mfg Interface")
println("="^70)

# Create callback for monitoring
callback, history = create_mfg_callback(
    print_every=1,
    save_best=true,
    save_history=true
)

# Train using high-level interface
result = train_mfg(J_legacy, Θ_init;
                   optimizer=LBFGS(),
                   maxiters=5,
                   abstol=1e-10,
                   callback=callback)

println("\nHigh-level training completed")
@printf("Final objective: %.6e\n", result.objective)
@printf("Return code: %s\n", result.retcode)
@printf("Best objective: %.6e (iter %d)\n",
        history[:best_objective], history[:best_iteration])

#=============================================================================
Summary
=============================================================================#

println("\n" * "="^70)
println("Migration Summary")
println("="^70)
println("\n✓ All examples completed successfully!")
println("\nKey Features Demonstrated:")
println("  1. Backward compatibility with legacy solvers")
println("  2. DifferentialEquations.jl integration")
println("  3. Fixed-step and adaptive ODE solvers")
println("  4. Optimization.jl integration")
println("  5. L-BFGS for memory-efficient optimization")
println("  6. High-level train_mfg interface")
println("  7. Callback system for monitoring")
println("\nNext Steps:")
println("  - Run comprehensive test suite")
println("  - Benchmark performance improvements")
println("  - Migrate existing examples")
println("="^70)

