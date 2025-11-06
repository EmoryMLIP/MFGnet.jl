"""
Optimization.jl Interface for MFGnet

Provides compatibility layer between MFGnet's custom BFGS optimizer and the Optimization.jl
ecosystem, enabling:
- L-BFGS (memory-efficient quasi-Newton)
- Multiple optimizer choices (Newton, ConjugateGradient, Adam, etc.)
- Unified interface for all optimizers
- Backward compatibility with custom BFGS

# Exports
- `create_optimization_problem` - Wrap MFG objective for Optimization.jl
- `train_mfg` - High-level training function
- `create_mfg_callback` - Create callbacks for monitoring
"""

using Optimization
using OptimizationOptimJL
using Zygote
using Printf
using LinearAlgebra
import Flux  # For Flux.params

export create_optimization_problem, train_mfg, create_mfg_callback

#=============================================================================
Optimization Problem Creation
=============================================================================#

"""
    create_optimization_problem(J::MeanFieldGame, Θ_init, parms, ps; use_diffeq=false, diffeq_config=nothing)

Create an OptimizationProblem from MFG objective function

# Arguments
- `J::MeanFieldGame`: MFG problem structure
- `Θ_init`: Initial parameter structure (nested tuples)
- `parms`: Mutable parameter copies for Zygote
- `ps`: Flux.params tracking structure
- `use_diffeq::Bool`: Use DifferentialEquations.jl (default: false)
- `diffeq_config`: DiffEq configuration (default: nothing)

# Returns
- `prob::OptimizationProblem`: Ready to solve with Optimization.solve
- `parms`: Parameter structure (for reconstruction after optimization)

# Example
```julia
using Optimization, OptimizationOptimJL

# Create problem
prob, parms = create_optimization_problem(J, Θ, parms, ps)

# Solve with L-BFGS
sol = solve(prob, LBFGS(); maxiters=200)

# Extract optimized parameters
Θ_opt = vec2param!(sol.u, parms)
```
"""
function create_optimization_problem(J::MeanFieldGame, Θ_init, parms, ps;
                                    use_diffeq=false, diffeq_config=nothing)
    # Flatten initial parameters
    u0 = param2vec(Θ_init)

    # Store config in problem parameters
    p = (J=J, parms=parms, ps=ps, use_diffeq=use_diffeq, diffeq_config=diffeq_config)

    # Define objective function
    function objective(u, p)
        J, parms, ps = p.J, p.parms, p.ps

        if p.use_diffeq && !isnothing(p.diffeq_config)
            # Use DifferentialEquations.jl path
            parms_local = vec2param!(copy(u), deepcopy(parms))
            sol = solve_mfg_ode(J, parms_local, p.diffeq_config)
            (d, nex) = size(J.X0)
            UN = extract_final_state(sol, d, nex)

            # Compute costs (same as MeanFieldGame functor)
            R = eltype(UN)
            costL = dot(vec(UN[end-2,:]), J.w)
            costF = dot(vec(UN[end-1,:]), J.w)
            costG = dot(J.G(UN), J.w)
            costHJ = dot(vec(UN[end,:]), J.w)
            phi1 = vec(J.Φ([UN[1:d,:]; fill(R(1.0), 1, size(J.X0,2))], parms_local))
            costHJf = dot(abs.(phi1 - J.α[3].*vec(getDeltaG(J.G, UN))), J.w)

            cs = [costL, costF, costG, costHJ, costHJf]
            Jc = dot(J.α, cs)
            return Jc
        else
            # Use legacy path
            return evalObj(J, u, parms, ps)
        end
    end

    # Define gradient function
    function gradient!(G, u, p)
        J, parms, ps = p.J, p.parms, p.ps

        if p.use_diffeq && !isnothing(p.diffeq_config)
            # Use Zygote with DifferentialEquations.jl
            ∇ = Zygote.gradient(u) do u_inner
                objective(u_inner, p)
            end
            G .= ∇[1]
        else
            # Use legacy gradient computation
            _, grad = evalObjAndGrad(J, u, parms, ps)
            G .= grad
        end
        return nothing
    end

    # Create optimization function with AD backend
    optf = OptimizationFunction(objective, grad=gradient!)

    # Create problem
    prob = OptimizationProblem(optf, u0, p)

    return prob, parms
end

#=============================================================================
High-Level Training Interface
=============================================================================#

"""
    train_mfg(J::MeanFieldGame, Θ_init; optimizer=LBFGS(), kwargs...)

High-level interface for training MFG models

# Arguments
- `J::MeanFieldGame`: Problem structure
- `Θ_init`: Initial parameters (nested tuples)

# Keyword Arguments
- `optimizer`: Optimization.jl optimizer (default: LBFGS())
- `maxiters::Int`: Maximum iterations (default: 100)
- `abstol::Real`: Absolute tolerance (default: 1e-8)
- `reltol::Real`: Relative tolerance (default: 1e-6)
- `callback`: Optimization callback (default: nothing)
- `use_diffeq::Bool`: Use DifferentialEquations.jl (default: false)
- `diffeq_config`: DiffEq configuration (default: nothing)

# Returns
Named tuple with:
- `Θ_opt`: Optimized parameters (nested structure)
- `objective`: Final objective value
- `retcode`: Return code (:Success, :MaxIters, etc.)
- `original_result`: Full OptimizationSolution object

# Example
```julia
# Basic usage with L-BFGS
result = train_mfg(J, Θ_init; optimizer=LBFGS(), maxiters=200)

# With adaptive ODE solver
config = AdaptiveConfig(alg=Tsit5(), reltol=1e-6)
result = train_mfg(J, Θ_init;
                   optimizer=LBFGS(),
                   maxiters=200,
                   use_diffeq=true,
                   diffeq_config=config)

# Access results
Θ_opt = result.Θ_opt
final_obj = result.objective
```
"""
function train_mfg(J::MeanFieldGame, Θ_init;
                   optimizer=LBFGS(),
                   maxiters=100,
                   abstol=1e-8,
                   reltol=1e-6,
                   callback=nothing,
                   use_diffeq=false,
                   diffeq_config=nothing)

    # Create mutable parameter copies for Zygote
    parms = myMap(x->x, Θ_init)
    ps = Flux.params(parms)

    # Create optimization problem
    prob, parms = create_optimization_problem(J, Θ_init, parms, ps;
                                              use_diffeq=use_diffeq,
                                              diffeq_config=diffeq_config)

    # Solve
    sol = solve(prob, optimizer;
                maxiters=maxiters,
                abstol=abstol,
                reltol=reltol,
                callback=callback)

    # Reconstruct optimized parameters
    Θ_opt = vec2param!(sol.u, parms)

    # Return structured result
    return (
        Θ_opt=Θ_opt,
        objective=sol.objective,
        retcode=sol.retcode,
        original_result=sol
    )
end

#=============================================================================
Callback Creation
=============================================================================#

"""
    create_mfg_callback(; kwargs...)

Create a callback function for monitoring MFG training

# Keyword Arguments
- `validation_mfg::Union{Nothing,MeanFieldGame}`: Validation problem (default: nothing)
- `print_every::Int`: Print frequency (default: 1)
- `save_best::Bool`: Track best parameters (default: true)
- `save_history::Bool`: Save optimization history (default: true)
- `custom_metrics::Function`: Custom metric function (default: nothing)

# Returns
- `callback`: Callback function for Optimization.jl
- `history`: Shared history dictionary (mutable)

# Example
```julia
# Create callback with validation
callback, history = create_mfg_callback(
    validation_mfg=Jv,
    print_every=10,
    save_best=true
)

# Use in training
result = train_mfg(J, Θ_init;
                   optimizer=LBFGS(),
                   maxiters=200,
                   callback=callback)

# Access history
train_losses = history[:train_objective]
val_losses = history[:val_objective]
best_params = history[:best_params]
```
"""
function create_mfg_callback(;
    validation_mfg::Union{Nothing,MeanFieldGame}=nothing,
    print_every::Int=1,
    save_best::Bool=true,
    save_history::Bool=true,
    custom_metrics::Union{Nothing,Function}=nothing)

    # Shared state for callback
    history = Dict{Symbol,Any}(
        :train_objective => Float64[],
        :val_objective => Float64[],
        :grad_norm => Float64[],
        :iterations => Int[],
        :best_objective => Inf,
        :best_params => nothing,
        :best_iteration => 0
    )

    iter_count = Ref(0)

    function callback_fn(state, loss_val)
        iter_count[] += 1
        iter = iter_count[]

        # Store training objective
        if save_history
            push!(history[:iterations], iter)
            push!(history[:train_objective], loss_val)

            # Store gradient norm if available
            if hasfield(typeof(state), :gradient) && !isnothing(state.gradient)
                push!(history[:grad_norm], norm(state.gradient))
            end
        end

        # Compute validation objective
        val_obj = NaN
        if !isnothing(validation_mfg)
            parms_current = vec2param!(copy(state.u), history[:best_params])
            val_obj = validation_mfg(parms_current)
            if save_history
                push!(history[:val_objective], val_obj)
            end
        end

        # Track best parameters
        if save_best
            obj_to_compare = isnan(val_obj) ? loss_val : val_obj
            if obj_to_compare < history[:best_objective]
                history[:best_objective] = obj_to_compare
                history[:best_params] = copy(state.u)
                history[:best_iteration] = iter
            end
        end

        # Custom metrics
        if !isnothing(custom_metrics)
            custom_metrics(state, history, iter)
        end

        # Print progress
        if mod(iter, print_every) == 0
            @printf("Iter %4d: train=%.3e", iter, loss_val)
            if !isnothing(validation_mfg) && !isnan(val_obj)
                @printf("  val=%.3e", val_obj)
            end
            if hasfield(typeof(state), :gradient) && !isnothing(state.gradient)
                @printf("  |∇|=%.3e", norm(state.gradient))
            end
            if save_best
                @printf("  best=%.3e@%d", history[:best_objective], history[:best_iteration])
            end
            println()
        end

        return false  # Don't stop optimization
    end

    return callback_fn, history
end

#=============================================================================
Backward Compatibility with Custom BFGS
=============================================================================#

"""
    bfgs_legacy(args...; kwargs...)

Deprecated: Use `train_mfg` with `LBFGS()` optimizer instead

This function provides backward compatibility but emits a deprecation warning.
"""
function bfgs_legacy(f::Function, fdf::Function, x::Vector; kwargs...)
    @warn """
    bfgs_legacy is deprecated. Please use Optimization.jl interface instead:

    # Old code:
    Θopt, flag, his, X, H = bfgs(f, fdf, Θ0, maxIter=200)

    # New code:
    prob, parms = create_optimization_problem(J, Θ_init, parms, ps)
    sol = solve(prob, LBFGS(); maxiters=200)
    Θopt = vec2param!(sol.u, parms)

    See documentation for migration guide.
    """ maxlog=1

    # Call original BFGS
    return bfgs(f, fdf, x; kwargs...)
end

