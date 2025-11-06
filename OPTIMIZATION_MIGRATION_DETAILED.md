# MFGnet.jl Optimization Migration: Detailed Implementation Guide
**Migration to Optimization.jl Ecosystem**

**Version:** 2.0 - Implementation Ready
**Date:** 2025-11-06
**Author:** Code Architecture Agent
**Target Julia Version:** 1.10+

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current Implementation Analysis](#current-implementation-analysis)
3. [Migration Architecture](#migration-architecture)
4. [Concrete Code Examples](#concrete-code-examples)
5. [Callback Migration](#callback-migration)
6. [Step-by-Step Implementation Plan](#step-by-step-implementation-plan)
7. [Testing Strategy with Code](#testing-strategy-with-code)
8. [Risk Assessment](#risk-assessment)
9. [Performance Benchmarks](#performance-benchmarks)
10. [Appendix: Complete Examples](#appendix-complete-examples)

---

## Executive Summary

### Current State
MFGnet.jl uses a custom BFGS implementation in `/home/user/MFGnet.jl/src/bfgs.jl` with:
- Manual Hessian approximation updates
- Custom Armijo line search
- Integration with Zygote for gradients
- Parameter flattening/unflattening via `/home/user/MFGnet.jl/src/param2vec.jl`

### Target State
Migrate to Optimization.jl ecosystem to gain:
- **10+ optimizer choices** (LBFGS, BFGS, Newton, Adam, AdaGrad, etc.)
- **Unified interface** for easy algorithm swapping
- **Better maintenance** via community-supported packages
- **Automatic gradient handling** through multiple AD backends
- **Constraint support** for bounded/constrained optimization

### Backward Compatibility
- Keep custom BFGS as `bfgs_legacy` for 1 major version
- New code uses Optimization.jl by default
- Clear deprecation warnings with migration examples

---

## Current Implementation Analysis

### 1. Custom BFGS Implementation

**File:** `/home/user/MFGnet.jl/src/bfgs.jl`

**Key Components:**

```julia
# Current BFGS signature
function bfgs(f::Function, fdf::Function, x::Vector;
              H=Matrix(1.0I,length(x),length(x)),
              maxIter=20, atol=1e-8, out::Int=0,
              storeInterm::Bool=false,
              lineSearch::Function=(f,fk,dfk,xk,pk,ak)->armijo(f,fk,dfk,xk,pk,maxIter=30,t=ak),
              cb::Function=()->())
    # Returns: x, flag, his, X, H
end

# Armijo line search
function armijo(f::Function, fk, dfk, xk, pk;
                t=1.0, maxIter=10, c1=1e-4, b=0.5)
    # Returns: t, LS
end
```

**Limitations:**
1. Only BFGS available (no LBFGS, Newton, Adam, etc.)
2. Manual Hessian update (memory intensive for large problems)
3. Simple line search (no strong Wolfe conditions)
4. No constraint handling
5. Limited convergence diagnostics

### 2. Parameter Management

**File:** `/home/user/MFGnet.jl/src/param2vec.jl`

**Current Approach:**

```julia
# Flatten nested tuple structure to vector
Θ = ((K1, b1), (K2, b2), (A, c, z))  # Nested tuples
θ_vec = param2vec(Θ)                  # → Vector

# Unflatten vector back to structure
vec2param!(θ_vec, Θ)                  # In-place update
```

**Issues:**
- Nested tuple structure is opaque
- No named access to parameters
- Manual indexing for gradient extraction
- Type instability in some cases

### 3. Gradient Computation

**File:** `/home/user/MFGnet.jl/src/utils.jl`

**Current Workflow:**

```julia
function evalObjAndGrad(J, Θ::Vector, parms, ps)
    # 1. Unflatten vector to parameter structure
    parms = vec2param!(Θ, parms)

    # 2. Compute objective and gradient via Zygote
    Jc, back = Zygote.pullback(() -> J(parms), ps)
    gc = back(Zygote.sensitivity(Jc))

    # 3. Flatten gradients back to vector
    dJ = Θ .* 0.0
    cnt = 0
    for p in ps
        if !isnothing(gc[p])
            gp = vec(gc[p])
            dJ[cnt+1:cnt+length(gp)] = gp
            cnt += length(gp)
        end
    end

    return Jc, dJ
end
```

**Complexity:**
- Manual gradient extraction and flattening
- Requires maintaining both `parms` structure and `ps` Flux.params
- Error-prone indexing

### 4. Current Usage in Examples

**File:** `/home/user/MFGnet.jl/examples/ROLNWF2019/runObstacleExperiment.jl`

**Typical Usage Pattern:**

```julia
# Setup
parms = (w0, (ΘN), A0, b0, z0)  # Nested parameters
ps = Flux.params(parms)          # Flux params for Zygote
Θ0 = MFGnet.param2vec(parms)     # Flatten to vector

# Define objective wrappers
f   = (Θ) -> evalObj(J, Θ, parms, ps)
fdf = (Θ) -> evalObjAndGrad(J, Θ, parms, ps)

# Optimize
Θopt, flag, His, X, H = MFGnet.bfgs(
    f, fdf, Θ0,
    maxIter=200,
    out=0,
    atol=1e-10,
    cb=cbBFGS  # Custom callback
)

# Unflatten result
parms_opt = vec2param!(Θopt, parms)
```

**Callback Structure:**

```julia
# Complex callback for monitoring, validation, checkpointing
cb = function(J, Jv, iter, His, parms, doPlots=true)
    # 1. Evaluate validation loss
    Jvc = Jv(parms)

    # 2. Track best parameters
    if Jvc < bestLoss
        bestLoss = Jvc
        Θbest = copy(parms)
    end

    # 3. Log metrics
    His[iter,:] = [J.cs[1:5]..., Jv.cs[1:5]...]
    @printf("iter=%04d obj=%1.3e ...\n", iter, sum(His[iter,1:5]))

    # 4. Periodic sampling
    if mod(iter, sampleFreq) == 0
        J.X0 = sample(rho0, nTrain)
        # Update densities...
    end

    # 5. Checkpointing
    if mod(iter, saveIter) == 0
        save("checkpoint-iter$iter.jld", "Θ", Θbest, "His", His)
    end

    # 6. Visualization
    if doPlots
        # Create plots...
    end
end

cbBFGS = (iter) -> cb(J, Jv, iter, His, parms, doPlots)
```

---

## Migration Architecture

### Overview Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    MFGnet.jl Application                     │
│  - Define MeanFieldGame problem                              │
│  - Initialize neural network parameters Θ                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              ComponentArrays.jl Integration                  │
│  - Convert nested tuples to ComponentArray                   │
│  - Named parameter access: θ.layer1.K, θ.layer1.b           │
│  - Automatic gradient structure matching                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│           Optimization.jl OptimizationProblem                │
│  - loss(θ, p) = mfg(componentarray_to_params(θ))            │
│  - AD via Zygote (AutoZygote backend)                        │
│  - Unified callback interface                                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  Optimizer Selection                         │
│                                                              │
│  ┌─────────────────────────────────────────────────┐        │
│  │  OptimizationOptimJL (Optim.jl backend)         │        │
│  │  - LBFGS()     [DEFAULT for medium/large]       │        │
│  │  - BFGS()      [small problems <1000 params]    │        │
│  │  - Newton()    [very small, with Hessian]       │        │
│  │  - ConjugateGradient()                          │        │
│  └─────────────────────────────────────────────────┘        │
│                                                              │
│  ┌─────────────────────────────────────────────────┐        │
│  │  OptimizationOptimisers (Optimisers.jl)         │        │
│  │  - Adam(lr)    [large problems, stochastic]     │        │
│  │  - AdaGrad(lr)                                  │        │
│  │  - RMSProp(lr)                                  │        │
│  └─────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

#### 1. Use ComponentArrays for Parameter Management

**Why?**
- Named parameter access: `θ.layer1.weights` instead of manual indexing
- Automatic gradient structure matching
- Type-stable operations
- Better than nested tuples for optimization

**Example:**

```julia
# Old way (nested tuples)
Θ = ((K1, b1), (K2, b2))
θ_vec = param2vec(Θ)  # Manual flattening

# New way (ComponentArray)
θ = ComponentArray(
    layer1 = (K = K1, b = b1),
    layer2 = (K = K2, b = b2)
)
# Access: θ.layer1.K, θ.layer2.b
# Automatic vectorization: vec(θ)
```

#### 2. Keep Custom BFGS as Legacy Option

**Reasoning:**
- Ensure exact reproducibility of published results
- Gradual migration path
- Emergency fallback if Optimization.jl has issues

**Implementation:**

```julia
# Rename existing BFGS
bfgs_legacy(args...; kwargs...) = bfgs(args...; kwargs...)

# Add deprecation warning to original
function bfgs(args...; kwargs...)
    @warn """
    Direct use of `bfgs` is deprecated.
    Use `train_mfg` with `optimizer=LBFGS()` instead.
    For legacy behavior, use `bfgs_legacy`.
    """ maxlog=1
    return bfgs_legacy(args...; kwargs...)
end
```

#### 3. Wrap MFG Objective for Optimization.jl

**Strategy:**
- Create thin wrapper converting ComponentArray ↔ nested tuples
- Let Optimization.jl handle gradient computation
- Support both Zygote and ForwardDiff backends

---

## Concrete Code Examples

### Example 1: Basic Migration

**Before (Current):**

```julia
using MFGnet
using Flux, Zygote

# Setup problem
d = 2
nex = 100
X0 = randn(d, nex)
rho0(x) = ones(size(x, 2))
w = ones(nex) / nex

F = F0()
G = Gkl(rho0, rho0, rho0(X0), rho0(X0), 1.0)
mfg = MeanFieldGame(F, G, X0, rho0, w)

# Initialize parameters (nested tuples)
Φ = getPotentialResNet(4, 1.0, 4, Float64)
w0, ΘN, A0, b0, z0 = initializeWeights(d, 32, 4, identity)
parms = (w0, (ΘN), A0, b0, z0)
ps = Flux.params(parms)

# Flatten parameters
Θ0 = MFGnet.param2vec(parms)

# Define objective wrappers
f   = (Θ) -> evalObj(mfg, Θ, parms, ps)
fdf = (Θ) -> evalObjAndGrad(mfg, Θ, parms, ps)

# Optimize with custom BFGS
Θopt, flag, His, X, H = MFGnet.bfgs(
    f, fdf, Θ0,
    maxIter=200,
    atol=1e-10,
    out=1
)

# Unflatten result
parms_opt = vec2param!(Θopt, parms)
```

**After (Optimization.jl):**

```julia
using MFGnet
using Optimization
using OptimizationOptimJL  # For LBFGS

# Setup problem (same as before)
d = 2
nex = 100
X0 = randn(d, nex)
rho0(x) = ones(size(x, 2))
w = ones(nex) / nex

F = F0()
G = Gkl(rho0, rho0, rho0(X0), rho0(X0), 1.0)
mfg = MeanFieldGame(F, G, X0, rho0, w)

# Initialize parameters (same as before)
Φ = getPotentialResNet(4, 1.0, 4, Float64)
w0, ΘN, A0, b0, z0 = initializeWeights(d, 32, 4, identity)
Θ_init = (w0, (ΘN), A0, b0, z0)

# Train with Optimization.jl (one line!)
result = train_mfg(
    mfg, Θ_init,
    optimizer = LBFGS(),
    maxiters = 200,
    abstol = 1e-10,
    verbose = true
)

# Extract result
Θ_opt = result.Θ_opt
loss_history = result.loss_history
```

**Key Improvements:**
- **10 lines → 3 lines** for optimization setup
- No manual parameter flattening/unflattening
- No need to maintain separate `parms` and `ps`
- Automatic gradient computation
- Easy to switch optimizers

### Example 2: Using Different Optimizers

```julia
# === LBFGS (quasi-Newton, memory efficient) ===
result_lbfgs = train_mfg(mfg, Θ_init,
    optimizer = LBFGS(),
    maxiters = 500
)

# === BFGS (full quasi-Newton, small problems) ===
result_bfgs = train_mfg(mfg, Θ_init,
    optimizer = BFGS(),
    maxiters = 500
)

# === Adam (first-order, large problems) ===
result_adam = train_mfg(mfg, Θ_init,
    optimizer = Adam(0.001),  # Learning rate
    maxiters = 5000
)

# === Newton with Hessian (very accurate, small problems) ===
result_newton = train_mfg(mfg, Θ_init,
    optimizer = Newton(; linsolve=KrylovJL_GMRES()),
    maxiters = 100
)

# === Conjugate Gradient (no Hessian) ===
result_cg = train_mfg(mfg, Θ_init,
    optimizer = ConjugateGradient(),
    maxiters = 1000
)
```

### Example 3: Advanced Options

```julia
using Optimization
using OptimizationOptimJL
using ComponentArrays

# Manual control for advanced users
# Step 1: Create optimization problem
prob = create_optimization_problem(
    mfg, Θ_init,
    adtype = Optimization.AutoZygote(),  # AD backend
    use_diffeq = true,                   # Use DifferentialEquations.jl
    ode_solver = Tsit5(),                # ODE solver
    sensealg = QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))
)

# Step 2: Define custom callback
function custom_callback(state, loss_val)
    if state.iter % 10 == 0
        @printf("Iter %4d: loss = %.6e\n", state.iter, loss_val)

        # Custom logic
        if loss_val < 1e-6
            return true  # Stop optimization
        end
    end
    return false  # Continue
end

# Step 3: Solve with fine-grained control
sol = solve_optimization(
    prob, LBFGS(),
    maxiters = 1000,
    abstol = 1e-10,
    reltol = 1e-8,
    callback = custom_callback,
    verbose = true,
    # Optim.jl specific options
    g_tol = 1e-8,
    f_tol = 1e-8,
    x_tol = 1e-8
)

# Step 4: Extract results
Θ_opt = componentarray_to_params(sol.u, get_param_structure(Θ_init))
final_loss = sol.objective
converged = (sol.retcode == :Success)
```

---

## Callback Migration

### Current Callback Structure

**From:** `/home/user/MFGnet.jl/examples/ROLNWF2019/runObstacleExperiment.jl`

```julia
# Current callback signature: cb(J, Jv, iter, His, parms, doPlots)
cb = function(J, Jv, iter, His, parms, doPlots=true)
    global bestLoss, Θbest

    # 1. Validation loss
    Jvc = Jv(parms)
    if Jvc < bestLoss
        bestLoss = Jvc
        Θbest = copy(MFGnet.param2vec(parms))
    end

    # 2. Log metrics to history array
    His[iter,:] = [J.cs[1:5]..., Jv.cs[1:5]...]

    # 3. Print progress
    @printf("iter=%04d obj=%1.3e costL=%1.3e ...\n",
            iter, sum(His[iter,1:5]), His[iter,1])

    # 4. Periodic resampling
    if mod(iter, sampleFreq) == 0
        J.X0 = sample(rho0, nTrain)
        J.rho0x = rho0(J.X0)
    end

    # 5. Checkpointing
    if mod(iter, saveIter) == 0
        save("checkpoint-iter$iter.jld", "Θ", Θbest, "His", His)
    end

    # 6. Visualization
    if doPlots
        # Create and display plots
    end
end

# Wrapper for BFGS
cbBFGS = (iter) -> cb(J, Jv, iter, His, parms, doPlots)
```

### New Callback Structure

**Optimization.jl Callback Signature:**

```julia
function callback(state, loss_val)
    # state: OptimizationState with fields
    #   - iter: iteration number
    #   - u: current parameters (ComponentArray)
    #   - objective: current loss value
    #   - time: elapsed time

    # Return true to stop, false to continue
    return should_stop::Bool
end
```

### Migration Strategy: Callback Wrapper

**Create a callback adapter:**

```julia
"""
    create_mfg_callback(; validation_mfg=nothing, history_size=1000,
                        sample_freq=25, save_freq=25, save_path="checkpoints/",
                        do_plots=false, verbose=true)

Create Optimization.jl compatible callback with MFGnet-specific functionality
"""
function create_mfg_callback(;
    training_mfg,           # Training MFG problem
    validation_mfg=nothing, # Validation MFG problem (optional)
    Θ_structure,            # Parameter structure template
    history_size=1000,      # Pre-allocate history array
    sample_freq=0,          # Resample particles every N iters (0=disable)
    save_freq=0,            # Save checkpoint every N iters (0=disable)
    save_path="checkpoints/",
    do_plots=false,
    verbose=true
)
    # Pre-allocate history array
    # Columns: [train_obj, train_costL, train_costF, train_costG, train_costHJ, train_costHJfinal,
    #           val_obj, val_costL, val_costF, val_costG, val_costHJ, val_costHJfinal]
    His = zeros(history_size, 12)

    # Track best validation loss
    best_val_loss = Inf
    Θ_best = nothing

    # Particle sampling state
    rho0 = training_mfg.rho0
    n_particles = size(training_mfg.X0, 2)

    # Optimization.jl callback function
    function callback(state, loss_val)
        iter = state.iter

        # Convert current parameters back to nested tuple format
        Θ_current = componentarray_to_params(state.u, Θ_structure)

        # === 1. Compute validation loss ===
        val_loss = if !isnothing(validation_mfg)
            validation_mfg(Θ_current)
        else
            NaN
        end

        # === 2. Track best parameters ===
        if !isnothing(validation_mfg) && val_loss < best_val_loss
            best_val_loss = val_loss
            Θ_best = deepcopy(Θ_current)
        end

        # === 3. Log metrics ===
        if iter <= history_size
            His[iter, 1:6] = training_mfg.cs
            if !isnothing(validation_mfg)
                His[iter, 7:12] = validation_mfg.cs
            end
        end

        # === 4. Print progress ===
        if verbose && (iter == 1 || iter % 10 == 0)
            if iter == 1
                @printf("%-6s %-12s %-12s %-12s %-12s %-12s %-12s %-12s\n",
                       "Iter", "Train Obj", "CostL", "CostF", "CostG", "CostHJ", "Val Obj", "Best Val")
                @printf("%s\n", "="^100)
            end

            @printf("%6d %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e\n",
                   iter, loss_val,
                   training_mfg.cs[1], training_mfg.cs[2], training_mfg.cs[3],
                   training_mfg.cs[4],
                   val_loss, best_val_loss)
        end

        # === 5. Periodic resampling ===
        if sample_freq > 0 && mod(iter, sample_freq) == 0
            if verbose
                println("  → Resampling particles...")
            end
            X0_new = sample(rho0, n_particles)
            training_mfg.X0 = X0_new
            training_mfg.rho0x = rho0(X0_new)
            # Update functionals that depend on X0
            if hasfield(typeof(training_mfg.G), :rho0x)
                training_mfg.G.rho0x = rho0(X0_new)
            end
        end

        # === 6. Checkpointing ===
        if save_freq > 0 && mod(iter, save_freq) == 0
            mkpath(save_path)
            checkpoint_file = joinpath(save_path, "checkpoint_iter_$(iter).jld2")

            using JLD2
            @save checkpoint_file Θ=Θ_current Θ_best=Θ_best iter=iter loss=loss_val history=His[1:iter,:]

            if verbose
                println("  → Checkpoint saved: $checkpoint_file")
            end
        end

        # === 7. Visualization ===
        if do_plots && mod(iter, 10) == 0
            # Plot validation solution
            plot_mfg_solution(validation_mfg, Θ_current)
        end

        # Return false to continue, true to stop
        return false
    end

    # Return callback and history array
    return callback, His
end
```

### Complete Migration Example

**Before:**

```julia
# Old callback system
His = zeros(maxIter, 10)
cbBFGS = (iter) -> cb(J, Jv, iter, His, parms, doPlots)

Θ0 = MFGnet.param2vec(parms)
f   = (Θ) -> evalObj(J, Θ, parms, ps)
fdf = (Θ) -> evalObjAndGrad(J, Θ, parms, ps)

Θopt, flag, His, X, H = MFGnet.bfgs(f, fdf, Θ0,
                                     maxIter=maxIter,
                                     out=0, atol=1e-10,
                                     cb=cbBFGS)
```

**After:**

```julia
# New callback system
callback, His = create_mfg_callback(
    training_mfg = J,
    validation_mfg = Jv,
    Θ_structure = get_param_structure(parms),
    sample_freq = sampleFreq,
    save_freq = saveIter,
    save_path = saveStr,
    do_plots = doPlots,
    verbose = true
)

result = train_mfg(
    J, parms,
    optimizer = LBFGS(),
    maxiters = maxIter,
    abstol = 1e-10,
    callback = callback
)

Θopt = result.Θ_opt
```

---

## Step-by-Step Implementation Plan

### Phase 1: Foundation (Week 1-2) - NEW CODE ONLY

**Goal:** Create new optimization infrastructure without touching existing code.

#### Task 1.1: Add Dependencies

**File:** `/home/user/MFGnet.jl/Project.toml`

```toml
[deps]
# ... existing deps ...
ComponentArrays = "b0b7db55-cfe3-40fc-9ded-d10e2dbeff66"
Optimization = "7f7a1694-90dd-40f0-9382-eb1efda571ba"
OptimizationOptimJL = "36348300-93cb-4f02-beb5-3c3902f8871e"
OptimizationOptimisers = "42dfb2eb-d2b4-4451-abcd-913932933ac1"

[compat]
ComponentArrays = "0.13, 0.14, 0.15"
Optimization = "3.19, 3.20, 3.21"
OptimizationOptimJL = "0.1, 0.2"
OptimizationOptimisers = "0.1, 0.2"
```

**Commands:**

```bash
cd /home/user/MFGnet.jl
julia --project=. -e 'using Pkg; Pkg.add(["ComponentArrays", "Optimization", "OptimizationOptimJL", "OptimizationOptimisers"])'
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

#### Task 1.2: Create ComponentArray Utilities

**File:** `/home/user/MFGnet.jl/src/componentarray_utils.jl`

```julia
"""
ComponentArray utilities for parameter management
"""

export params_to_componentarray, componentarray_to_params, get_param_structure

using ComponentArrays

"""
    params_to_componentarray(Θ)

Convert nested tuple parameter structure to ComponentArray

# Example
```julia
Θ = ((K1, b1), (K2, b2))
θ = params_to_componentarray(Θ)
# Now access: θ.layer_1_1 (K1), θ.layer_1_2 (b1), etc.
```
"""
function params_to_componentarray(Θ::Tuple)
    flat_dict = Dict{Symbol, Any}()
    _flatten_tuple!(flat_dict, Θ, :layer)
    return ComponentArray(; flat_dict...)
end

function params_to_componentarray(Θ::AbstractArray)
    return ComponentArray(param=vec(Θ))
end

function _flatten_tuple!(dict::Dict, t::Tuple, prefix::Symbol)
    for (i, item) in enumerate(t)
        key = Symbol(prefix, "_", i)
        if item isa Tuple
            _flatten_tuple!(dict, item, key)
        else
            dict[key] = vec(item)
        end
    end
end

"""
    get_param_structure(Θ)

Extract structure template for reconstruction
"""
function get_param_structure(Θ::Tuple)
    return tuple([get_param_structure(t) for t in Θ]...)
end

function get_param_structure(Θ::AbstractArray)
    return Θ  # Store shape info
end

"""
    componentarray_to_params(θ::ComponentArray, structure)

Convert ComponentArray back to nested tuple structure
"""
function componentarray_to_params(θ::ComponentArray, structure::Tuple)
    return _reconstruct_tuple(θ, structure, :layer, Ref(0))
end

function componentarray_to_params(θ::ComponentArray, structure::AbstractArray)
    return reshape(θ.param, size(structure))
end

function _reconstruct_tuple(θ::ComponentArray, structure::Tuple, prefix::Symbol, idx::Ref{Int})
    result = []
    for (i, item) in enumerate(structure)
        key = Symbol(prefix, "_", i)
        if item isa Tuple
            push!(result, _reconstruct_tuple(θ, item, key, idx))
        else
            push!(result, reshape(getproperty(θ, key), size(item)))
        end
    end
    return tuple(result...)
end

"""
    count_parameters(Θ)

Count total number of scalar parameters
"""
function count_parameters(Θ::Tuple)
    return sum(count_parameters(t) for t in Θ)
end

function count_parameters(Θ::AbstractArray)
    return length(Θ)
end
```

**Test:**

```julia
# Quick test
K1 = randn(3, 2)
b1 = randn(3)
K2 = randn(2, 3)
b2 = randn(2)
Θ = ((K1, b1), (K2, b2))

θ = params_to_componentarray(Θ)
@assert length(θ) == count_parameters(Θ)

structure = get_param_structure(Θ)
Θ_reconstructed = componentarray_to_params(θ, structure)
@assert Θ_reconstructed[1][1] ≈ K1
@assert Θ_reconstructed[1][2] ≈ b1
```

#### Task 1.3: Create Optimization Wrapper

**File:** `/home/user/MFGnet.jl/src/optimization_wrapper.jl`

```julia
"""
Optimization.jl integration for MFGnet.jl
"""

export create_optimization_problem, solve_optimization, train_mfg, select_optimizer

using Optimization
using OptimizationOptimJL
using OptimizationOptimisers
using ComponentArrays
using Printf

"""
    OptimMFGProblem

Wrapper for MFG optimization problem
"""
struct OptimMFGProblem{R<:Real}
    mfg::MeanFieldGame{R}
    opt_prob::OptimizationProblem
    initial_params::ComponentArray
    param_structure
end

"""
    create_optimization_problem(mfg, Θ_init; kwargs...)

Create Optimization.jl problem from MeanFieldGame

# Arguments
- `mfg::MeanFieldGame` - MFG problem
- `Θ_init` - Initial parameters (nested tuple)
- `adtype` - AD backend (default: AutoZygote)
- `kwargs...` - Additional options

# Returns
OptimMFGProblem ready for optimization
"""
function create_optimization_problem(
    mfg::MeanFieldGame{R},
    Θ_init;
    adtype=Optimization.AutoZygote(),
    kwargs...
) where R<:Real

    # Convert to ComponentArray
    θ0 = params_to_componentarray(Θ_init)
    structure = get_param_structure(Θ_init)

    # Objective function (Optimization.jl convention: loss(θ, p))
    function objective(θ, p)
        # Convert back to nested tuple
        Θ = componentarray_to_params(θ, structure)
        # Evaluate MFG
        return mfg(Θ)
    end

    # Create OptimizationFunction with automatic gradients
    opt_func = OptimizationFunction(objective, adtype)

    # Create OptimizationProblem
    opt_prob = OptimizationProblem(opt_func, θ0, nothing)

    return OptimMFGProblem(mfg, opt_prob, θ0, structure)
end

"""
    solve_optimization(prob, alg; kwargs...)

Solve optimization problem

# Common Algorithms
- `LBFGS()` - Limited-memory BFGS (recommended for medium/large)
- `BFGS()` - Full BFGS (small problems)
- `Adam(lr)` - Adam optimizer (large problems)
- `Newton()` - Newton with Hessian

# Returns
OptimizationSolution with fields:
- `u` - Optimal parameters (ComponentArray)
- `objective` - Final loss
- `retcode` - Status
"""
function solve_optimization(
    prob::OptimMFGProblem,
    alg;
    maxiters=1000,
    abstol=1e-8,
    reltol=1e-6,
    callback=nothing,
    verbose=true,
    kwargs...
)

    # Default callback
    if isnothing(callback) && verbose
        callback = function(state, loss_val)
            if state.iter % 10 == 0
                @printf("Iter %4d: loss = %.6e\n", state.iter, loss_val)
            end
            return false
        end
    end

    # Solve
    sol = solve(prob.opt_prob, alg;
               maxiters=maxiters,
               abstol=abstol,
               reltol=reltol,
               callback=callback,
               kwargs...)

    if verbose
        if sol.retcode == :Success
            println("✓ Optimization converged")
        else
            @warn "Optimization status: $(sol.retcode)"
        end
        @printf("Final loss: %.6e\n", sol.objective)
    end

    return sol
end

"""
    train_mfg(mfg, Θ_init; kwargs...)

High-level training interface

# Example
```julia
result = train_mfg(
    mfg, Θ_init,
    optimizer = LBFGS(),
    maxiters = 1000,
    abstol = 1e-8
)
```
"""
function train_mfg(
    mfg::MeanFieldGame,
    Θ_init;
    optimizer=:auto,
    maxiters=1000,
    abstol=1e-8,
    reltol=1e-6,
    callback=nothing,
    verbose=true,
    checkpoint_every=0,
    save_path="checkpoints/",
    kwargs...
)

    # Auto-select optimizer
    if optimizer == :auto
        n_params = count_parameters(Θ_init)
        optimizer, opts = select_optimizer(n_params)
        if verbose
            println("Auto-selected: $optimizer ($(n_params) parameters)")
        end
    end

    # Create problem
    prob = create_optimization_problem(mfg, Θ_init)

    # Training history
    loss_history = Float64[]
    iter_history = Int[]

    # Wrap callback
    function training_callback(state, loss_val)
        push!(loss_history, loss_val)
        push!(iter_history, state.iter)

        # User callback
        if !isnothing(callback)
            stop = callback(state, loss_val)
            if stop
                return true
            end
        end

        # Checkpointing
        if checkpoint_every > 0 && mod(state.iter, checkpoint_every) == 0
            mkpath(save_path)
            Θ_current = componentarray_to_params(state.u, prob.param_structure)

            using JLD2
            @save joinpath(save_path, "checkpoint_iter_$(state.iter).jld2") Θ=Θ_current iter=state.iter loss=loss_val

            if verbose
                println("  → Checkpoint saved")
            end
        end

        return false
    end

    # Solve
    t_start = time()
    sol = solve_optimization(
        prob, optimizer;
        maxiters=maxiters,
        abstol=abstol,
        reltol=reltol,
        callback=training_callback,
        verbose=verbose,
        kwargs...
    )
    t_elapsed = time() - t_start

    # Convert solution back
    Θ_opt = componentarray_to_params(sol.u, prob.param_structure)

    if verbose
        @printf("\nTraining complete in %.2f seconds\n", t_elapsed)
        @printf("Iterations: %d\n", length(loss_history))
    end

    return (
        Θ_opt = Θ_opt,
        loss_history = loss_history,
        iter_history = iter_history,
        sol = sol,
        time_elapsed = t_elapsed
    )
end

"""
    select_optimizer(n_params; prefer_first_order=false)

Automatic optimizer selection based on problem size
"""
function select_optimizer(n_params::Int; prefer_first_order::Bool=false)
    if n_params < 100 && !prefer_first_order
        return (BFGS(), (maxiters=500, abstol=1e-8))
    elseif n_params < 10_000 && !prefer_first_order
        return (LBFGS(), (maxiters=1000, abstol=1e-8))
    else
        return (Adam(0.001), (maxiters=5000, abstol=1e-6))
    end
end
```

#### Task 1.4: Update Module Exports

**File:** `/home/user/MFGnet.jl/src/MFGnet.jl`

```julia
module MFGnet

# ... existing exports ...

# New optimization exports
include("componentarray_utils.jl")
include("optimization_wrapper.jl")

export create_optimization_problem, solve_optimization, train_mfg
export params_to_componentarray, componentarray_to_params

end # module
```

### Phase 2: Testing (Week 2) - VALIDATE NEW CODE

#### Task 2.1: Unit Tests for ComponentArrays

**File:** `/home/user/MFGnet.jl/test/test_componentarrays.jl`

```julia
using Test
using MFGnet
using ComponentArrays

@testset "ComponentArray Utilities" begin

    @testset "Flat Array Conversion" begin
        K = randn(5, 3)
        θ = params_to_componentarray(K)

        @test θ isa ComponentArray
        @test length(θ) == length(K)

        structure = get_param_structure(K)
        K_reconstructed = componentarray_to_params(θ, structure)

        @test K_reconstructed ≈ K
    end

    @testset "Nested Tuple Conversion" begin
        K1 = randn(5, 3)
        b1 = randn(5)
        K2 = randn(4, 5)
        b2 = randn(4)
        Θ = ((K1, b1), (K2, b2))

        # Convert to ComponentArray
        θ = params_to_componentarray(Θ)

        @test θ isa ComponentArray
        @test length(θ) == count_parameters(Θ)

        # Convert back
        structure = get_param_structure(Θ)
        Θ_reconstructed = componentarray_to_params(θ, structure)

        @test Θ_reconstructed isa Tuple
        @test length(Θ_reconstructed) == 2
        @test Θ_reconstructed[1][1] ≈ K1
        @test Θ_reconstructed[1][2] ≈ b1
        @test Θ_reconstructed[2][1] ≈ K2
        @test Θ_reconstructed[2][2] ≈ b2
    end

    @testset "Deep Nesting" begin
        A = randn(2, 2)
        B = randn(2)
        C = randn(3, 2)
        D = randn(3)
        Θ = ((A, B), ((C, D),))

        θ = params_to_componentarray(Θ)
        @test length(θ) == count_parameters(Θ)

        structure = get_param_structure(Θ)
        Θ_reconstructed = componentarray_to_params(θ, structure)

        @test Θ_reconstructed[1][1] ≈ A
        @test Θ_reconstructed[1][2] ≈ B
        @test Θ_reconstructed[2][1][1] ≈ C
        @test Θ_reconstructed[2][1][2] ≈ D
    end

    @testset "Count Parameters" begin
        Θ = ((randn(5,3), randn(5)), (randn(4,5), randn(4)))
        @test count_parameters(Θ) == 5*3 + 5 + 4*5 + 4
    end
end
```

#### Task 2.2: Integration Tests

**File:** `/home/user/MFGnet.jl/test/test_optimization_integration.jl`

```julia
using Test
using MFGnet
using Optimization
using OptimizationOptimJL

@testset "Optimization Integration" begin

    @testset "Small MFG Problem" begin
        # Create tiny problem for fast testing
        d = 1
        nex = 20
        X0 = randn(d, nex)
        rho0(x) = ones(size(x, 2))
        w = ones(nex) / nex

        F = F0()
        G = Gkl(rho0, rho0, rho0(X0), rho0(X0), 1.0)
        mfg = MeanFieldGame(F, G, X0, rho0, w)

        # Simple 1-layer potential
        K = randn(Float64, 5, d+1)
        b = randn(Float64, 5)
        Θ_init = (K, b)

        # Test: create_optimization_problem
        prob = create_optimization_problem(mfg, Θ_init)

        @test prob isa OptimMFGProblem
        @test prob.opt_prob isa OptimizationProblem
        @test length(prob.initial_params) == count_parameters(Θ_init)
    end

    @testset "LBFGS Optimization" begin
        d = 1
        nex = 20
        X0 = randn(d, nex)
        rho0(x) = ones(size(x, 2))
        w = ones(nex) / nex

        F = F0()
        G = Gkl(rho0, rho0, rho0(X0), rho0(X0), 1.0)
        mfg = MeanFieldGame(F, G, X0, rho0, w)

        K = randn(Float64, 5, d+1)
        b = randn(Float64, 5)
        Θ_init = (K, b)

        # Run short optimization
        result = train_mfg(
            mfg, Θ_init,
            optimizer = LBFGS(),
            maxiters = 10,
            verbose = false
        )

        @test result.Θ_opt isa Tuple
        @test length(result.loss_history) > 0
        @test result.loss_history[end] <= result.loss_history[1]  # Should decrease
    end

    @testset "Callback Functionality" begin
        d = 1
        nex = 20
        X0 = randn(d, nex)
        rho0(x) = ones(size(x, 2))
        w = ones(nex) / nex

        F = F0()
        G = Gkl(rho0, rho0, rho0(X0), rho0(X0), 1.0)
        mfg = MeanFieldGame(F, G, X0, rho0, w)

        K = randn(Float64, 5, d+1)
        b = randn(Float64, 5)
        Θ_init = (K, b)

        # Track callback invocations
        callback_count = Ref(0)
        function test_callback(state, loss_val)
            callback_count[] += 1
            return false
        end

        result = train_mfg(
            mfg, Θ_init,
            optimizer = LBFGS(),
            maxiters = 10,
            callback = test_callback,
            verbose = false
        )

        @test callback_count[] > 0
    end
end
```

#### Task 2.3: Comparison Tests (Legacy vs New)

**File:** `/home/user/MFGnet.jl/test/test_legacy_comparison.jl`

```julia
using Test
using MFGnet
using Optimization
using OptimizationOptimJL
using Flux, Zygote

@testset "Legacy vs New Comparison" begin

    @testset "Same Initial Point, Similar Results" begin
        # Setup problem
        d = 2
        nex = 50
        X0 = randn(d, nex)
        rho0(x) = ones(size(x, 2))
        w = ones(nex) / nex

        F = F0()
        G = Gkl(rho0, rho0, rho0(X0), rho0(X0), 1.0)
        mfg = MeanFieldGame(F, G, X0, rho0, w)

        # Simple parameters
        K = randn(Float64, 10, d+1)
        b = randn(Float64, 10)
        Θ_init = (K, b)

        # === Legacy BFGS ===
        parms = MFGnet.myMap(x->x, Θ_init)
        ps = Flux.params(parms)
        Θ0 = MFGnet.param2vec(parms)

        f = (Θ) -> evalObj(mfg, Θ, parms, ps)
        fdf = (Θ) -> evalObjAndGrad(mfg, Θ, parms, ps)

        Θopt_legacy, flag_legacy, His_legacy, X, H = MFGnet.bfgs_legacy(
            f, fdf, Θ0,
            maxIter=50,
            atol=1e-6,
            out=0
        )

        parms_legacy = vec2param!(Θopt_legacy, parms)
        loss_legacy = mfg(parms_legacy)

        # === New Optimization.jl ===
        result_new = train_mfg(
            mfg, Θ_init,
            optimizer = BFGS(),  # Use BFGS (not LBFGS) for fair comparison
            maxiters = 50,
            abstol = 1e-6,
            verbose = false
        )

        loss_new = mfg(result_new.Θ_opt)

        # Should achieve similar loss (not identical due to line search differences)
        @test isapprox(loss_legacy, loss_new, rtol=0.1)

        println("Legacy final loss: $(loss_legacy)")
        println("New final loss:    $(loss_new)")
    end
end
```

### Phase 3: Migration Examples (Week 3) - SHOW USERS HOW TO MIGRATE

#### Task 3.1: Create Migration Guide

**File:** `/home/user/MFGnet.jl/docs/MIGRATION_GUIDE.md`

```markdown
# Migration Guide: Custom BFGS → Optimization.jl

## Quick Reference

| Old API | New API |
|---------|---------|
| `MFGnet.bfgs(f, fdf, x0, ...)` | `train_mfg(mfg, Θ_init, optimizer=LBFGS())` |
| Manual `param2vec`, `vec2param!` | Automatic via ComponentArrays |
| `evalObj`, `evalObjAndGrad` wrappers | Direct MFG evaluation |
| Custom callbacks | Optimization.jl callbacks |

## Step-by-Step Migration

### Before (Custom BFGS)

```julia
# 1. Setup parameters
parms = (w0, (ΘN), A0, b0, z0)
ps = Flux.params(parms)
Θ0 = MFGnet.param2vec(parms)

# 2. Define wrappers
f   = (Θ) -> evalObj(J, Θ, parms, ps)
fdf = (Θ) -> evalObjAndGrad(J, Θ, parms, ps)

# 3. Optimize
Θopt, flag, His, X, H = MFGnet.bfgs(f, fdf, Θ0, maxIter=200, atol=1e-10)

# 4. Unflatten
parms_opt = vec2param!(Θopt, parms)
```

### After (Optimization.jl)

```julia
# 1. Setup parameters (same)
Θ_init = (w0, (ΘN), A0, b0, z0)

# 2. Train (one line!)
result = train_mfg(J, Θ_init, optimizer=LBFGS(), maxiters=200, abstol=1e-10)

# 3. Extract result
Θ_opt = result.Θ_opt
```

## Migrating Callbacks

### Before

```julia
cb = function(J, Jv, iter, His, parms, doPlots)
    # Custom logic
end

cbBFGS = (iter) -> cb(J, Jv, iter, His, parms, doPlots)
Θopt, flag, His, X, H = MFGnet.bfgs(..., cb=cbBFGS)
```

### After

```julia
callback = function(state, loss_val)
    iter = state.iter
    Θ_current = componentarray_to_params(state.u, structure)

    # Custom logic (convert to new format)

    return false  # Continue optimization
end

result = train_mfg(J, Θ_init, callback=callback)
```

See detailed callback migration in next section.
```

#### Task 3.2: Create Modern Example

**File:** `/home/user/MFGnet.jl/examples/modern_optimization_demo.jl`

```julia
"""
Modern Optimization API Demo for MFGnet.jl

Demonstrates:
1. Using Optimization.jl with multiple algorithms
2. Migrated callback system
3. Validation and checkpointing
4. Easy algorithm comparison
"""

using MFGnet
using Optimization
using OptimizationOptimJL
using OptimizationOptimisers
using LinearAlgebra
using Printf
using Plots

# === Problem Setup ===
println("Setting up 2D MFG problem...")

d = 2
nTrain = 100
nVal = 200

# Initial and target densities
rho0 = Gaussian(d, [0.3, 0.3], [0.0, 2.0])
rho1 = Gaussian(d, [0.3, 0.3], [0.0, -2.0])

# Training set
X0_train = sample(rho0, nTrain)
w_train = ones(nTrain) / nTrain

# Validation set
X0_val = sample(rho0, nVal)
w_val = ones(nVal) / nVal

# Costs
F = F0()
G_train = Gkl(rho0, rho1, rho0(X0_train), rho1(X0_train), 1.0)
G_val = Gkl(rho0, rho1, rho0(X0_val), rho1(X0_val), 1.0)

# MFG problems
α = [1.0, 2.0, 4.0, 2.0, 2.0]
mfg_train = MeanFieldGame(F, G_train, X0_train, rho0, w_train; α=α)
mfg_val = MeanFieldGame(F, G_val, X0_val, rho0, w_val; α=α)

# Neural network potential
Φ = getPotentialResNet(4, 1.0, 4, Float64)
w0, ΘN, A0, b0, z0 = initializeWeights(d, 32, 4, identity)
Θ_init = (w0, (ΘN), A0, b0, z0)

println("Parameters: $(count_parameters(Θ_init))")

# === Callback Setup ===
callback, His = create_mfg_callback(
    training_mfg = mfg_train,
    validation_mfg = mfg_val,
    Θ_structure = get_param_structure(Θ_init),
    sample_freq = 25,
    save_freq = 50,
    save_path = "results/modern_demo/",
    verbose = true
)

# === Training with Different Optimizers ===

println("\n" * "="^70)
println("Training with LBFGS")
println("="^70)

result_lbfgs = train_mfg(
    mfg_train, Θ_init,
    optimizer = LBFGS(),
    maxiters = 200,
    abstol = 1e-10,
    callback = callback,
    verbose = true
)

println("\n" * "="^70)
println("Training with Adam")
println("="^70)

result_adam = train_mfg(
    mfg_train, Θ_init,
    optimizer = Adam(0.01),
    maxiters = 500,
    abstol = 1e-10,
    verbose = true
)

# === Comparison ===

println("\n" * "="^70)
println("Comparison")
println("="^70)
@printf("LBFGS: %.6e in %.2f sec (%d iters)\n",
       result_lbfgs.sol.objective,
       result_lbfgs.time_elapsed,
       length(result_lbfgs.loss_history))

@printf("Adam:  %.6e in %.2f sec (%d iters)\n",
       result_adam.sol.objective,
       result_adam.time_elapsed,
       length(result_adam.loss_history))

# === Visualization ===

p = plot(
    xlabel="Iteration",
    ylabel="Loss",
    title="Optimization Comparison",
    yscale=:log10,
    legend=:topright
)

plot!(p, result_lbfgs.loss_history, label="LBFGS", lw=2)
plot!(p, result_adam.loss_history, label="Adam", lw=2)

savefig(p, "results/modern_demo/comparison.png")
display(p)

println("\n✓ Demo complete")
```

### Phase 4: Deprecation (Week 3-4) - MAINTAIN BACKWARD COMPATIBILITY

#### Task 4.1: Add Deprecation Warnings

**File:** `/home/user/MFGnet.jl/src/bfgs.jl` (modify existing file)

```julia
# Add at the top of the file

"""
    bfgs_legacy(...)

Legacy BFGS implementation (preserved for backward compatibility)

This is the original custom BFGS implementation. Use `train_mfg` with
`optimizer=LBFGS()` for new code.
"""
const bfgs_legacy = bfgs  # Save original implementation

"""
    bfgs(...)

**DEPRECATED:** Use `train_mfg` with Optimization.jl instead.

# Migration
Instead of:
```julia
Θ0 = param2vec(parms)
f = (Θ) -> evalObj(J, Θ, parms, ps)
fdf = (Θ) -> evalObjAndGrad(J, Θ, parms, ps)
Θopt, flag, His, X, H = bfgs(f, fdf, Θ0, maxIter=200)
```

Use:
```julia
result = train_mfg(J, parms, optimizer=LBFGS(), maxiters=200)
Θopt = result.Θ_opt
```

See MIGRATION_GUIDE.md for details.
"""
function bfgs(args...; kwargs...)
    @warn """
    `bfgs` is deprecated and will be removed in v1.0.

    Migration:
    - Use `train_mfg(mfg, Θ_init, optimizer=LBFGS())` instead
    - See `docs/MIGRATION_GUIDE.md` for examples
    - For temporary legacy behavior, use `bfgs_legacy`
    """ maxlog=1

    return bfgs_legacy(args...; kwargs...)
end
```

**File:** `/home/user/MFGnet.jl/src/utils.jl` (add deprecation)

```julia
# Add deprecation warnings

"""
    evalObj(J, Θ, parms, ps)

**DEPRECATED:** Used with legacy `bfgs` function.
For new code, use `train_mfg` which handles parameters automatically.
"""
function evalObj(J, Θ::Vector, parms, ps)
    @warn "`evalObj` is deprecated. Use `train_mfg` instead." maxlog=1
    parms = vec2param!(Θ, parms)
    return J(parms)
end

"""
    evalObjAndGrad(J, Θ, parms, ps)

**DEPRECATED:** Used with legacy `bfgs` function.
For new code, use `train_mfg` which computes gradients automatically.
"""
function evalObjAndGrad(J, Θ::Vector, parms, ps)
    @warn "`evalObjAndGrad` is deprecated. Use `train_mfg` instead." maxlog=1
    # ... existing implementation ...
end
```

### Phase 5: Documentation (Week 4) - COMPLETE DOCUMENTATION

#### Task 5.1: Update README

**File:** `/home/user/MFGnet.jl/README.md` (add section)

```markdown
## Quick Start (Modern API)

```julia
using MFGnet
using Optimization
using OptimizationOptimJL

# Define problem
rho0 = Gaussian(2, [0.3, 0.3], [0.0, 2.0])
rho1 = Gaussian(2, [0.3, 0.3], [0.0, -2.0])

X0 = sample(rho0, 100)
w = ones(100) / 100

F = F0()
G = Gkl(rho0, rho1, rho0(X0), rho1(X0), 1.0)
mfg = MeanFieldGame(F, G, X0, rho0, w)

# Initialize neural network
Φ = getPotentialResNet(4, 1.0, 4, Float64)
Θ_init = initializeWeights(2, 32, 4, identity)

# Train with LBFGS
result = train_mfg(
    mfg, Θ_init,
    optimizer = LBFGS(),
    maxiters = 200
)

# Or try Adam for large problems
result = train_mfg(
    mfg, Θ_init,
    optimizer = Adam(0.01),
    maxiters = 1000
)
```

## Migration from v0.2

Users of v0.2.x can continue using the legacy `bfgs` function, but we recommend migrating to the new `train_mfg` interface.

See [`docs/MIGRATION_GUIDE.md`](docs/MIGRATION_GUIDE.md) for detailed migration instructions.
```

---

## Testing Strategy with Code

### Testing Pyramid

```
                  ┌─────────────────┐
                  │  End-to-End     │  ← Full training runs
                  │  (slow)         │
                  └─────────────────┘

              ┌───────────────────────┐
              │    Integration        │  ← Component interaction
              │    (medium speed)     │
              └───────────────────────┘

        ┌─────────────────────────────────┐
        │         Unit Tests               │  ← Individual functions
        │         (fast)                   │
        └─────────────────────────────────┘
```

### Test Organization

```
test/
├── runtests.jl                       # Main test runner
│
├── unit/
│   ├── test_componentarrays.jl       # ComponentArray conversion
│   ├── test_param_counting.jl        # Parameter utilities
│   └── test_optimizer_selection.jl   # Auto-selection logic
│
├── integration/
│   ├── test_optimization_wrapper.jl  # Optimization.jl integration
│   ├── test_small_problems.jl        # Small MFG problems
│   └── test_callback_system.jl       # Callback functionality
│
├── comparison/
│   ├── test_legacy_vs_new.jl         # Compare with legacy BFGS
│   ├── test_numerical_accuracy.jl    # Ensure same results
│   └── test_performance.jl           # Benchmark speed
│
└── e2e/
    ├── test_full_training.jl         # Complete training workflows
    └── test_examples.jl              # Run example scripts
```

### Continuous Integration

**.github/workflows/test.yml**

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        julia-version: ['1.10', '1.11']

    steps:
      - uses: actions/checkout@v3

      - uses: julia-actions/setup-julia@v1
        with:
          version: ${{ matrix.julia-version }}

      - uses: julia-actions/cache@v1

      - name: Install dependencies
        run: |
          julia --project=. -e 'using Pkg; Pkg.instantiate()'

      - name: Run tests
        run: |
          julia --project=. -e 'using Pkg; Pkg.test(coverage=true)'

      - name: Submit coverage
        uses: codecov/codecov-action@v3
```

---

## Risk Assessment

### Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Breaking changes to published results** | Low | High | Keep legacy BFGS, extensive comparison tests |
| **Performance regression** | Low | Medium | Benchmark before/after, keep legacy option |
| **User adoption resistance** | Medium | Low | Clear migration guide, gradual deprecation |
| **Dependency issues** | Low | Medium | Pin versions, test on multiple Julia versions |
| **Callback complexity** | Medium | Medium | Provide helper functions, examples |

### Mitigation Strategies

#### 1. Reproducibility Protection

```julia
# Add to tests
@testset "Reproducibility" begin
    # Ensure published results can be reproduced with legacy BFGS
    include("../examples/ROLNWF2019/runObstacleExperiment.jl")
    # Check that results match reference values
end
```

#### 2. Performance Monitoring

```julia
using BenchmarkTools

@testset "Performance Benchmarks" begin
    # Compare optimization overhead
    legacy_time = @belapsed bfgs_legacy(f, fdf, x0, maxIter=100)
    new_time = @belapsed train_mfg(mfg, Θ, optimizer=LBFGS(), maxiters=100)

    @test new_time < 1.5 * legacy_time  # Allow 50% overhead max
end
```

---

## Performance Benchmarks

### Expected Improvements

| Metric | Custom BFGS | Optimization.jl | Improvement |
|--------|-------------|-----------------|-------------|
| **Line Search** | Armijo only | Wolfe, HagerZhang | Better convergence |
| **Memory (LBFGS)** | O(n²) | O(m·n), m≪n | 10-100x reduction |
| **Optimizer Choice** | 1 (BFGS) | 10+ algorithms | More flexibility |
| **Gradient Handling** | Manual flatten/unflatten | Automatic | 50% less code |
| **Callback Complexity** | Custom per-project | Standardized | Easier maintenance |

### Benchmark Code

```julia
using BenchmarkTools
using MFGnet
using Optimization
using OptimizationOptimJL

function benchmark_optimization()
    # Setup problem
    d = 2
    nex = 100
    X0 = randn(d, nex)
    rho0(x) = ones(size(x, 2))
    w = ones(nex) / nex

    F = F0()
    G = Gkl(rho0, rho0, rho0(X0), rho0(X0), 1.0)
    mfg = MeanFieldGame(F, G, X0, rho0, w)

    K = randn(Float64, 32, d+1)
    b = randn(Float64, 32)
    Θ_init = (K, b)

    # Benchmark legacy BFGS
    parms = MFGnet.myMap(x->x, Θ_init)
    ps = Flux.params(parms)
    Θ0 = MFGnet.param2vec(parms)
    f = (Θ) -> evalObj(mfg, Θ, parms, ps)
    fdf = (Θ) -> evalObjAndGrad(mfg, Θ, parms, ps)

    t_legacy = @belapsed begin
        MFGnet.bfgs_legacy($f, $fdf, $Θ0, maxIter=50, out=0)
    end

    # Benchmark new Optimization.jl
    t_new = @belapsed begin
        train_mfg($mfg, $Θ_init, optimizer=BFGS(), maxiters=50, verbose=false)
    end

    @printf("Legacy BFGS:        %.3f seconds\n", t_legacy)
    @printf("Optimization.jl:    %.3f seconds\n", t_new)
    @printf("Speedup:            %.2fx\n", t_legacy / t_new)
end

benchmark_optimization()
```

---

## Appendix: Complete Examples

### A.1: Full Migrated Example (Obstacle Problem)

**File:** `/home/user/MFGnet.jl/examples/modern_obstacle_experiment.jl`

```julia
"""
Obstacle Problem using Modern Optimization API

Migrated from: examples/ROLNWF2019/runObstacleExperiment.jl
"""

using MFGnet
using Optimization
using OptimizationOptimJL
using LinearAlgebra
using Printf
using JLD2

# === Configuration ===
d = 2
nTrain = 256  # 16^2
nVal = 4096   # 64^2
m = 32        # Network width
nTh = 4       # ResNet depth
nt = 4        # Time steps
maxIter = 200

# Densities
mu0 = [0.0, 3.0]
sig0 = [0.3, 0.3]
rho0 = Gaussian(d, sig0, mu0)

mu1 = [0.0, -3.0]
sig1 = [0.3, 0.3]
rho1 = Gaussian(d, sig1, mu1)

# Obstacle
sigQ = [1.0, 0.5]
Qheight = 50.0
Q1 = Gaussian(2, sigQ, zeros(2), Qheight)
Q(x) = Q1(x[1:2,:])

# Training data
X0_train = sample(rho0, nTrain)
w_train = ones(nTrain) / nTrain

# Validation data
X0_val = sample(rho0, nVal)
w_val = ones(nVal) / nVal

# Costs
α = [1.0, 2.0, 4.0, 2.0, 2.0]
muFp = 1.0
muFe = 1e-2

F1_train = Fp(Q, rho0(X0_train), rho1(X0_train), muFp)
F2_train = Fe(rho0, rho0(X0_train), muFe)
F_train = Fcomb([F1_train; F2_train])

F1_val = Fp(Q, rho0(X0_val), rho1(X0_val), muFp)
F2_val = Fe(rho0, rho0(X0_val), muFe)
F_val = Fcomb([F1_val; F2_val])

G_train = Gkl(rho0, rho1, rho0(X0_train), rho1(X0_train), 1.0)
G_val = Gkl(rho0, rho1, rho0(X0_val), rho1(X0_val), 1.0)

# MFG problems
tspan = [0.0, 1.0]
mfg_train = MeanFieldGame(F_train, G_train, X0_train, rho0, w_train; α=α, tspan=tspan)
mfg_val = MeanFieldGame(F_val, G_val, X0_val, rho0, w_val; α=α, tspan=tspan)

# Neural network
Φ = getPotentialResNet(nTh, 1.0, nTh, Float64)
w0, ΘN, A0, b0, z0 = initializeWeights(d, m, nTh, identity)
Θ_init = (w0, (ΘN), A0, b0, z0)

println("\n" * "="^70)
println("Obstacle Problem - Modern API")
println("="^70)
@printf("Parameters:    %d\n", count_parameters(Θ_init))
@printf("Training:      %d particles\n", nTrain)
@printf("Validation:    %d particles\n", nVal)
println("="^70 * "\n")

# === Callback with validation and checkpointing ===
callback, His = create_mfg_callback(
    training_mfg = mfg_train,
    validation_mfg = mfg_val,
    Θ_structure = get_param_structure(Θ_init),
    sample_freq = 25,
    save_freq = 25,
    save_path = "results/obstacle_modern/",
    verbose = true
)

# === Train ===
result = train_mfg(
    mfg_train, Θ_init,
    optimizer = LBFGS(),
    maxiters = maxIter,
    abstol = 1e-10,
    callback = callback,
    verbose = true
)

# === Save Results ===
@save "results/obstacle_modern/final_result.jld2" Θ_opt=result.Θ_opt history=His

println("\n✓ Training complete")
@printf("Final training loss:   %.6e\n", result.loss_history[end])
@printf("Time elapsed:          %.2f seconds\n", result.time_elapsed)
```

### A.2: Algorithm Comparison Script

**File:** `/home/user/MFGnet.jl/examples/compare_optimizers.jl`

```julia
"""
Compare different optimization algorithms on same MFG problem
"""

using MFGnet
using Optimization
using OptimizationOptimJL
using OptimizationOptimisers
using DataFrames
using CSV
using Plots

function compare_optimizers(mfg, Θ_init; maxiters=200)
    algorithms = [
        ("LBFGS", LBFGS()),
        ("BFGS", BFGS()),
        ("ConjugateGradient", ConjugateGradient()),
        ("Adam-0.01", Adam(0.01)),
        ("Adam-0.001", Adam(0.001)),
    ]

    results = DataFrame(
        Algorithm = String[],
        FinalLoss = Float64[],
        Iterations = Int[],
        Time = Float64[],
        Converged = Bool[]
    )

    for (name, alg) in algorithms
        println("\n" * "="^70)
        println("Testing: $name")
        println("="^70)

        try
            result = train_mfg(
                mfg, Θ_init,
                optimizer = alg,
                maxiters = maxiters,
                verbose = false
            )

            push!(results, (
                Algorithm = name,
                FinalLoss = result.loss_history[end],
                Iterations = length(result.loss_history),
                Time = result.time_elapsed,
                Converged = (result.sol.retcode == :Success)
            ))

            @printf("  Final loss: %.6e\n", result.loss_history[end])
            @printf("  Time:       %.2f sec\n", result.time_elapsed)
            @printf("  Converged:  %s\n", result.sol.retcode)

        catch e
            @warn "Algorithm $name failed: $e"
        end
    end

    return results
end

# Setup problem
d = 2
nex = 100
X0 = randn(d, nex)
rho0(x) = ones(size(x, 2))
w = ones(nex) / nex

F = F0()
G = Gkl(rho0, rho0, rho0(X0), rho0(X0), 1.0)
mfg = MeanFieldGame(F, G, X0, rho0, w)

K = randn(Float64, 32, d+1)
b = randn(Float64, 32)
Θ_init = (K, b)

# Run comparison
results = compare_optimizers(mfg, Θ_init, maxiters=200)

# Display results
println("\n" * "="^70)
println("Comparison Results")
println("="^70)
show(results, allrows=true, allcols=true)

# Save to CSV
CSV.write("results/optimizer_comparison.csv", results)

# Plot
p = plot(
    results.Algorithm,
    results.Time,
    xlabel = "Algorithm",
    ylabel = "Time (seconds)",
    title = "Optimization Time Comparison",
    bar = true,
    legend = false,
    xrotation = 45
)
savefig(p, "results/optimizer_comparison.png")
```

---

## Conclusion

This detailed migration guide provides:

1. **Complete code examples** ready to copy-paste
2. **Step-by-step implementation plan** with specific files and tasks
3. **Comprehensive testing strategy** from unit to end-to-end
4. **Callback migration** showing how to adapt existing code
5. **Performance benchmarks** to validate improvements
6. **Risk mitigation** strategies for safe migration
7. **Backward compatibility** to protect existing users

### Next Steps

1. **Week 1-2**: Implement foundation (ComponentArrays, optimization wrapper)
2. **Week 2-3**: Write and run tests
3. **Week 3-4**: Migrate examples and add deprecation warnings
4. **Week 4**: Complete documentation
5. **Release**: Tag v0.3.0 with new API

### Key Benefits

- **10x less boilerplate** code for users
- **10+ optimizer** choices instead of 1
- **Automatic gradient** handling
- **Unified interface** following Julia ecosystem standards
- **Easy experimentation** with different algorithms
- **Better maintenance** through community-supported packages

---

**End of Detailed Migration Guide**
