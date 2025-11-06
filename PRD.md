# Product Requirements Document: MFGnet.jl Modernization (Phase 3)

**Version:** 1.0
**Date:** 2025-01-05
**Status:** Planning
**Owner:** MFGnet.jl Development Team

---

## Executive Summary

This PRD outlines the modernization plan for MFGnet.jl to leverage modern Julia ML frameworks and ecosystem packages. The goal is to reduce codebase complexity, improve maintainability, enable GPU computing, and provide better numerical accuracy while maintaining backward compatibility where possible.

**Expected Outcomes:**
- **40-50% reduction** in codebase size (from ~1900 LOC to ~800-1000 LOC)
- **Native GPU support** without additional code
- **10-100x speedup** for GPU-accelerated problems
- **Improved numerical accuracy** through adaptive time-stepping
- **Better developer experience** through standard interfaces
- **Enhanced maintainability** via modern package ecosystem

---

## 1. Migration to Lux.jl for Neural Networks

### 1.1 Objective

Replace custom neural network implementation (SingleLayer, NN, ResNN) with Lux.jl, a modern functional ML framework designed for scientific computing.

### 1.2 Current State Analysis

**Custom Implementation:**
- ~500 lines of manual gradient code across:
  - `src/singleLayer.jl` (242 lines)
  - `src/NN.jl` (144 lines)
  - `src/ResNN.jl` (222 lines)
- Manual Jacobian, Hessian computations (`getJSTmv`, `getJSJSTmv`, etc.)
- Type instability issues with temporary storage
- No GPU support

**Issues:**
1. High maintenance burden for gradient computations
2. Duplication between NN and ResNN implementations
3. Limited to CPU execution
4. Manual memory management

### 1.3 Proposed Solution

Migrate to **Lux.jl v0.5+** for the following reasons:

**Why Lux.jl over Flux.jl:**
- ✅ Explicit parameter handling (aligns with current architecture)
- ✅ Functional design (no global state)
- ✅ Better AD integration for second-order derivatives
- ✅ Designed for scientific computing use cases
- ✅ Explicit separation of model structure and parameters

**Implementation Plan:**

#### Step 1: Create Custom Activation Layer
```julia
using Lux

struct SoftPlusActivation{R} <: Lux.AbstractExplicitLayer
    # σ(x) = |x| + log(1 + exp(-2|x|))
end

function (l::SoftPlusActivation)(x, ps, st)
    y = abs.(x) .+ log.(one(eltype(x)) .+ exp.(-2 .* abs.(x)))
    return y, st
end
```

#### Step 2: Replace Network Definitions
```julia
# Current: Custom NN struct
N = NN([SingleLayer(); SingleLayer()])

# Proposed: Lux Chain
using ComponentArrays

function create_mfg_network(d_in, widths::Vector{Int}, d_out; R=Float64)
    layers = [Dense(d_in => widths[1], softplus)]
    for i in 1:length(widths)-1
        push!(layers, Dense(widths[i] => widths[i+1], softplus))
    end
    push!(layers, Dense(widths[end] => d_out))

    return Chain(layers...)
end

# Initialize
rng = Random.default_rng()
model = create_mfg_network(10, [20, 20], 5)
ps, st = Lux.setup(rng, model)
```

#### Step 3: Replace ResNN with Lux Residual Blocks
```julia
struct ResidualBlock{L} <: Lux.AbstractExplicitContainerLayer{(:layer,)}
    layer::L
    timesteps::Vector{Float64}
end

function (rb::ResidualBlock)(x, ps, st)
    for (i, (t_start, t_end)) in enumerate(zip(rb.timesteps[1:end-1], rb.timesteps[2:end]))
        h = t_end - t_start
        y, st = rb.layer(x, ps, st)
        x = x .+ h .* y
    end
    return x, st
end
```

### 1.4 Migration Timeline

**Week 1-2:** Design and prototype
- Create Lux layer equivalents for current architecture
- Prototype PotentialNN using Lux
- Validate gradient computations match current implementation

**Week 3-4:** Implementation
- Replace SingleLayer with Lux Dense + custom activation
- Replace NN with Lux Chain
- Replace ResNN with custom Lux residual layer
- Update PotentialNN to use Lux models

**Week 5:** Testing and validation
- Run all existing tests with new implementation
- Add gradient comparison tests (old vs. new)
- Performance benchmarking

**Week 6:** Documentation and examples
- Update README with new API
- Create migration guide
- Update examples to use Lux

### 1.5 Backward Compatibility

**Deprecation Strategy:**
```julia
# Provide deprecation warnings for 2 minor versions
Base.@deprecate NN(layers) create_mfg_network_legacy(layers)

function create_mfg_network_legacy(layers)
    @warn "NN constructor is deprecated. Use Lux models directly. See migration guide."
    # Provide compatibility wrapper
end
```

### 1.6 Success Criteria

- ✅ All existing tests pass with new implementation
- ✅ Gradient computations match within `rtol=1e-10`
- ✅ Code reduction: Remove >400 lines of gradient code
- ✅ GPU functionality: Models run on CuArrays without modification
- ✅ Performance: CPU performance within 10% of current (likely faster)

---

## 2. Integration with DifferentialEquations.jl

### 2.1 Objective

Replace custom RK1/RK4 time steppers with DifferentialEquations.jl for adaptive, high-accuracy time integration.

### 2.2 Current State

**Custom Implementation:**
- `src/timeStepping.jl` (63 lines)
- Fixed time-step RK1 and RK4
- No adaptive stepping
- No stiffness detection
- Manual implementation of Runge-Kutta methods

**Limitations:**
1. Inefficient for problems with varying time scales
2. User must manually choose step size
3. No automatic error control
4. Limited algorithm selection

### 2.3 Proposed Solution

Use **DifferentialEquations.jl v7+** (part of SciML ecosystem).

**Benefits:**
- Adaptive time-stepping (automatically adjusts step size)
- Extensive algorithm library (30+ ODE solvers)
- Built-in error estimation
- Stiff problem detection
- Sensitivity analysis (adjoint methods for gradients)
- Event handling and callbacks
- GPU and distributed computing support

**Implementation:**

#### Replace integrate() function
```julia
using DifferentialEquations
using OrdinaryDiffEq

function integrate_mfg(odefun, J, U0, Θ, tspan;
                       alg=Tsit5(),  # Adaptive 5th-order Tsitouras
                       saveat=nothing,
                       kwargs...)

    # Wrap as in-place ODE
    function ode!(du, u, p, t)
        du .= odefun(J, u, Θ, t)
    end

    prob = ODEProblem(ode!, U0, (tspan[1], tspan[end]), nothing)
    sol = solve(prob, alg; saveat=saveat, kwargs...)

    return sol
end
```

### 2.4 Algorithm Selection Guide

Provide users with guidance on algorithm selection:

| Problem Type | Recommended Algorithm | Notes |
|--------------|----------------------|-------|
| Non-stiff, moderate accuracy | `Tsit5()` | Default, adaptive 5th order |
| Non-stiff, high accuracy | `Vern7()` | 7th order, very accurate |
| Stiff problems | `RadauIIA5()` | Implicit, stable |
| Explicit RK4 (backward compat) | `RK4()` | Fixed step, 4th order |
| Low accuracy, fast | `Euler()` | 1st order |

### 2.5 Migration Path

**Backward Compatibility:**
```julia
# Provide wrapper for old API
function integrate(stepper, odefun, J, U0, Θ, tspan, N; kwargs...)
    if stepper isa RK4Step
        alg = RK4()
        dt = (tspan[2] - tspan[1]) / N
    elseif stepper isa RK1Step
        alg = Euler()
        dt = (tspan[2] - tspan[1]) / N
    else
        error("Unknown stepper type")
    end

    @warn "Old API is deprecated. Use integrate_mfg with DifferentialEquations.jl algorithms."

    return integrate_mfg(odefun, J, U0, Θ, tspan; alg=alg, dt=dt, adaptive=false, kwargs...)
end
```

### 2.6 Timeline

**Week 1:** Design and prototype (2-3 days)
**Week 2:** Implementation (3-4 days)
**Week 3:** Testing and validation (4-5 days)
**Week 4:** Documentation (2-3 days)

### 2.7 Success Criteria

- ✅ Adaptive stepping improves efficiency by >2x for typical problems
- ✅ Higher accuracy with fewer function evaluations
- ✅ Backward compatible API for existing code
- ✅ Example demonstrating adaptive vs. fixed stepping

---

## 3. Adopt Optimization.jl Ecosystem

### 3.1 Objective

Replace custom BFGS implementation with Optimization.jl + OptimizationOptimJL.jl for access to multiple optimizers and better convergence.

### 3.2 Current State

- `src/bfgs.jl` (94 lines)
- Custom line search (Armijo backtracking)
- Manual Hessian approximation
- Limited convergence diagnostics
- No alternative optimizers

### 3.3 Proposed Solution

**Packages:**
- `Optimization.jl v3-4`: Unified optimization interface
- `OptimizationOptimJL.jl v0.3`: BFGS, L-BFGS, Newton, etc.
- `OptimizationNLopt.jl`: Additional algorithms (optional)

**Implementation:**

```julia
using Optimization
using OptimizationOptimJL

function optimize_mfg(objective, gradient!, x0;
                      algorithm=BFGS(),
                      maxiter=200,
                      abstol=1e-6,
                      callback=nothing)

    # Define optimization function
    function optf(x, p)
        obj_val = objective(x)
        grad = similar(x)
        gradient!(grad, x)
        return obj_val, grad
    end

    opt_func = OptimizationFunction(optf, Optimization.AutoForwardDiff())
    prob = OptimizationProblem(opt_func, x0)

    sol = solve(prob, algorithm;
                maxiters=maxiter,
                abstol=abstol,
                callback=callback)

    return sol
end
```

**Available Algorithms:**
- `BFGS()`: Quasi-Newton, default
- `LBFGS()`: Limited-memory BFGS (better for large problems)
- `NewtonTrustRegion()`: Second-order method
- `ConjugateGradient()`: First-order, memory efficient
- `GradientDescent()`: Simple, robust

### 3.4 Benefits

1. **L-BFGS Support:** O(n) memory instead of O(n²)
2. **Better line searches:** Multiple strategies available
3. **Standardized callbacks:** Monitoring, early stopping
4. **Interoperability:** Works with AD frameworks
5. **Maintenance:** Community-supported, well-tested

### 3.5 Migration Timeline

**Week 1:** Implementation (3-4 days)
**Week 2:** Testing (3-4 days)
**Week 3:** Documentation and examples (2-3 days)

### 3.6 Success Criteria

- ✅ L-BFGS reduces memory usage by >50x for large problems
- ✅ Convergence rates match or exceed custom BFGS
- ✅ Easy algorithm switching without code changes
- ✅ Backward compatible wrapper provided

---

## 4. Parameter Management with ComponentArrays.jl

### 4.1 Objective

Replace custom `param2vec` functions with ComponentArrays.jl for automatic parameter flattening/unflattening.

### 4.2 Current State

- `src/param2vec.jl` (47 lines)
- Manual tuple traversal
- Type-unstable in places
- Requires manual synchronization between vec and param representations

### 4.3 Proposed Solution

```julia
using ComponentArrays

# Create named parameter structure
θ = ComponentArray(
    network_layer1 = (K=K1, b=b1),
    network_layer2 = (K=K2, b=b2),
    potential = (A=A, c=c, z=z)
)

# Automatic operations
θ_flat = getdata(θ)              # Flatten to vector
θ_reconstructed = ComponentArray(θ_flat, getaxes(θ))  # Reconstruct
θ.network_layer1.K               # Named access
```

**Benefits:**
1. Named parameter access (better readability)
2. Automatic AD compatibility
3. No manual synchronization needed
4. Type-stable by design
5. Composable with Lux parameter trees

### 4.4 Integration with Lux

```julia
# Lux already uses ComponentArrays internally
model = create_mfg_network(10, [20], 5)
ps, st = Lux.setup(rng, model)

# ps is already a ComponentArray!
ps_flat = ComponentArrays.getdata(ps)

# Optimize over flat parameters
sol = optimize_mfg(θ -> loss(θ, model, st), ps_flat)

# Reconstruct
ps_opt = ComponentArray(sol.u, getaxes(ps))
```

### 4.5 Timeline

**Week 1:** Implementation (2-3 days)
**Week 2:** Integration with Lux migration (2-3 days)
**Week 3:** Testing (2-3 days)

---

## 5. Testing Enhancements

### 5.1 Add Package Quality Tools

**Aqua.jl:**
```julia
using Aqua

@testset "Package Quality" begin
    Aqua.test_all(MFGnet,
                  ambiguities=false,
                  unbound_args=true,
                  undefined_exports=true)
end
```

**JET.jl (Static Analysis):**
```julia
using JET

@testset "Type Stability" begin
    @test_opt target_modules=(MFGnet,) create_mfg_network(10, [20], 5)
end
```

### 5.2 Add Finite Difference Validation

```julia
using FiniteDifferences

@testset "Gradient Correctness" begin
    fdm = central_fdm(5, 1)
    ∇f_fd = grad(fdm, objective, x)[1]
    ∇f_ad = gradient(objective, x)[1]
    @test ∇f_fd ≈ ∇f_ad rtol=1e-5
end
```

### 5.3 Performance Regression Tests

```julia
using BenchmarkTools

@testset "Performance Benchmarks" begin
    @test (@benchmark forward_pass($model, $x, $ps, $st)).time < 1e6  # 1ms
end
```

---

## 6. Documentation with Documenter.jl

### 6.1 Setup

```julia
# docs/make.jl
using Documenter
using MFGnet

makedocs(
    sitename = "MFGnet.jl",
    modules = [MFGnet],
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Tutorials" => [
            "Basic Example" => "tutorials/basic.md",
            "Advanced: Custom Layers" => "tutorials/custom_layers.md",
            "GPU Computing" => "tutorials/gpu.md"
        ],
        "API Reference" => [
            "Neural Networks" => "api/networks.md",
            "Mean Field Games" => "api/mfg.md",
            "Cost Functions" => "api/costs.md",
            "Optimization" => "api/optimization.md"
        ],
        "Migration Guide" => "migration.md",
        "Contributing" => "contributing.md"
    ]
)

deploydocs(
    repo = "github.com/USER/MFGnet.jl.git"
)
```

### 6.2 Documentation Standards

- Every exported function has comprehensive docstring
- Include mathematical notation in LaTeX
- Provide code examples in docstrings
- Cross-reference related functions
- Include complexity analysis where relevant

---

## 7. Timeline and Milestones

### Phase 3A: Core Migration (Weeks 1-8)

**Weeks 1-4: Lux.jl Migration**
- Week 1-2: Design, prototype, validate
- Week 3-4: Implementation, testing
- Deliverable: Working Lux-based neural networks

**Weeks 5-6: DifferentialEquations.jl Integration**
- Week 5: Implementation
- Week 6: Testing, documentation
- Deliverable: Adaptive time-stepping support

**Weeks 7-8: Optimization.jl Integration**
- Week 7: Implementation, testing
- Week 8: Documentation, examples
- Deliverable: Multiple optimizer support

### Phase 3B: Polish and Enhancement (Weeks 9-12)

**Week 9: ComponentArrays Integration**
- Implement parameter management
- Integration testing

**Week 10: Documentation**
- Setup Documenter.jl
- Write tutorials and API docs
- Create migration guide

**Week 11: Enhanced Testing**
- Add Aqua, JET, FiniteDifferences
- Performance benchmarks
- CI/CD enhancements

**Week 12: Final Review and Release**
- Code review
- Performance validation
- Release v0.3.0

---

## 8. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|---------|------------|
| Breaking API changes | High | High | Provide deprecation warnings, compatibility layer for 2 versions |
| Performance regression | Medium | High | Benchmark before/after, optimize hot paths |
| GPU compatibility issues | Medium | Medium | Test on CUDA early, use KernelAbstractions.jl for portability |
| Lux ecosystem changes | Low | Medium | Pin to stable Lux versions, monitor releases |
| Learning curve for users | High | Low | Comprehensive migration guide, examples |

---

## 9. Success Metrics

### Quantitative

- **Code Reduction:** 40-50% (1900 LOC → 800-1000 LOC)
- **Test Coverage:** Maintain >80%
- **Performance:** CPU within 10% of current, GPU 10-100x faster
- **Memory:** L-BFGS reduces large problem memory by >50x
- **Build Time:** <1 minute
- **Documentation Coverage:** 100% of exported functions

### Qualitative

- Easier to onboard new contributors
- Positive community feedback
- Adoption by research groups
- Integration into course materials
- Cited in publications

---

## 10. Compatibility Matrix

| Package | Min Version | Julia Compat | Purpose |
|---------|------------|--------------|---------|
| Lux.jl | 0.5 | 1.9+ | Neural networks |
| ComponentArrays.jl | 0.15 | 1.6+ | Parameter management |
| DifferentialEquations.jl | 7 | 1.9+ | ODE solving |
| Optimization.jl | 3 | 1.9+ | Optimization interface |
| Documenter.jl | 1 | 1.9+ | Documentation |
| Aqua.jl | 0.8 | 1.9+ | Package quality |
| JET.jl | 0.9 | 1.10+ | Static analysis |

**Julia Version Support:** 1.10+

---

## 11. Post-Release Support

### Version 0.3.0 Release Plan

1. **Pre-release (beta):** 2-week testing period
2. **Release:** Publish to Julia General Registry
3. **Announcement:** Post on Julia Discourse, Twitter
4. **Support:** Monitor issues, provide migration assistance

### Deprecation Timeline

- **v0.3.0:** Introduce new API, deprecate old with warnings
- **v0.4.0:** Remove compatibility layer for old API
- **v1.0.0:** Stable API guarantee

---

## 12. Open Questions

1. **GPU Default:** Should GPU be default if CUDA is available?
   - **Decision Needed:** Week 4 of Phase 3A

2. **Breaking Changes:** Accept breaking changes or maintain full backward compat?
   - **Proposal:** Deprecation warnings for 2 minor versions
   - **Decision Needed:** Before Phase 3A starts

3. **Examples Repository:** Separate repo for extensive examples?
   - **Proposal:** Keep in main repo until >20 examples
   - **Decision Needed:** Week 10

---

## 13. Resources Required

### Development Time
- **Lead Developer:** 8-12 weeks full-time equivalent
- **Code Review:** 1-2 hours/week from senior developer
- **Testing:** 20-30 hours community testing

### Infrastructure
- **CI/CD:** GitHub Actions (free for open source)
- **Documentation Hosting:** GitHub Pages (free)
- **GPU Testing:** Access to CUDA-capable machine (could use GitHub's gpus)

### Community
- **Beta Testers:** 5-10 active users
- **Documentation Reviewers:** 2-3 technical writers
- **Benchmark Contributors:** 1-2 performance experts

---

## Appendix A: Code Examples

### Before (Current)
```julia
# Custom network
N = NN([SingleLayer(); SingleLayer()])

# Custom time stepping
stepper = RK4Step()
U = integrate(stepper, odefun, J, U0, Θ, [0.0, 1.0], 100)

# Custom BFGS
x, flag, his, X, H = bfgs(f, fdf, x0, maxIter=200)

# Custom param management
Θvec = param2vec(Θparm)
Θparm = vec2param(Θvec, Θparm)
```

### After (Proposed)
```julia
# Lux network
model = create_mfg_network(10, [20, 20], 5)
ps, st = Lux.setup(rng, model)

# DifferentialEquations.jl
sol = integrate_mfg(odefun, J, U0, Θ, [0.0, 1.0]; alg=Tsit5())

# Optimization.jl
sol = optimize_mfg(objective, gradient!, x0; algorithm=LBFGS())

# ComponentArrays (automatic)
ps_flat = getdata(ps)
ps_reconstructed = ComponentArray(ps_flat, getaxes(ps))
```

---

## Appendix B: References

1. Lux.jl Documentation: https://lux.csail.mit.edu/
2. DifferentialEquations.jl Docs: https://docs.sciml.ai/DiffEqDocs/
3. Optimization.jl Guide: https://docs.sciml.ai/Optimization/
4. ComponentArrays.jl: https://github.com/jonniedie/ComponentArrays.jl
5. Julia Package Development: https://pkgdocs.julialang.org/

---

**Document Status:** ✅ Complete
**Next Review:** Before Phase 3A kickoff
**Approval Required From:** Project maintainers, community feedback
