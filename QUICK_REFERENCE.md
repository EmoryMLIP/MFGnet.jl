# MFGnet.jl Optimization Migration - Quick Reference

## 🚀 Quick Start

### Install New Dependencies
```bash
julia --project=. -e 'using Pkg; Pkg.add(["ComponentArrays", "Optimization", "OptimizationOptimJL", "OptimizationOptimisers"])'
```

### Basic Usage Pattern

**Old (Custom BFGS):**
```julia
Θ0 = param2vec(parms)
f = (Θ) -> evalObj(J, Θ, parms, ps)
fdf = (Θ) -> evalObjAndGrad(J, Θ, parms, ps)
Θopt, flag, His, X, H = bfgs(f, fdf, Θ0, maxIter=200)
```

**New (Optimization.jl):**
```julia
result = train_mfg(J, parms, optimizer=LBFGS(), maxiters=200)
Θopt = result.Θ_opt
```

---

## 📚 API Quick Reference

### Core Functions

#### `train_mfg` - High-level training
```julia
result = train_mfg(
    mfg,                    # MeanFieldGame problem
    Θ_init,                 # Initial parameters (nested tuple)
    optimizer = LBFGS(),    # Optimization algorithm
    maxiters = 1000,        # Maximum iterations
    abstol = 1e-8,          # Absolute tolerance
    callback = nothing,     # Callback function
    verbose = true          # Print progress
)

# Returns: (Θ_opt, loss_history, iter_history, sol, time_elapsed)
```

#### `create_optimization_problem` - Manual control
```julia
prob = create_optimization_problem(
    mfg, Θ_init,
    adtype = Optimization.AutoZygote()
)

sol = solve_optimization(prob, LBFGS(), maxiters=1000)
```

#### `create_mfg_callback` - Callback helper
```julia
callback, His = create_mfg_callback(
    training_mfg = J,
    validation_mfg = Jv,
    Θ_structure = get_param_structure(Θ_init),
    sample_freq = 25,
    save_freq = 25,
    save_path = "checkpoints/"
)
```

### Utility Functions

```julia
# ComponentArray conversion
θ = params_to_componentarray(Θ)           # Tuple → ComponentArray
Θ = componentarray_to_params(θ, structure) # ComponentArray → Tuple
structure = get_param_structure(Θ)         # Extract structure template
n = count_parameters(Θ)                    # Count total parameters
```

---

## 🎯 Optimizer Selection

### Available Algorithms

| Algorithm | Use Case | Memory | Speed | Import |
|-----------|----------|--------|-------|--------|
| **LBFGS()** | Medium/large (100-10k params) | Low | Fast | OptimizationOptimJL |
| **BFGS()** | Small (<100 params) | High | Fast | OptimizationOptimJL |
| **Newton()** | Very small, high accuracy | High | Slow | OptimizationOptimJL |
| **ConjugateGradient()** | Large, no Hessian | Low | Medium | OptimizationOptimJL |
| **Adam(lr)** | Very large, stochastic | Low | Fast | OptimizationOptimisers |
| **AdaGrad(lr)** | Sparse gradients | Low | Fast | OptimizationOptimisers |
| **RMSProp(lr)** | Non-stationary | Low | Fast | OptimizationOptimisers |

### Auto-Selection
```julia
# Automatic based on problem size
result = train_mfg(mfg, Θ, optimizer=:auto)

# Manual selection
n_params = count_parameters(Θ)
if n_params < 100
    optimizer = BFGS()
elseif n_params < 10_000
    optimizer = LBFGS()
else
    optimizer = Adam(0.001)
end
```

---

## 📝 Code Snippets

### Snippet 1: Basic Training
```julia
using MFGnet
using Optimization
using OptimizationOptimJL

result = train_mfg(
    mfg, Θ_init,
    optimizer = LBFGS(),
    maxiters = 200,
    abstol = 1e-10,
    verbose = true
)

@printf("Final loss: %.6e\n", result.sol.objective)
```

### Snippet 2: Algorithm Comparison
```julia
algorithms = [
    ("LBFGS", LBFGS()),
    ("BFGS", BFGS()),
    ("Adam", Adam(0.01))
]

for (name, alg) in algorithms
    result = train_mfg(mfg, Θ_init, optimizer=alg, maxiters=100, verbose=false)
    @printf("%s: %.6e in %.2fs\n", name, result.sol.objective, result.time_elapsed)
end
```

### Snippet 3: With Validation and Checkpointing
```julia
callback, His = create_mfg_callback(
    training_mfg = mfg_train,
    validation_mfg = mfg_val,
    Θ_structure = get_param_structure(Θ_init),
    sample_freq = 25,
    save_freq = 50,
    save_path = "results/experiment1/"
)

result = train_mfg(
    mfg_train, Θ_init,
    optimizer = LBFGS(),
    maxiters = 500,
    callback = callback
)
```

### Snippet 4: Custom Callback
```julia
function my_callback(state, loss_val)
    if state.iter % 10 == 0
        @printf("Iter %4d: loss = %.6e\n", state.iter, loss_val)
    end

    # Early stopping
    if loss_val < 1e-8
        return true  # Stop
    end

    return false  # Continue
end

result = train_mfg(mfg, Θ_init, callback=my_callback)
```

### Snippet 5: Manual Problem Setup
```julia
using ComponentArrays

# Create problem
prob = create_optimization_problem(mfg, Θ_init)

# Solve with custom options
sol = solve_optimization(
    prob, LBFGS(),
    maxiters = 1000,
    abstol = 1e-10,
    g_tol = 1e-8,  # Gradient tolerance
    f_tol = 1e-8,  # Function tolerance
    verbose = true
)

# Extract result
Θ_opt = componentarray_to_params(sol.u, prob.param_structure)
```

---

## 🔄 Migration Patterns

### Pattern 1: Simple Optimization

**Before:**
```julia
parms = (w0, (ΘN), A0, b0, z0)
ps = Flux.params(parms)
Θ0 = MFGnet.param2vec(parms)

f = (Θ) -> evalObj(J, Θ, parms, ps)
fdf = (Θ) -> evalObjAndGrad(J, Θ, parms, ps)

Θopt, flag, His, X, H = MFGnet.bfgs(f, fdf, Θ0, maxIter=200, atol=1e-10)
parms_opt = vec2param!(Θopt, parms)
```

**After:**
```julia
Θ_init = (w0, (ΘN), A0, b0, z0)

result = train_mfg(J, Θ_init, optimizer=LBFGS(), maxiters=200, abstol=1e-10)
Θ_opt = result.Θ_opt
```

### Pattern 2: With Callback

**Before:**
```julia
cb = function(J, Jv, iter, His, parms, doPlots)
    Jvc = Jv(parms)
    His[iter,:] = [J.cs[1:5]..., Jv.cs[1:5]...]
    @printf("iter=%04d obj=%1.3e\n", iter, sum(His[iter,1:5]))
end

cbBFGS = (iter) -> cb(J, Jv, iter, His, parms, doPlots)

Θopt, flag, His, X, H = MFGnet.bfgs(f, fdf, Θ0, cb=cbBFGS, ...)
```

**After:**
```julia
callback, His = create_mfg_callback(
    training_mfg = J,
    validation_mfg = Jv,
    Θ_structure = get_param_structure(parms),
    verbose = true
)

result = train_mfg(J, parms, callback=callback, ...)
```

### Pattern 3: Multiple Training Levels

**Before:**
```julia
for level in 1:length(nTrain)
    X0 = sample(rho0, nTrain[level])
    J = MeanFieldGame(F, G, X0, rho0, w, ...)

    Θ0 = (level == 1) ? param2vec(Θ_init) : param2vec(Θ_prev)

    Θopt, flag, His, X, H = MFGnet.bfgs(f, fdf, Θ0, maxIter=maxIter[level])

    Θ_prev = vec2param!(Θopt, Θ_prev)
end
```

**After:**
```julia
Θ_current = Θ_init

for level in 1:length(nTrain)
    X0 = sample(rho0, nTrain[level])
    J = MeanFieldGame(F, G, X0, rho0, w, ...)

    result = train_mfg(J, Θ_current, optimizer=LBFGS(), maxiters=maxIter[level])

    Θ_current = result.Θ_opt  # Warm start next level
end
```

---

## 🐛 Common Issues

### Issue 1: ComponentArray Reconstruction Error
**Symptom:** `KeyError` when accessing ComponentArray fields

**Solution:** Ensure structure template is saved before optimization
```julia
structure = get_param_structure(Θ_init)  # Save before optimization
prob = create_optimization_problem(mfg, Θ_init)
sol = solve_optimization(prob, LBFGS())
Θ_opt = componentarray_to_params(sol.u, structure)  # Use saved structure
```

### Issue 2: Callback Not Called
**Symptom:** Custom callback function never executes

**Solution:** Check callback signature and return value
```julia
function callback(state, loss_val)  # Correct signature
    # Your logic here
    return false  # Must return Bool (false = continue, true = stop)
end
```

### Issue 3: Gradient Computation Fails
**Symptom:** `MethodError` or gradient is `nothing`

**Solution:** Ensure MFG is differentiable
```julia
# Test gradient computation
using Zygote
loss, grad = Zygote.withgradient(Θ -> mfg(Θ), Θ_init)
@assert !isnothing(grad[1])  # Should not be nothing
```

### Issue 4: Optimizer Not Converging
**Symptom:** Loss not decreasing, optimization stops early

**Solutions:**
```julia
# 1. Try different optimizer
optimizer = Adam(0.01)  # First-order instead of LBFGS

# 2. Adjust tolerances
abstol = 1e-6  # Less strict
reltol = 1e-4

# 3. Increase iterations
maxiters = 2000

# 4. Check gradients
result = train_mfg(mfg, Θ, optimizer=LBFGS(), g_tol=1e-6)
```

---

## 📦 File Structure

```
/home/user/MFGnet.jl/
├── src/
│   ├── MFGnet.jl                      # Main module
│   ├── bfgs.jl                        # LEGACY (with deprecation)
│   ├── param2vec.jl                   # LEGACY (still used internally)
│   ├── utils.jl                       # LEGACY (with deprecation)
│   ├── componentarray_utils.jl        # NEW: ComponentArray conversion
│   └── optimization_wrapper.jl        # NEW: Optimization.jl integration
│
├── test/
│   ├── runtests.jl
│   ├── test_componentarrays.jl        # NEW
│   ├── test_optimization_integration.jl  # NEW
│   └── test_legacy_comparison.jl      # NEW
│
├── examples/
│   ├── modern_optimization_demo.jl    # NEW
│   ├── modern_obstacle_experiment.jl  # NEW
│   └── compare_optimizers.jl          # NEW
│
├── docs/
│   ├── MIGRATION_GUIDE.md             # NEW
│   ├── API_REFERENCE.md               # NEW
│   └── OPTIMIZER_SELECTION.md         # NEW
│
└── OPTIMIZATION_MIGRATION_DETAILED.md # Implementation guide
```

---

## 🧪 Testing Commands

```bash
# Run all tests
julia --project=. -e 'using Pkg; Pkg.test()'

# Run specific test
julia --project=. test/test_componentarrays.jl

# Run with coverage
julia --project=. -e 'using Pkg; Pkg.test(coverage=true)'

# Benchmark comparison
julia --project=. examples/compare_optimizers.jl
```

---

## 📊 Performance Tips

### Tip 1: Use LBFGS for Most Problems
```julia
# Good default for 100-10,000 parameters
optimizer = LBFGS()
```

### Tip 2: Warm Start Between Levels
```julia
# Use previous solution as initial guess
Θ_current = Θ_init
for level in levels
    result = train_mfg(mfg[level], Θ_current, ...)
    Θ_current = result.Θ_opt  # Warm start
end
```

### Tip 3: Adjust Tolerances
```julia
# For exploration: loose tolerances, fast
result = train_mfg(mfg, Θ, maxiters=100, abstol=1e-6)

# For production: tight tolerances, accurate
result = train_mfg(mfg, Θ, maxiters=1000, abstol=1e-10)
```

### Tip 4: Profile Before Optimizing
```julia
using Profile

@profile train_mfg(mfg, Θ, maxiters=10)
Profile.print()
```

---

## 🔗 Quick Links

- **Detailed Guide:** [OPTIMIZATION_MIGRATION_DETAILED.md](OPTIMIZATION_MIGRATION_DETAILED.md)
- **Summary:** [MIGRATION_SUMMARY.md](MIGRATION_SUMMARY.md)
- **Architecture:** [MIGRATION_ARCHITECTURE.md](MIGRATION_ARCHITECTURE.md)

### External Documentation

- [Optimization.jl](https://docs.sciml.ai/Optimization/stable/)
- [OptimizationOptimJL](https://docs.sciml.ai/Optimization/stable/optimization_packages/optim/)
- [ComponentArrays.jl](https://jonniedie.github.io/ComponentArrays.jl/stable/)
- [Zygote.jl](https://fluxml.ai/Zygote.jl/stable/)

---

## ❓ FAQ

### Q: Do I need to rewrite all my code?
**A:** No! Legacy `bfgs` continues to work. Migrate at your own pace.

### Q: Will my published results change?
**A:** No. Use `bfgs_legacy` to reproduce exact results.

### Q: Which optimizer should I use?
**A:** Start with LBFGS. Try Adam if LBFGS is slow or uses too much memory.

### Q: How do I migrate callbacks?
**A:** Use `create_mfg_callback` helper or write custom callback following Optimization.jl convention.

### Q: Can I use GPU?
**A:** Yes, ComponentArrays and Optimization.jl support GPU. Ensure your MFG problem is GPU-compatible.

### Q: What if I get errors?
**A:** Check Common Issues section above, or consult detailed guide.

---

## 🎓 Learning Path

1. **Day 1:** Read MIGRATION_SUMMARY.md
2. **Day 2:** Run examples/modern_optimization_demo.jl
3. **Day 3:** Migrate one simple example
4. **Week 1:** Use new API for new projects
5. **Month 1:** Gradually migrate existing code

---

## 📞 Support

- **Documentation:** See detailed migration guide
- **Examples:** Check `examples/modern_*.jl`
- **Issues:** Open GitHub issue
- **Questions:** See FAQ or detailed guide

---

**Last Updated:** 2025-11-06
**Version:** 2.0 (Implementation Ready)
