# TDD Quick Start Guide for MFGnet.jl Migration

**Purpose:** Quick reference for test-driven development during the Julia ecosystem migration.

---

## The TDD Mantra

```
🔴 RED → 🟢 GREEN → 🔵 REFACTOR
Write failing test → Make it pass → Improve code
```

---

## Essential Commands

### Run Tests

```bash
# Quick smoke test (30 sec)
julia --project=. test/runtests_migration.jl --quick

# Full test suite (30 min)
julia --project=. test/runtests_migration.jl --full

# Original tests (verify no regression)
julia --project=. -e 'using Pkg; Pkg.test()'

# Single test file
julia --project=. test/unit/test_ode_wrapper.jl
```

### Create Test Problem

```julia
include("test/utils/test_problems.jl")
using .TestProblems

# Fast 1D problem (unit tests)
prob = gaussian_1d_transport(nex=50)

# 2D problem (integration tests)
prob = gaussian_2d_transport(nex=200)

# High-dimensional (performance)
prob = high_dim_problem(d=10, nex=500)
```

### Compare Results

```julia
include("test/utils/test_utils.jl")
using .TestUtils

# Compare objectives
compare_objectives(Jc1, Jc2; rtol=1e-3)

# Compare gradients
compare_gradients(∇1, ∇2; rtol=1e-3)

# Compare states
compare_states(U1, U2; rtol=1e-3)
```

### Validate Gradients

```julia
using Zygote

# Finite difference check
∇_AD = Zygote.gradient(f, Θ)[1]
∇_FD = finite_difference_gradient(f, Θ; h=1e-5)
compare_gradients(∇_AD, ∇_FD; rtol=1e-3)

# Taylor test (should show O(h²))
taylor_test(f, ∇f, Θ; verbose=true)
```

---

## TDD Workflow Templates

### Template 1: Unit Test for New Component

```julia
# test/unit/test_my_feature.jl

using Test
using MFGnet

include("../utils/test_utils.jl")
include("../utils/test_problems.jl")
using .TestUtils
using .TestProblems

@testset "My Feature" begin

    @testset "Basic construction" begin
        prob = gaussian_1d_transport(nex=30)
        mfg, Θ = prob.mfg, prob.Θ_init

        # ❌ This will fail initially
        result = my_new_feature(mfg, Θ)

        @test result isa ExpectedType
        @test isfinite(result.value)
    end

    @testset "Edge cases" begin
        # Test with single particle
        prob = gaussian_1d_transport(nex=1)
        # ...
    end
end
```

**Workflow:**
1. Write test → Run → Fails ❌
2. Implement `my_new_feature` → Run → Passes ✅
3. Refactor → Run → Still passes ✅

### Template 2: Comparison Test (Legacy vs New)

```julia
@testset "Feature Equivalence" begin
    prob = gaussian_1d_transport(nex=50)
    mfg, Θ = prob.mfg, prob.Θ_init

    # Legacy implementation
    result_legacy = old_implementation(mfg, Θ)

    # New implementation
    result_new = new_implementation(mfg, Θ)

    # Should be close
    compare_objectives(result_legacy, result_new;
                      abstol=1e-6,
                      reltol=1e-3,
                      name="Legacy vs New")
end
```

### Template 3: Gradient Validation Test

```julia
@testset "Gradient Validation" begin
    prob = gaussian_1d_transport(nex=20)
    mfg, Θ = prob.mfg, prob.Θ_init

    f(θ) = my_feature(mfg, θ)
    ∇f(θ) = Zygote.gradient(f, θ)[1]

    # Finite difference check
    ∇_AD = ∇f(Θ)
    ∇_FD = finite_difference_gradient(f, Θ; h=1e-5)
    compare_gradients(∇_AD, ∇_FD; rtol=1e-3)

    # Taylor test
    taylor_test(f, ∇f, Θ; verbose=false)
end
```

### Template 4: End-to-End Test

```julia
@testset "Full Training Pipeline" begin
    prob = gaussian_1d_transport(nex=50)
    mfg, Θ_init = prob.mfg, prob.Θ_init

    # Define objective and gradient
    function fdf(θ_vec)
        Θ = reconstruct_params(θ_vec, Θ_init)
        using Zygote
        Jc, ∇Jc = Zygote.withgradient(θ -> mfg(θ), Θ)
        return Jc, vec_params(∇Jc[1])
    end

    f(θ_vec) = fdf(θ_vec)[1]
    θ0 = vec_params(Θ_init)

    # Optimize
    θ_opt, flag, his, _, _ = bfgs(f, fdf, θ0;
                                  maxIter=20,
                                  atol=1e-6,
                                  out=-1)

    # Check convergence
    @test flag in [0, -1]
    @test his[end, 1] < his[1, 1]  # Loss decreases
end
```

---

## Common Test Patterns

### Pattern: Test with Multiple Precisions

```julia
@testset "Precision types" begin
    for R in [Float64, Float32]
        prob = gaussian_1d_transport(nex=30; R=R)
        mfg, Θ = prob.mfg, prob.Θ_init

        Jc = mfg(Θ)
        @test eltype(Jc) == R
    end
end
```

### Pattern: Test Solver Consistency

```julia
@testset "Solver consistency" begin
    prob = gaussian_1d_transport(nex=50)
    mfg, Θ = prob.mfg, prob.Θ_init

    solvers = [
        ("Tsit5", Tsit5()),
        ("Vern7", Vern7()),
    ]

    results = Dict()
    for (name, solver) in solvers
        Jc = mfg(Θ; use_diffeq=true, solver=solver)
        results[name] = Jc
    end

    # All should be close
    vals = collect(values(results))
    for i in 1:length(vals)-1
        @test isapprox(vals[i], vals[i+1]; rtol=1e-4)
    end
end
```

### Pattern: Performance Test

```julia
@testset "Performance comparison" begin
    prob = gaussian_1d_transport(nex=100)
    mfg, Θ = prob.mfg, prob.Θ_init

    # Benchmark legacy
    t_legacy = @elapsed mfg(Θ)

    # Benchmark new
    t_new = @elapsed mfg(Θ; use_diffeq=true, solver=Tsit5())

    # New should be faster
    speedup = t_legacy / t_new
    println("Speedup: $(speedup)x")

    @test speedup > 2.0  # At least 2x faster
end
```

---

## Debugging Checklist

### When Test Fails

1. **Run in isolation**
   ```julia
   # Run just the failing test
   @testset "Failing test" begin
       # ... test code ...
   end
   ```

2. **Add diagnostics**
   ```julia
   println("Expected: $expected")
   println("Got: $actual")
   println("Difference: $(expected - actual)")
   ```

3. **Reduce problem size**
   ```julia
   prob = gaussian_1d_transport(nex=10)  # Smaller for debugging
   ```

4. **Check types**
   ```julia
   @show typeof(result)
   @show size(result)
   ```

5. **Relax tolerances temporarily**
   ```julia
   compare_objectives(Jc1, Jc2; rtol=1e-2)  # See if close
   ```

### When Gradient Validation Fails

1. **Check function is smooth**
   ```julia
   # Plot function values
   hs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
   vals = [f(add_direction(Θ, v, h)) for h in hs]
   ```

2. **Use central differences**
   ```julia
   ∇_FD = finite_difference_gradient(f, Θ; method=:central, h=1e-5)
   ```

3. **Check single component**
   ```julia
   # Test gradient w.r.t. first parameter only
   grad_vec = vec_params(∇Jc)
   fd_vec = vec_params(∇_FD)
   @test isapprox(grad_vec[1], fd_vec[1]; rtol=1e-3)
   ```

4. **Increase FD step size**
   ```julia
   ∇_FD = finite_difference_gradient(f, Θ; h=1e-4)  # Larger h
   ```

---

## Tolerance Guidelines

### When to Use Tight Tolerances (1e-8)

- Generating reference solutions
- Testing against analytical solutions
- Unit tests with identical implementations

### When to Use Moderate Tolerances (1e-3 - 1e-6)

- Comparing different solvers
- Comparing legacy vs new implementations
- Finite difference validation
- End-to-end tests

### When to Use Relaxed Tolerances (1e-2)

- Long-time integration
- Stiff problems
- First-order methods
- Large-scale problems

---

## Test Naming Conventions

### Test Files

```
test_<component>.jl         # Unit test
test_<feature>_comparison.jl # Comparison test
test_<property>_validation.jl # Validation test
```

### Test Sets

```julia
@testset "Component - Specific Feature" begin
    # Tests for specific feature of a component
end

@testset "Edge Case - Description" begin
    # Edge case tests
end

@testset "Performance - Description" begin
    # Performance tests
end
```

---

## Pre-Commit Checklist

Before committing code, ensure:

- [ ] Tests written BEFORE implementation
- [ ] All tests pass locally
- [ ] New features have unit tests
- [ ] Gradients validated (if applicable)
- [ ] Performance acceptable
- [ ] Edge cases covered
- [ ] Documentation updated
- [ ] No deprecation warnings (except intended)

```bash
# Run this before committing
julia --project=. test/runtests_migration.jl --full
```

---

## Common Mistakes to Avoid

### ❌ DON'T: Implement before testing

```julia
# BAD: Write implementation first
function my_feature(args...)
    # ... implementation ...
end

# Then write tests
@testset "My feature" begin
    # ... tests ...
end
```

### ✅ DO: Write tests first

```julia
# GOOD: Write test first
@testset "My feature" begin
    result = my_feature(args...)  # ❌ Fails - not implemented
    @test result isa ExpectedType
end

# Then implement
function my_feature(args...)
    # ... implementation ...
end  # ✅ Now test passes
```

### ❌ DON'T: Use overly tight tolerances

```julia
# BAD: Too strict for solver comparison
@test Jc_legacy ≈ Jc_new atol=1e-12  # Will fail
```

### ✅ DO: Use appropriate tolerances

```julia
# GOOD: Reasonable for solver differences
compare_objectives(Jc_legacy, Jc_new; rtol=1e-3)
```

### ❌ DON'T: Test too many things at once

```julia
# BAD: Testing many things in one test
@test result.objective < 1.0 &&
      result.converged &&
      result.iterations < 100 &&
      norm(result.gradient) < 1e-5
```

### ✅ DO: One assertion per concept

```julia
# GOOD: Separate tests for each property
@test result.objective < 1.0
@test result.converged
@test result.iterations < 100
@test norm(result.gradient) < 1e-5
```

---

## Useful Snippets

### Quick gradient check

```julia
f(θ) = mfg(θ)
∇_AD = Zygote.gradient(f, Θ)[1]
∇_FD = finite_difference_gradient(f, Θ; h=1e-5)
println("Gradient error: ", norm(vec_params(∇_AD) - vec_params(∇_FD)))
```

### Quick performance comparison

```julia
println("Legacy: ", @elapsed mfg(Θ))
println("DiffEq: ", @elapsed mfg(Θ; use_diffeq=true, solver=Tsit5()))
```

### Quick convergence test

```julia
function fdf(θ_vec)
    Θ = reconstruct_params(θ_vec, Θ_init)
    Jc, ∇Jc = Zygote.withgradient(θ -> mfg(θ), Θ)
    return Jc, vec_params(∇Jc[1])
end

θ_opt, flag, his, _, _ = bfgs(f, fdf, vec_params(Θ_init); maxIter=10, out=1)
```

---

## Help & Resources

### Documentation

- `TEST_ARCHITECTURE.md` - Full test design (comprehensive)
- `README_TESTS.md` - Test suite guide (user-focused)
- `TEST_STRATEGY_SUMMARY.md` - Executive summary (high-level)
- `TDD_QUICKSTART.md` - This file (quick reference)

### Example Tests

- `test/unit/test_ode_wrapper.jl` - Unit test examples
- `test/validation/test_gradients.jl` - Gradient validation examples
- `test/e2e/test_full_training.jl` - End-to-end examples

### Get Help

- Read existing test examples
- Check test utilities in `test/utils/`
- Open GitHub issue for questions
- Review test architecture document

---

## Remember

**Write tests first. Always.**

The extra time spent writing tests saves debugging time later. Tests are:
- Documentation of expected behavior
- Safety net for refactoring
- Proof of correctness
- Performance baseline

**Happy testing! 🎯**
