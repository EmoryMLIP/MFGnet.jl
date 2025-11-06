# MFGnet.jl Test Suite

Comprehensive testing infrastructure for the Julia ecosystem migration (DifferentialEquations.jl and Optimization.jl).

## Overview

This test suite implements Test-Driven Development (TDD) principles for migrating MFGnet.jl to modern Julia packages. The tests are designed to:

1. **Validate correctness** - Ensure new implementation matches legacy behavior
2. **Check performance** - Verify expected speedups are achieved
3. **Ensure backward compatibility** - Legacy code continues to work
4. **Enable safe refactoring** - Catch regressions immediately

## Test Structure

```
test/
├── runtests.jl                    # Original test runner (preserved)
├── runtests_migration.jl          # NEW: Migration test runner
├── TEST_ARCHITECTURE.md           # Full test design document
├── README_TESTS.md               # This file
│
├── utils/                         # Test utilities
│   ├── test_utils.jl             # Comparison, FD gradients, Taylor tests
│   ├── test_problems.jl          # Standard test problems library
│   └── test_tolerances.jl        # Tolerance specifications
│
├── unit/                          # Unit tests for new components
│   ├── test_ode_wrapper.jl       # ODE problem construction
│   ├── test_optimization_wrapper.jl
│   ├── test_componentarrays.jl
│   └── test_sensitivity.jl
│
├── integration/                   # Integration tests
│   ├── test_ode_solvers.jl       # Solver comparisons
│   ├── test_backward_compat.jl   # Legacy API compatibility
│   └── test_numerical_equivalence.jl
│
├── validation/                    # Validation tests
│   ├── test_gradients.jl         # Gradient correctness
│   ├── test_convergence.jl       # Optimization convergence
│   └── test_conservation.jl      # Physical properties
│
├── e2e/                          # End-to-end tests
│   ├── test_full_training.jl     # Complete training pipeline
│   └── test_benchmarks.jl        # Performance benchmarks
│
└── reference_data/               # Baseline solutions
    └── golden_solutions.jld2
```

## Running Tests

### Quick Smoke Test (< 30 seconds)

```bash
julia --project=. test/runtests_migration.jl --quick
```

Runs basic sanity checks to ensure nothing is broken.

### Unit Tests Only (< 5 minutes)

```bash
julia --project=. test/runtests_migration.jl --unit
```

Tests individual components in isolation.

### Full Test Suite (< 30 minutes)

```bash
julia --project=. test/runtests_migration.jl --full
```

Runs all tests including integration, validation, and end-to-end tests.

### Original Tests (Backward Compatibility)

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Runs the original test suite to ensure no regression.

## Test Categories

### 1. Unit Tests

**Purpose:** Test individual components in isolation.

**Files:**
- `test_ode_wrapper.jl` - ODE problem construction and solving
- `test_optimization_wrapper.jl` - Optimization problem setup
- `test_componentarrays.jl` - Parameter structure conversion
- `test_sensitivity.jl` - Gradient computation through ODEs

**Example:**
```julia
@testset "ODE Wrapper Construction" begin
    prob_data = gaussian_1d_transport(nex=20)
    mfg, Θ = prob_data.mfg, prob_data.Θ_init

    prob = create_ode_problem(mfg, Θ)
    @test prob isa ODEMFGProblem
end
```

### 2. Integration Tests

**Purpose:** Test component interactions and backward compatibility.

**Files:**
- `test_backward_compat.jl` - Legacy API still works
- `test_numerical_equivalence.jl` - New vs old give same results
- `test_ode_solvers.jl` - Different solvers give consistent results

**Example:**
```julia
@testset "Legacy vs DiffEq Equivalence" begin
    Jc_legacy = mfg(Θ)
    Jc_diffeq = mfg(Θ, use_diffeq=true, solver=RK4())

    compare_objectives(Jc_legacy, Jc_diffeq; rtol=1e-3)
end
```

### 3. Validation Tests

**Purpose:** Verify correctness of numerical results.

**Files:**
- `test_gradients.jl` - Gradient correctness (FD check, Taylor test)
- `test_convergence.jl` - Optimization convergence properties
- `test_conservation.jl` - Physical properties preserved

**Example:**
```julia
@testset "Taylor Test" begin
    f(θ) = mfg(θ)
    ∇f(θ) = Zygote.gradient(f, θ)[1]

    # Should show quadratic convergence
    taylor_test(f, ∇f, Θ; verbose=true)
end
```

### 4. End-to-End Tests

**Purpose:** Test complete workflows from start to finish.

**Files:**
- `test_full_training.jl` - Complete training pipeline
- `test_benchmarks.jl` - Performance measurements

**Example:**
```julia
@testset "Full Training Pipeline" begin
    result = train_mfg(mfg, Θ_init;
                      optimizer=LBFGS(),
                      maxiters=100)

    @test result.loss_history[end] < result.loss_history[1]
end
```

## Test Utilities

### Standard Test Problems

The `TestProblems` module provides ready-to-use test problems:

```julia
using .TestProblems

# 1D Gaussian transport (fast, non-stiff)
prob = gaussian_1d_transport(nex=100)

# 2D crowd motion (moderate complexity)
prob = crowd_motion_2d(nex=200)

# Ring to ring (topological features)
prob = ring_to_ring_2d(nex=200)

# High-dimensional (performance testing)
prob = high_dim_problem(d=10, nex=500)
```

Each problem returns:
```julia
(
    mfg = MeanFieldGame,
    Θ_init = initial_parameters,
    properties = (stiffness=:nonstiff, dimension=2, ...)
)
```

### Comparison Functions

The `TestUtils` module provides utilities for comparing results:

```julia
using .TestUtils

# Compare objectives
compare_objectives(Jc1, Jc2; abstol=1e-6, reltol=1e-3)

# Compare gradients
compare_gradients(∇1, ∇2; abstol=1e-6, reltol=1e-3)

# Compare states
compare_states(U1, U2; abstol=1e-6, reltol=1e-3)
```

### Gradient Validation

```julia
# Finite difference validation
∇_AD = Zygote.gradient(f, Θ)[1]
∇_FD = finite_difference_gradient(f, Θ; h=1e-5)
compare_gradients(∇_AD, ∇_FD; rtol=1e-3)

# Taylor test (convergence rates)
taylor_test(f, ∇f, Θ; verbose=true, n_tests=10)

# Adjoint test
adjoint_test(J_forward, J_adjoint, v, w)
```

### Parameter Manipulation

```julia
# Flatten nested structure
v = vec_params(Θ)

# Reconstruct from flat vector
Θ_recon = reconstruct_params(v, Θ)

# Count parameters
n = count_params(Θ)

# Add direction (for line search)
Θ_new = add_direction(Θ, direction, step_size)

# Dot product
d = dot_params(Θ1, Θ2)
```

## Tolerance Specifications

Two tolerance levels are defined:

### Migration Tolerances (Relaxed)

Used when comparing legacy vs new implementations:

```julia
MIGRATION_TOLERANCES = TestTolerances(
    objective_abstol = 1e-6,
    objective_reltol = 1e-3,
    gradient_abstol = 1e-6,
    gradient_reltol = 1e-3,
    state_abstol = 1e-6,
    state_reltol = 1e-3,
    fd_reltol = 1e-3
)
```

### Reference Tolerances (Strict)

Used for generating golden reference solutions:

```julia
REFERENCE_TOLERANCES = TestTolerances(
    objective_abstol = 1e-10,
    objective_reltol = 1e-8,
    gradient_abstol = 1e-8,
    gradient_reltol = 1e-6,
    state_abstol = 1e-8,
    state_reltol = 1e-6,
    fd_reltol = 1e-4
)
```

## Writing New Tests

### Template for Unit Tests

```julia
using Test
using MFGnet

include("../utils/test_utils.jl")
include("../utils/test_problems.jl")
using .TestUtils
using .TestProblems

@testset "My New Feature" begin

    @testset "Basic functionality" begin
        # Create test problem
        prob = gaussian_1d_transport(nex=30)
        mfg, Θ = prob.mfg, prob.Θ_init

        # Test your feature
        result = my_new_feature(mfg, Θ)

        # Assertions
        @test result isa ExpectedType
        @test isfinite(result.value)
    end

    @testset "Edge cases" begin
        # Test edge cases
        # ...
    end
end
```

### Template for Validation Tests

```julia
@testset "Gradient Validation - My Feature" begin
    prob = gaussian_1d_transport(nex=20)
    mfg, Θ = prob.mfg, prob.Θ_init

    # Compute gradient
    f(θ) = my_feature(mfg, θ)
    ∇_AD = Zygote.gradient(f, Θ)[1]

    # Finite difference validation
    ∇_FD = finite_difference_gradient(f, Θ; h=1e-5)

    # Compare
    compare_gradients(∇_AD, ∇_FD;
                     abstol=1e-6,
                     reltol=1e-3)
end
```

## Continuous Integration

The test suite integrates with GitHub Actions:

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
      - uses: actions/checkout@v4
      - uses: julia-actions/setup-julia@v1
        with:
          version: ${{ matrix.julia-version }}
      - uses: julia-actions/cache@v1
      - name: Run tests
        run: julia --project=. -e 'using Pkg; Pkg.test()'
```

## Test-Driven Development Workflow

### 1. Before Implementing New Feature

```bash
# Write tests first
vim test/unit/test_my_feature.jl

# Tests should fail (feature doesn't exist yet)
julia --project=. test/runtests_migration.jl --unit
```

### 2. Implement Feature

```bash
# Implement the feature
vim src/my_feature.jl

# Tests should now pass
julia --project=. test/runtests_migration.jl --unit
```

### 3. Validate Results

```bash
# Run validation tests
julia --project=. test/runtests_migration.jl --full

# Check gradients are correct
julia --project=. test/validation/test_gradients.jl
```

### 4. Benchmark Performance

```bash
# Run benchmarks
julia --project=. test/e2e/test_benchmarks.jl
```

## Debugging Failed Tests

### 1. Run Single Test File

```julia
# Run specific test file
include("test/unit/test_ode_wrapper.jl")
```

### 2. Run Single Test Set

```julia
# Run specific test set
@testset "ODE Wrapper Construction" begin
    # ... test code ...
end
```

### 3. Enable Verbose Output

```julia
# In test utilities, set verbose=true
taylor_test(f, ∇f, Θ; verbose=true)

# Or add print statements
println("Jc_legacy = $Jc_legacy")
println("Jc_diffeq = $Jc_diffeq")
```

### 4. Reduce Problem Size

```julia
# Use smaller problems for debugging
prob = gaussian_1d_transport(nex=10)  # Instead of 100
```

### 5. Check Tolerances

```julia
# Relax tolerances temporarily to see if test is close
compare_objectives(Jc1, Jc2; abstol=1e-3, reltol=1e-2)
```

## Performance Testing

### Quick Performance Check

```julia
prob = gaussian_1d_transport(nex=50)
mfg, Θ = prob.mfg, prob.Θ_init

# Legacy
@time Jc_legacy = mfg(Θ)

# DiffEq
@time Jc_diffeq = mfg(Θ, use_diffeq=true, solver=Tsit5())
```

### Detailed Benchmarks

```julia
using BenchmarkTools

# Benchmark forward pass
@benchmark $mfg($Θ) samples=10

# Benchmark with gradient
@benchmark Zygote.gradient(θ -> $mfg(θ), $Θ) samples=5
```

## Success Criteria

Tests pass when:

### Unit Tests
- ✅ All new components can be constructed
- ✅ Basic operations work without errors
- ✅ Outputs have expected types and shapes

### Integration Tests
- ✅ Legacy vs new: objectives agree to rtol < 1e-3
- ✅ Legacy vs new: gradients agree to rtol < 1e-3
- ✅ All original tests still pass

### Validation Tests
- ✅ Gradients match finite differences (rtol < 1e-3)
- ✅ Taylor test shows O(h²) convergence
- ✅ Optimization converges
- ✅ Physical properties preserved

### Performance
- ✅ ODE solving: 5-20x faster (non-stiff)
- ✅ ODE solving: 50-100x faster (stiff)
- ✅ Full training: 10-50x faster

## Support

For questions or issues with tests:

1. Check TEST_ARCHITECTURE.md for detailed design
2. Look at existing test examples
3. Open an issue on GitHub

## Contributing

When adding new tests:

1. Follow existing test structure
2. Use test utilities for consistency
3. Add both happy path and edge cases
4. Document expected behavior
5. Update this README if adding new test categories
