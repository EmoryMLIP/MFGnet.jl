# Julia Testing Practices

Comprehensive guide to testing, coverage, and CI/CD for Julia packages.

## Basic Testing with Test.jl

```julia
using Test

@test 2 + 2 == 4
@test_throws DivideError 1 ÷ 0
@test sin(π) ≈ 0 atol=1e-10
```

### Test Sets

```julia
using Test

@testset "Arithmetic" begin
    @test 2 + 2 == 4
    @test 3 * 3 == 9

    @testset "Division" begin
        @test 6 ÷ 2 == 3
        @test_throws DivideError 1 ÷ 0
    end
end
```

Output shows passed/failed tests organized by set.

## Package Testing Structure

Standard test structure:
```
MyPackage/
├── src/
│   └── MyPackage.jl
├── test/
│   ├── runtests.jl       # Main test file
│   ├── test_module1.jl   # Tests for module1
│   ├── test_module2.jl   # Tests for module2
│   └── test_utils.jl     # Test utilities
├── Project.toml          # Package dependencies
└── test/Project.toml     # Test-only dependencies
```

### test/runtests.jl

```julia
using MyPackage
using Test

@testset "MyPackage.jl" begin
    include("test_module1.jl")
    include("test_module2.jl")
end
```

### test/Project.toml

```toml
[deps]
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
MyPackage = "..."  # Your package UUID

[extras]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
```

## Running Tests

```julia
# From Julia REPL
] test MyPackage

# From command line
julia --project=. -e 'using Pkg; Pkg.test()'

# Run specific test file
julia --project=. test/test_module1.jl

# With coverage
julia --project=. --code-coverage=user -e 'using Pkg; Pkg.test()'
```

## Test Macros

### @test

```julia
@test expression
@test 2 + 2 == 4
@test maximum([1,2,3]) == 3
```

### @test_throws

```julia
@test_throws ExceptionType expression

@test_throws DomainError sqrt(-1)
@test_throws ArgumentError foo(invalid_arg)
@test_throws "error message substring" bar()
```

### Approximate Equality

```julia
@test a ≈ b                    # Default tolerance
@test a ≈ b atol=1e-6          # Absolute tolerance
@test a ≈ b rtol=1e-6          # Relative tolerance
@test a ≈ b atol=1e-6 rtol=1e-6  # Both

# For arrays
@test all(A .≈ B)
```

### @test_logs

```julia
@test_logs (:warn, "warning message") my_function()
@test_logs (:info, r"pattern") my_function()

# Match multiple logs
@test_logs (:info, "start") (:info, "end") my_function()
```

### @test_skip and @test_broken

```julia
# Skip test (known issue, won't run)
@test_skip currently_broken_function()

# Broken test (runs but failure expected)
@test_broken sqrt(-1) == 0im  # Currently wrong, but we know
```

### @inferred

```julia
using Test

# Test type stability
@inferred my_function(args...)

# If type-unstable, this test fails
function unstable(x)
    x > 0 ? x : 0.0  # Returns Int or Float64
end

@test_throws ErrorException @inferred unstable(5)
```

## Property-Based Testing

```julia
using Test

# Test properties across random inputs
@testset "Properties" begin
    for _ in 1:1000
        x = rand()
        y = rand()

        # Property: addition is commutative
        @test x + y ≈ y + x

        # Property: abs is idempotent
        @test abs(abs(x)) ≈ abs(x)
    end
end
```

## Reference Testing

Compare against reference outputs:

```julia
using Test, ReferenceTests

@testset "Reference tests" begin
    # Compare against reference file
    @test_reference "references/output1.txt" my_function(input1)

    # Update references with UPDATE_REFERENCES=true
    # julia --project=. test/runtests.jl
end
```

## Testing Best Practices

### 1. Test Edge Cases

```julia
@testset "Edge cases" begin
    @test my_divide(6, 2) == 3           # Normal case
    @test my_divide(0, 5) == 0           # Zero numerator
    @test_throws DivideError my_divide(5, 0)  # Zero denominator
    @test my_divide(-6, 2) == -3         # Negative numbers
    @test isnan(my_divide(0, 0))         # Undefined
end
```

### 2. Test Type Stability

```julia
@testset "Type stability" begin
    @test @inferred my_function(1, 2.0)
    @test @inferred my_function(Float32(1), Float32(2))
end
```

### 3. Test Generic Code

```julia
@testset "Numeric types" begin
    for T in (Float32, Float64, BigFloat)
        x = T(2)
        @test my_function(x) isa T
        @test my_function(x) ≈ T(4)
    end
end
```

### 4. Test GPU Code

```julia
using CUDA

@testset "GPU" begin
    if CUDA.functional()
        x_cpu = rand(100)
        x_gpu = CuArray(x_cpu)

        result_cpu = my_function(x_cpu)
        result_gpu = Array(my_function(x_gpu))

        @test result_cpu ≈ result_gpu
    else
        @test_skip "No GPU available"
    end
end
```

### 5. Separate Fast and Slow Tests

```julia
# test/runtests.jl
using Test

@testset "MyPackage" begin
    include("fast_tests.jl")

    # Slow tests only if requested
    if get(ENV, "FULL_TESTS", "false") == "true"
        include("slow_tests.jl")
    end
end
```

Run with: `FULL_TESTS=true julia --project=. test/runtests.jl`

## Test Coverage

### Generate Coverage

```julia
julia --project=. --code-coverage=user test/runtests.jl
```

Generates `.cov` files showing line-by-line coverage.

### View Coverage Locally

```julia
using Pkg
Pkg.add("Coverage")

using Coverage

# Process coverage files
coverage = process_folder()

# Show coverage percentage
covered, total = get_summary(coverage)
println("Coverage: ", round(100 * covered / total, digits=2), "%")

# Generate HTML report (requires lcov)
LCOV.writefile("coverage.info", coverage)
# Then: genhtml coverage.info -o coverage_html
```

### Coverage on CI (Codecov)

Add to GitHub Actions:

```yaml
- uses: julia-actions/julia-processcoverage@v1
- uses: codecov/codecov-action@v3
  with:
    files: lcov.info
```

## Continuous Integration

### GitHub Actions

`.github/workflows/CI.yml`:

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        julia-version: ['1.10', 'nightly']
    steps:
      - uses: actions/checkout@v4
      - uses: julia-actions/setup-julia@v1
        with:
          version: ${{ matrix.julia-version }}
      - uses: julia-actions/cache@v1
      - uses: julia-actions/julia-buildpkg@v1
      - uses: julia-actions/julia-runtest@v1
      - uses: julia-actions/julia-processcoverage@v1
      - uses: codecov/codecov-action@v3
        with:
          files: lcov.info
```

### GPU Testing on CI

```yaml
gpu-test:
  runs-on: ubuntu-latest
  container:
    image: julia:1.10
    options: --gpus all
  steps:
    - uses: actions/checkout@v4
    - name: Run GPU tests
      run: |
        julia --project=. -e 'using Pkg; Pkg.test()'
```

## Benchmarking in Tests

```julia
using Test, BenchmarkTools

@testset "Performance" begin
    # Ensure no regressions
    x = rand(1000)

    result = @benchmark my_function($x)

    # Check performance bounds
    @test median(result).time < 1e6  # < 1ms
    @test result.allocs == 0         # No allocations
end
```

## Test Utilities

### Fixtures

```julia
# test/test_utils.jl
function create_test_data()
    x = rand(100)
    y = rand(100)
    return x, y
end

# test/runtests.jl
include("test_utils.jl")

@testset "Tests using fixtures" begin
    x, y = create_test_data()
    @test length(x) == length(y)
end
```

### Temporary Files

```julia
using Test

@testset "File I/O" begin
    mktempdir() do dir
        filepath = joinpath(dir, "test.txt")

        # Write and read
        write_data(filepath, data)
        result = read_data(filepath)

        @test result == data
        # Cleanup automatic when mktempdir block exits
    end
end
```

## Debugging Failing Tests

### Show Values

```julia
@test maximum([1,2,3]) == 4
# Test Failed at REPL[1]:1
#   Expression: maximum([1, 2, 3]) == 4
#    Evaluated: 3 == 4
```

### Custom Test Messages

```julia
@test my_function(x) ≈ expected "my_function failed for x=$x"
```

### Break on Test Failure

```julia
# Run with --check-bounds=no to get line numbers
# Add Infiltrator.jl for debugging

using Infiltrator

@testset "Debug" begin
    result = my_function(x)
    @infiltrate
    @test result ≈ expected
end
```

## Testing Checklist

Before releasing:
- [ ] All tests pass on all Julia versions
- [ ] Test coverage > 80% (aim for > 90%)
- [ ] Edge cases tested
- [ ] Type stability tested
- [ ] GPU code tested (if applicable)
- [ ] Documentation examples tested (doctests)
- [ ] CI passes on all platforms
- [ ] Performance benchmarks stable
- [ ] No test warnings or deprecations

## Common Testing Pitfalls

1. **Global state**: Tests should be independent
2. **Random seeds**: Use `Random.seed!()` for reproducibility
3. **Floating-point comparison**: Use `≈` not `==`
4. **Over-testing implementation**: Test behavior, not internal details
5. **Slow tests**: Separate slow tests from fast unit tests
6. **Untested error paths**: Test error handling
7. **Missing type tests**: Test with multiple numeric types

## Documentation Tests (Doctests)

```julia
\"\"\"
    add_one(x)

Add one to x.

# Examples
```jldoctest
julia> add_one(1)
2

julia> add_one(2.0)
3.0
```
\"\"\"
function add_one(x)
    return x + 1
end
```

Run with:
```julia
using Documenter
doctest(MyPackage)
```

## Summary

**Test structure:**
- Organize tests in `test/` directory
- Use `@testset` for organization
- Separate test dependencies in `test/Project.toml`

**What to test:**
- Correctness (edge cases, error conditions)
- Type stability
- Generic code (multiple types)
- GPU compatibility (if applicable)
- Performance (no regressions)

**CI/CD:**
- Test on multiple Julia versions
- Test on multiple OSes
- Measure and track coverage
- Run benchmarks to catch regressions

**Best practices:**
- Keep tests fast (separate slow tests)
- Make tests reproducible
- Test behavior, not implementation
- Aim for high coverage (>80%)
