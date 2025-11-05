# Julia Package Structure

Standard structure and best practices for Julia packages.

## Standard Package Layout

```
MyPackage/
├── src/
│   ├── MyPackage.jl         # Main module file
│   ├── submodule1.jl        # Additional modules
│   └── utils.jl             # Utilities
├── test/
│   ├── runtests.jl          # Main test file
│   ├── Project.toml         # Test dependencies
│   └── test_*.jl            # Test modules
├── docs/
│   ├── make.jl              # Documentation builder
│   ├── Project.toml         # Docs dependencies
│   └── src/
│       ├── index.md         # Landing page
│       └── api.md           # API reference
├── examples/
│   └── example1.jl          # Usage examples
├── benchmark/
│   └── benchmarks.jl        # Performance benchmarks
├── Project.toml             # Package metadata & dependencies
├── README.md                # Package description
├── LICENSE                  # License file
├── .github/
│   └── workflows/
│       └── CI.yml           # GitHub Actions CI
└── .gitignore
```

## Project.toml

Package metadata and dependencies:

```toml
name = "MyPackage"
uuid = "12345678-1234-1234-1234-123456789abc"
authors = ["Your Name <email@example.com>"]
version = "0.1.0"

[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
julia = "1.10"
SomePackage = "2.0, 3"

[extras]
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[targets]
test = ["Test"]
```

### Version Compatibility

```toml
[compat]
julia = "1.10"              # Exactly 1.10
julia = "1.10, 1.11"        # 1.10 or 1.11
julia = "^1.10"             # >= 1.10.0, < 2.0.0
SomePackage = "2"           # >= 2.0.0, < 3.0.0
OtherPackage = "2.3"        # >= 2.3.0, < 3.0.0
ThirdPackage = "2.3.4"      # >= 2.3.4, < 3.0.0
```

## Main Module File (src/MyPackage.jl)

```julia
module MyPackage

# Imports from other packages
using LinearAlgebra
using SparseArrays

# Exports (public API)
export myfunction, MyType

# Include submodules
include("submodule1.jl")
include("utils.jl")

# Main code
struct MyType
    data::Vector{Float64}
end

function myfunction(x::MyType)
    # Implementation
end

end # module
```

### Organizing Large Packages

```julia
module MyPackage

export foo, bar, Baz

# Core functionality
include("core/types.jl")
include("core/operations.jl")

# Submodules
include("optimization/optimize.jl")
include("visualization/plots.jl")

# Optional features (requires package extensions)
include("gpu/cuda_support.jl")

# Utilities
include("utils/helpers.jl")

end # module
```

## Package Extensions (Julia 1.9+)

Conditional code that loads only if dependencies available:

### Project.toml

```toml
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[weakdeps]
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"

[extensions]
MyPackageCUDAExt = "CUDA"
```

### ext/MyPackageCUDAExt.jl

```julia
module MyPackageCUDAExt

using MyPackage
using CUDA

# GPU-specific implementations
function MyPackage.myfunction(x::CuArray)
    # GPU implementation
end

end
```

Users get GPU support automatically when they `using CUDA`.

## Documentation

### Docstrings

```julia
"""
    myfunction(x::AbstractArray; tol=1e-6)

Compute something with `x`.

# Arguments
- `x::AbstractArray`: Input array
- `tol::Float64`: Tolerance (default: 1e-6)

# Returns
- `result::Float64`: The computed result

# Examples
```jldoctest
julia> myfunction([1, 2, 3])
6.0
```

# See Also
- [`otherfunction`](@ref)
- [`MyType`](@ref)
"""
function myfunction(x::AbstractArray; tol=1e-6)
    # Implementation
end
```

### Documenter.jl Setup

**docs/make.jl**:

```julia
using Documenter
using MyPackage

makedocs(
    sitename = "MyPackage.jl",
    format = Documenter.HTML(),
    modules = [MyPackage],
    pages = [
        "Home" => "index.md",
        "API Reference" => "api.md",
        "Examples" => "examples.md",
    ]
)

deploydocs(
    repo = "github.com/username/MyPackage.jl.git",
)
```

**docs/src/index.md**:

```markdown
# MyPackage.jl

Welcome to MyPackage.jl documentation!

## Installation

\```julia
using Pkg
Pkg.add("MyPackage")
\```

## Quick Start

\```julia
using MyPackage
result = myfunction(data)
\```
```

**docs/src/api.md**:

```markdown
# API Reference

## Types

\```@docs
MyType
\```

## Functions

\```@docs
myfunction
otherfunction
\```
```

Build docs locally:
```bash
julia --project=docs docs/make.jl
```

## Registering Packages

### General Registry

1. Ensure tests pass
2. Add JuliaRegistrator bot to your repo
3. Comment `@JuliaRegistrator register` on commit

### Local Registry

```julia
using LocalRegistry

# Create local registry
create_registry("MyRegistry", "path/to/registry")

# Register package
register("MyPackage", registry="MyRegistry")
```

## Versioning

Follow Semantic Versioning (SemVer):
- **MAJOR**: Incompatible API changes
- **MINOR**: Add functionality (backwards-compatible)
- **PATCH**: Bug fixes (backwards-compatible)

Examples:
- `0.1.0` → `0.1.1`: Bug fix
- `0.1.1` → `0.2.0`: New feature
- `0.2.0` → `1.0.0`: Stable API
- `1.0.0` → `2.0.0`: Breaking change

## Best Practices

### 1. Minimal Dependencies

Only depend on packages you actually use:

```toml
# GOOD
[deps]
LinearAlgebra = "..."  # Only stdlib

# BAD (too many dependencies)
[deps]
LinearAlgebra = "..."
Plots = "..."  # Heavy dependency for one plot
...
```

Use package extensions for optional features.

### 2. Type Stability

Export type-stable functions:

```julia
# Check exports are type-stable
using JET

@test_opt myfunction(x)
```

### 3. Documentation

- Document all exported functions
- Include examples in docstrings
- Keep README up-to-date
- Use Documenter.jl for comprehensive docs

### 4. Testing

- Test all public API
- Test on multiple Julia versions
- Include coverage reporting
- Use CI (GitHub Actions)

### 5. Performance

- Include benchmarks in `benchmark/`
- Use BenchmarkTools.jl
- Track performance regressions

```julia
# benchmark/benchmarks.jl
using BenchmarkTools
using MyPackage

suite = BenchmarkGroup()

suite["myfunction"] = @benchmarkable myfunction($x) setup=(x=rand(1000))

results = run(suite)
```

### 6. Examples

Provide runnable examples:

```julia
# examples/basic_usage.jl
using MyPackage

# Generate some data
x = rand(100)

# Process it
result = myfunction(x)

# Visualize (if plotting available)
# using Plots
# plot(result)
```

## Package Templates

Use PkgTemplates.jl for standardized structure:

```julia
using PkgTemplates

t = Template(;
    user="yourusername",
    authors="Your Name",
    julia=v"1.10",
    plugins=[
        License(; name="MIT"),
        Git(; manifest=true, ssh=true),
        GitHubActions(; x64=true, coverage=true),
        Codecov(),
        Documenter{GitHubActions}(),
        Develop(),
    ],
)

t("MyNewPackage")
```

Generates complete package structure with CI/CD.

## Monorepo Structure (Multiple Packages)

For related packages:

```
MonoRepo/
├── PackageA/
│   ├── src/
│   ├── test/
│   └── Project.toml
├── PackageB/
│   ├── src/
│   ├── test/
│   └── Project.toml
└── Project.toml  # Workspace root
```

**Root Project.toml**:

```toml
[deps]
PackageA = {path = "PackageA"}
PackageB = {path = "PackageB"}
```

## Package Development Workflow

### Setting Up Dev Environment

```julia
# Clone and develop
] dev path/to/MyPackage

# Or from URL
] dev https://github.com/user/MyPackage.jl

# Use Revise.jl for auto-reloading
using Revise
using MyPackage

# Edit code, changes automatically reload
```

### Making Changes

1. Create feature branch
   ```bash
   git checkout -b feature/new-feature
   ```

2. Make changes

3. Test locally
   ```julia
   ] test MyPackage
   ```

4. Commit and push
   ```bash
   git add .
   git commit -m "Add new feature"
   git push origin feature/new-feature
   ```

5. Create pull request

6. CI runs tests

7. Merge when approved

### Releasing New Version

1. Update version in `Project.toml`
2. Update `CHANGELOG.md`
3. Tag release
   ```bash
   git tag -a v0.2.0 -m "Release v0.2.0"
   git push --tags
   ```
4. Register with `@JuliaRegistrator register`

## Common Project.toml Patterns

### Conditional Test Dependencies

```toml
[extras]
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"

[targets]
test = ["Test", "CUDA"]
```

### Development Dependencies

```toml
[deps]
# Production dependencies only

[extras]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
```

## README Template

```markdown
# MyPackage.jl

[![Build Status](badge-url)](ci-url)
[![Coverage](badge-url)](coverage-url)
[![Documentation](badge-url)](docs-url)

Brief description of package purpose.

## Installation

\```julia
using Pkg
Pkg.add("MyPackage")
\```

## Quick Start

\```julia
using MyPackage

# Basic example
x = rand(100)
result = myfunction(x)
\```

## Features

- Feature 1
- Feature 2
- Feature 3

## Documentation

See [documentation](docs-url) for detailed usage.

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License. See [LICENSE](LICENSE) for details.
```

## Summary

**Essential files:**
- `src/MyPackage.jl`: Main module
- `test/runtests.jl`: Tests
- `Project.toml`: Metadata & dependencies
- `README.md`: Package description
- `LICENSE`: License

**Best practices:**
- Minimal dependencies
- Comprehensive tests (CI/CD)
- Documentation (docstrings + Documenter.jl)
- Semantic versioning
- Examples and benchmarks

**Development:**
- Use `] dev` for development
- Use Revise.jl for auto-reloading
- Use PkgTemplates.jl for standardization
- Register in General registry when stable
