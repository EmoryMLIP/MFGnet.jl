#!/usr/bin/env julia
# Create a new Julia package with proper structure, tests, and CI/CD

using PkgTemplates

"""
    create_package(name::String; kwargs...)

Create a new Julia package with best-practice structure.

# Arguments
- `name`: Package name (e.g., "MyPackage")
- `dir`: Directory to create package in (default: current directory)
- `user`: GitHub username (optional)
- `license`: License type (default: "MIT")
- `julia_version`: Minimum Julia version (default: v"1.10")
- `gpu`: Include GPU testing setup (default: false)
"""
function create_package(
    name::String;
    dir::String = pwd(),
    user::String = "",
    license::String = "MIT",
    julia_version::VersionNumber = v"1.10",
    gpu::Bool = false
)

    plugins = [
        License(; name=license),
        Git(; manifest=true, ssh=true),
        GitHubActions(;
            extra_versions=[julia_version, "nightly"]
        ),
        Codecov(),
        Documenter{GitHubActions}(),
        Develop(),
    ]

    if gpu
        # Add GPU testing configuration
        push!(plugins,
            GitHubActions(;
                linux=true,
                osx=false,
                windows=false,
                extra_versions=[julia_version],
                coverage=true
            )
        )
    end

    t = Template(;
        user=user,
        dir=dir,
        julia=julia_version,
        plugins=plugins,
    )

    # Generate the package
    t(name)

    println("✅ Package '$name' created successfully!")
    println("📁 Location: $(joinpath(dir, name))")
    println("")
    println("Next steps:")
    println("  1. cd $(name)")
    println("  2. julia --project=.")
    println("  3. Add your code to src/$(name).jl")
    println("  4. Add tests to test/runtests.jl")
    println("  5. Run tests: ] test")

    return joinpath(dir, name)
end

# Command-line interface
if !isempty(ARGS)
    package_name = ARGS[1]

    # Parse optional arguments
    kwargs = Dict{Symbol,Any}()
    for i in 2:length(ARGS)
        if startswith(ARGS[i], "--dir=")
            kwargs[:dir] = split(ARGS[i], "=")[2]
        elseif startswith(ARGS[i], "--user=")
            kwargs[:user] = split(ARGS[i], "=")[2]
        elseif startswith(ARGS[i], "--license=")
            kwargs[:license] = split(ARGS[i], "=")[2]
        elseif ARGS[i] == "--gpu"
            kwargs[:gpu] = true
        end
    end

    create_package(package_name; kwargs...)
else
    println("Usage: julia create_package.jl <PackageName> [options]")
    println("")
    println("Options:")
    println("  --dir=PATH          Directory to create package in")
    println("  --user=USERNAME     GitHub username")
    println("  --license=LICENSE   License type (default: MIT)")
    println("  --gpu               Include GPU testing setup")
    println("")
    println("Example:")
    println("  julia create_package.jl MyPackage --user=myusername --gpu")
end
