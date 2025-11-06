"""
Library of standard test problems with known properties

Provides standard MFG problems for testing:
- Gaussian transport (1D/2D)
- Crowd motion
- Ring to ring
- High-dimensional problems
"""
module TestProblems

using MFGnet
using LinearAlgebra
using Random

export gaussian_1d_transport, gaussian_2d_transport, crowd_motion_2d
export ring_to_ring_2d, high_dim_problem
export initialize_single_layer_params, initialize_resnn_params

# ============================================================================
# Parameter Initialization Helpers
# ============================================================================

"""
    initialize_single_layer_params(d, m; R=Float64)

Initialize parameters for SingleLayer potential: (w, (K, b), A, c, z)

# Arguments
- `d`: Spatial dimension
- `m`: Number of hidden units
- `R`: Numeric type (Float64, Float32)
"""
function initialize_single_layer_params(d, m; R=Float64)
    K = R.(0.01 * randn(m, d+1))
    b = R.(0.1 * randn(m))
    w = R.(ones(m) / sqrt(m))
    A = R.(zeros(d+1, d+1))
    c = R.(zeros(d+1))
    z = R.([1.0])

    return (w, (K, b), A, c, z)
end

"""
    initialize_resnn_params(d, m, nt; R=Float64)

Initialize parameters for ResNN potential.

# Arguments
- `d`: Spatial dimension
- `m`: Hidden dimension per layer
- `nt`: Number of time steps in ResNN
- `R`: Numeric type
"""
function initialize_resnn_params(d, m, nt; R=Float64)
    # ResNN uses layers at different time steps
    layers = []
    for _ in 1:nt
        K = R.(0.01 * randn(m, d+1))
        b = R.(0.1 * randn(m))
        push!(layers, (K, b))
    end

    w = R.(ones(m) / sqrt(m))
    A = R.(zeros(d+1, d+1))
    c = R.(zeros(d+1))
    z = R.([1.0])

    return (w, tuple(layers...), A, c, z)
end

# ============================================================================
# 1D Test Problems
# ============================================================================

"""
    gaussian_1d_transport(;nex, R)

Simple 1D Gaussian transport problem.

# Properties
- Analytical solution exists for linear case
- Non-stiff dynamics
- Fast to evaluate (good for quick tests)
- Initial: Gaussian at μ=0, σ=0.3
- Target: Gaussian at μ=1, σ=0.3

# Returns
NamedTuple with:
- `mfg`: MeanFieldGame problem
- `Θ_init`: Initial parameters
- `properties`: Problem properties (stiffness, dimension, etc.)
"""
function gaussian_1d_transport(;nex=100, R=Float64, seed=1234)
    Random.seed!(seed)

    d = 1

    # Initial and target densities (both Gaussian)
    σ0, μ0 = R(0.3), R(0.0)
    σ1, μ1 = R(0.3), R(1.0)

    norm_const_0 = R(1.0 / (σ0 * sqrt(2π)))
    norm_const_1 = R(1.0 / (σ1 * sqrt(2π)))

    rho0(x) = norm_const_0 * exp.(-(sum(x.^2, dims=1) .- μ0^2) / (2*σ0^2))
    rho1(x) = norm_const_1 * exp.(-(sum((x .- μ1).^2, dims=1)) / (2*σ1^2))

    # Sample particles from initial distribution
    X0 = randn(R, d, nex) .* σ0 .+ μ0
    w = ones(R, nex) / nex

    # Create MFG problem
    F = F0()  # No interaction
    G = Gls(rho0, rho1, rho0(X0), rho1(X0), R(1.0))

    # Simple single-layer potential
    Φ = PotentialNN(SingleLayer())

    mfg = MeanFieldGame(F, G, X0, rho0, w;
                       Φ=Φ,
                       α=R.([1.0, 1.0, 1.0, 0.1, 0.1]),
                       tspan=R.([0.0, 1.0]),
                       stepper=RK1Step(),
                       nt=10)

    # Initialize parameters
    m = 20  # Hidden units
    Θ_init = initialize_single_layer_params(d, m; R=R)

    properties = (
        stiffness = :nonstiff,
        dimension = 1,
        nparticles = nex,
        has_interaction = false,
        description = "1D Gaussian transport"
    )

    return (mfg=mfg, Θ_init=Θ_init, properties=properties)
end

"""
    gaussian_2d_transport(;nex, R)

2D Gaussian transport problem.

Similar to 1D but tests higher-dimensional handling.
"""
function gaussian_2d_transport(;nex=200, R=Float64, seed=1234)
    Random.seed!(seed)

    d = 2

    σ0, μ0 = R(0.3), R.(zeros(d))
    σ1, μ1 = R(0.3), R.(ones(d))

    rho0(x) = exp.(-sum(x.^2, dims=1) / (2*σ0^2))
    rho1(x) = exp.(-sum((x .- μ1).^2, dims=1) / (2*σ1^2))

    X0 = randn(R, d, nex) .* σ0
    w = ones(R, nex) / nex

    F = F0()
    G = Gls(rho0, rho1, rho0(X0), rho1(X0), R(1.0))

    Φ = PotentialNN(SingleLayer())
    mfg = MeanFieldGame(F, G, X0, rho0, w;
                       Φ=Φ,
                       α=R.([1.0, 1.0, 1.0, 0.1, 0.1]),
                       tspan=R.([0.0, 1.0]),
                       stepper=RK1Step(),
                       nt=10)

    m = 32
    Θ_init = initialize_single_layer_params(d, m; R=R)

    properties = (
        stiffness = :nonstiff,
        dimension = 2,
        nparticles = nex,
        has_interaction = false,
        description = "2D Gaussian transport"
    )

    return (mfg=mfg, Θ_init=Θ_init, properties=properties)
end

# ============================================================================
# 2D Test Problems
# ============================================================================

"""
    crowd_motion_2d(;nex, R)

2D crowd motion problem.

# Properties
- Initial: Concentrated in center
- Target: Spread out in a region
- Moderate stiffness
- Tests spatial dimension handling
"""
function crowd_motion_2d(;nex=200, R=Float64, seed=1234)
    Random.seed!(seed)

    d = 2

    # Initial: concentrated near origin
    # Target: spread out at (0.5, 0.5)
    rho0(x) = exp.(R(-5.0) * sum(x.^2, dims=1))
    rho1(x) = exp.(R(-0.5) * sum((x .- R.([0.5; 0.5])).^2, dims=1))

    X0 = randn(R, d, nex) .* R(0.2)
    w = ones(R, nex) / nex

    # Can use interaction term if desired
    F = F0()  # No interaction for now
    G = Gls(rho0, rho1, rho0(X0), rho1(X0), R(10.0))

    Φ = PotentialNN(SingleLayer())
    mfg = MeanFieldGame(F, G, X0, rho0, w;
                       Φ=Φ,
                       α=R.([1.0, 1.0, 1.0, 0.1, 0.1]),
                       tspan=R.([0.0, 1.0]),
                       stepper=RK1Step(),
                       nt=10)

    m = 32
    Θ_init = initialize_single_layer_params(d, m; R=R)

    properties = (
        stiffness = :mild,
        dimension = 2,
        nparticles = nex,
        has_interaction = false,
        description = "2D crowd motion"
    )

    return (mfg=mfg, Θ_init=Θ_init, properties=properties)
end

"""
    ring_to_ring_2d(;nex, R)

Ring to ring problem in 2D.

# Properties
- Initial: Particles on ring at radius 0.5
- Target: Particles on ring at radius 1.5
- Tests topological features
- Potentially stiff due to narrow target
"""
function ring_to_ring_2d(;nex=200, R=Float64, seed=1234)
    Random.seed!(seed)

    d = 2

    # Initial: ring at radius 0.5
    # Target: ring at radius 1.5
    r0, r1 = R(0.5), R(1.5)
    width = R(0.1)  # Width of ring

    rho0(x) = exp.(R(-10.0) * (sqrt.(sum(x.^2, dims=1)) .- r0).^2 / width^2)
    rho1(x) = exp.(R(-10.0) * (sqrt.(sum(x.^2, dims=1)) .- r1).^2 / width^2)

    # Sample particles uniformly on initial ring
    θ = range(R(0), R(2π), length=nex+1)[1:end-1]
    X0 = r0 .* R.([cos.(θ)'; sin.(θ)'])
    w = ones(R, nex) / nex

    F = F0()
    G = Gls(rho0, rho1, rho0(X0), rho1(X0), R(10.0))

    Φ = PotentialNN(SingleLayer())
    mfg = MeanFieldGame(F, G, X0, rho0, w;
                       Φ=Φ,
                       α=R.([1.0, 1.0, 1.0, 0.1, 0.1]),
                       tspan=R.([0.0, 1.0]),
                       stepper=RK1Step(),
                       nt=10)

    m = 32
    Θ_init = initialize_single_layer_params(d, m; R=R)

    properties = (
        stiffness = :mild,
        dimension = 2,
        nparticles = nex,
        has_interaction = false,
        description = "2D ring to ring transport"
    )

    return (mfg=mfg, Θ_init=Θ_init, properties=properties)
end

# ============================================================================
# High-Dimensional Test Problems
# ============================================================================

"""
    high_dim_problem(;d, nex, R)

High-dimensional Gaussian transport.

# Properties
- Tests scaling to high dimensions
- Useful for performance benchmarks
"""
function high_dim_problem(;d=10, nex=500, R=Float64, seed=1234)
    Random.seed!(seed)

    σ = R(0.3)
    μ0 = R.(zeros(d))
    μ1 = R.(ones(d))

    rho0(x) = exp.(-sum(x.^2, dims=1) / (2*σ^2))
    rho1(x) = exp.(-sum((x .- μ1).^2, dims=1) / (2*σ^2))

    X0 = randn(R, d, nex) .* σ
    w = ones(R, nex) / nex

    F = F0()
    G = Gls(rho0, rho1, rho0(X0), rho1(X0), R(1.0))

    Φ = PotentialNN(SingleLayer())
    mfg = MeanFieldGame(F, G, X0, rho0, w;
                       Φ=Φ,
                       α=R.([1.0, 1.0, 1.0, 0.1, 0.1]),
                       tspan=R.([0.0, 1.0]),
                       stepper=RK1Step(),
                       nt=10)

    m = 64  # More hidden units for high dimensions
    Θ_init = initialize_single_layer_params(d, m; R=R)

    properties = (
        stiffness = :nonstiff,
        dimension = d,
        nparticles = nex,
        has_interaction = false,
        description = "High-dimensional Gaussian transport (d=$d)"
    )

    return (mfg=mfg, Θ_init=Θ_init, properties=properties)
end

# ============================================================================
# Problem Registry
# ============================================================================

"""
    get_test_problem(name; kwargs...)

Get a test problem by name.

# Available problems
- `:gaussian_1d` - Simple 1D Gaussian transport
- `:gaussian_2d` - 2D Gaussian transport
- `:crowd_2d` - 2D crowd motion
- `:ring_2d` - Ring to ring in 2D
- `:high_dim` - High-dimensional problem
"""
function get_test_problem(name::Symbol; kwargs...)
    problem_dict = Dict(
        :gaussian_1d => gaussian_1d_transport,
        :gaussian_2d => gaussian_2d_transport,
        :crowd_2d => crowd_motion_2d,
        :ring_2d => ring_to_ring_2d,
        :high_dim => high_dim_problem
    )

    if !haskey(problem_dict, name)
        error("Unknown problem: $name. Available: $(keys(problem_dict))")
    end

    return problem_dict[name](; kwargs...)
end

"""
    list_test_problems()

List all available test problems.
"""
function list_test_problems()
    problems = [
        (:gaussian_1d, "1D Gaussian transport (fast, non-stiff)"),
        (:gaussian_2d, "2D Gaussian transport (moderate)"),
        (:crowd_2d, "2D crowd motion (moderate, mildly stiff)"),
        (:ring_2d, "2D ring to ring (topological features)"),
        (:high_dim, "High-dimensional transport (performance testing)")
    ]

    println("Available Test Problems:")
    println("="^70)
    for (name, desc) in problems
        println("  :$name - $desc")
    end
    println("="^70)

    return problems
end

end # module TestProblems
