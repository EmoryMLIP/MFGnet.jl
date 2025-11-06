"""
DifferentialEquations.jl Interface for MFGnet

Provides compatibility layer between MFGnet's custom ODE solvers and DifferentialEquations.jl
ecosystem, enabling:
- Adaptive time-stepping
- Wide variety of ODE solvers (Tsit5, RK4, implicit methods, etc.)
- Efficient gradient computation via DiffEqSensitivity.jl
- Callbacks for monitoring and control

# Exports
- `AdaptiveConfig`, `RK4Config`, `RK1Config` - ODE solver configurations
- `solve_mfg_ode` - Main solve function using DifferentialEquations.jl
- `mfg_ode!` - ODE right-hand side in DifferentialEquations.jl format
"""

using DifferentialEquations
using LinearAlgebra

export AdaptiveConfig, RK4Config, RK1Config
export solve_mfg_ode, mfg_ode!
export extract_final_state, extract_trajectory

#=============================================================================
Solver Configuration Structs
=============================================================================#

"""
    AdaptiveConfig(;kwargs...)

Adaptive ODE solver configuration using Tsit5 (default) or other algorithms

# Keyword Arguments
- `alg`: ODE algorithm (default: Tsit5())
- `reltol::Real`: Relative tolerance (default: 1e-6)
- `abstol::Real`: Absolute tolerance (default: 1e-8)
- `sensealg`: Sensitivity algorithm for gradients (default: nothing, uses DifferentialEquations.jl default)
- `save_everystep::Bool`: Save full trajectory (default: false)
- `saveat`: Specific times to save (default: nothing)
- `callback`: DifferentialEquations.jl callback (default: nothing)
- `maxiters::Int`: Maximum steps (default: 1e5)
"""
struct AdaptiveConfig{ALG,SENS,CB}
    alg::ALG
    reltol::Float64
    abstol::Float64
    sensealg::SENS
    save_everystep::Bool
    saveat::Union{Nothing,AbstractVector,Real}
    callback::CB
    maxiters::Int
end

function AdaptiveConfig(;
    alg=Tsit5(),
    reltol=1e-6,
    abstol=1e-8,
    sensealg=nothing,
    save_everystep=false,
    saveat=nothing,
    callback=nothing,
    maxiters=Int(1e5))

    return AdaptiveConfig(alg, reltol, abstol, sensealg, save_everystep, saveat, callback, maxiters)
end

"""
    RK4Config(nt, tspan; kwargs...)

Fixed-step RK4 configuration matching legacy behavior

# Arguments
- `nt::Int`: Number of time steps
- `tspan::Vector`: Time interval [t0, tf]

# Keyword Arguments
- `sensealg`: Sensitivity algorithm (default: nothing, uses DifferentialEquations.jl default)
- `save_everystep::Bool`: Save full trajectory (default: false)
- `callback`: DifferentialEquations.jl callback (default: nothing)
"""
struct RK4Config{SENS,CB}
    nt::Int
    dt::Float64
    tspan::Vector{Float64}
    sensealg::SENS
    save_everystep::Bool
    callback::CB
end

function RK4Config(nt, tspan;
    sensealg=nothing,
    save_everystep=false,
    callback=nothing)

    dt = (tspan[2] - tspan[1]) / nt
    return RK4Config(nt, dt, tspan, sensealg, save_everystep, callback)
end

"""
    RK1Config(nt, tspan; kwargs...)

Fixed-step Forward Euler configuration matching legacy behavior

# Arguments
- `nt::Int`: Number of time steps
- `tspan::Vector`: Time interval [t0, tf]

# Keyword Arguments
- `sensealg`: Sensitivity algorithm (default: nothing, uses DifferentialEquations.jl default)
- `save_everystep::Bool`: Save full trajectory (default: false)
- `callback`: DifferentialEquations.jl callback (default: nothing)
"""
struct RK1Config{SENS,CB}
    nt::Int
    dt::Float64
    tspan::Vector{Float64}
    sensealg::SENS
    save_everystep::Bool
    callback::CB
end

function RK1Config(nt, tspan;
    sensealg=nothing,
    save_everystep=false,
    callback=nothing)

    dt = (tspan[2] - tspan[1]) / nt
    return RK1Config(nt, dt, tspan, sensealg, save_everystep, callback)
end

#=============================================================================
ODE Problem Definition
=============================================================================#

"""
    mfg_ode!(du, u, p, t)

MFG characteristic ODE in DifferentialEquations.jl format (in-place)

# State Vector
u = [x; l; v; f; hj] reshaped to vector form

# Parameters
p = (J, Θ, d, nex) where:
- J::MeanFieldGame - Problem structure
- Θ - Neural network parameters
- d::Int - Spatial dimension
- nex::Int - Number of examples

# Dynamics
See odefun documentation for equations
"""
function mfg_ode!(du, u, p, t)
    J, Θ, d, nex = p
    R = eltype(u)

    # Reshape vector u back to matrix form (d+4) × nex
    U = reshape(u, (d+4, nex))

    # Augment position with time
    XT = [U[1:d,:]; fill(R(t), 1, nex)]

    # Evaluate potential and derivatives
    # NOTE: Phi computation required for side effect - populates internal caches
    # used by getGradPotential and getTraceHess
    Phi = J.Φ(XT, Θ)
    gradPhi = getGradPotential(J.Φ, XT, Θ)
    trH = getTraceHess(J.Φ, XT, Θ)

    # Compute time derivatives (same as odefun)
    dx = -(1/J.α[1]) * gradPhi[1:d,:]
    dl = -(1/J.α[1]) * trH
    dv = 0.5 .* sum(dx.^2, dims=1)
    df = reshape(J.F(U, R(t)), 1, nex)
    hj = abs.(-reshape(gradPhi[end,:], 1, :) -
               reshape(J.α[2] .* getDeltaF(J.F, U, R(t)), 1, :) +
               J.α[1] .* dv)

    # Assemble derivative (matrix form)
    dU = [dx; reshape(dl, 1, nex); dv; df; hj]

    # Reshape to vector and write in-place
    du .= vec(dU)

    return nothing
end

#=============================================================================
Solve Functions
=============================================================================#

"""
    solve_mfg_ode(J::MeanFieldGame, Θ, config::AdaptiveConfig)

Solve MFG characteristic ODE using adaptive DifferentialEquations.jl solver

# Returns
- `sol::ODESolution` - Solution object from DifferentialEquations.jl
  - Access final state: `sol[end]` or `sol.u[end]`
  - Access at specific time: `sol(t)`
  - Full trajectory: `sol.u` (if save_everystep=true)
"""
function solve_mfg_ode(J::MeanFieldGame{R}, Θ, config::AdaptiveConfig) where R
    (d, nex) = size(J.X0)

    # Initialize state
    U0 = [J.X0; zeros(R, 4, nex)]
    u0 = vec(U0)  # Flatten to vector

    # Package parameters
    p = (J, Θ, d, nex)

    # Create ODE problem
    tspan = (R(J.tspan[1]), R(J.tspan[2]))
    prob = ODEProblem(mfg_ode!, u0, tspan, p)

    # Solve with adaptive method
    solve_kwargs = Dict{Symbol,Any}(
        :reltol => config.reltol,
        :abstol => config.abstol,
        :save_everystep => config.save_everystep,
        :maxiters => config.maxiters
    )

    # Add optional arguments if they're not nothing
    if !isnothing(config.saveat)
        solve_kwargs[:saveat] = config.saveat
    end
    if !isnothing(config.callback)
        solve_kwargs[:callback] = config.callback
    end
    if !isnothing(config.sensealg)
        solve_kwargs[:sensealg] = config.sensealg
    end

    sol = solve(prob, config.alg; solve_kwargs...)

    # Check solution status
    if sol.retcode != :Success
        @warn "ODE solver did not converge successfully" retcode=sol.retcode
    end

    return sol
end

"""
    solve_mfg_ode(J::MeanFieldGame, Θ, config::RK4Config)

Solve MFG ODE using fixed-step RK4 (matches legacy behavior)

# Returns
- `sol::ODESolution` - Solution object
"""
function solve_mfg_ode(J::MeanFieldGame{R}, Θ, config::RK4Config) where R
    (d, nex) = size(J.X0)

    # Initialize state
    U0 = [J.X0; zeros(R, 4, nex)]
    u0 = vec(U0)

    # Package parameters
    p = (J, Θ, d, nex)

    # Create ODE problem
    tspan = (R(J.tspan[1]), R(J.tspan[2]))
    prob = ODEProblem(mfg_ode!, u0, tspan, p)

    # Solve with fixed-step RK4
    solve_kwargs = Dict{Symbol,Any}(
        :dt => config.dt,
        :adaptive => false,
        :save_everystep => config.save_everystep
    )

    # Add optional arguments if they're not nothing
    if !isnothing(config.callback)
        solve_kwargs[:callback] = config.callback
    end
    if !isnothing(config.sensealg)
        solve_kwargs[:sensealg] = config.sensealg
    end

    sol = solve(prob, RK4(); solve_kwargs...)

    return sol
end

"""
    solve_mfg_ode(J::MeanFieldGame, Θ, config::RK1Config)

Solve MFG ODE using fixed-step Forward Euler (matches legacy behavior)

# Returns
- `sol::ODESolution` - Solution object
"""
function solve_mfg_ode(J::MeanFieldGame{R}, Θ, config::RK1Config) where R
    (d, nex) = size(J.X0)

    # Initialize state
    U0 = [J.X0; zeros(R, 4, nex)]
    u0 = vec(U0)

    # Package parameters
    p = (J, Θ, d, nex)

    # Create ODE problem
    tspan = (R(J.tspan[1]), R(J.tspan[2]))
    prob = ODEProblem(mfg_ode!, u0, tspan, p)

    # Solve with fixed-step Euler
    solve_kwargs = Dict{Symbol,Any}(
        :dt => config.dt,
        :adaptive => false,
        :save_everystep => config.save_everystep
    )

    # Add optional arguments if they're not nothing
    if !isnothing(config.callback)
        solve_kwargs[:callback] = config.callback
    end
    if !isnothing(config.sensealg)
        solve_kwargs[:sensealg] = config.sensealg
    end

    sol = solve(prob, Euler(); solve_kwargs...)

    return sol
end

#=============================================================================
Utility Functions
=============================================================================#

"""
    extract_final_state(sol::ODESolution, d::Int, nex::Int)

Extract final state from ODESolution and reshape to matrix form

# Returns
- `UN::Matrix` - Final state matrix (d+4) × nex
"""
function extract_final_state(sol, d::Int, nex::Int)
    u_final = sol[end]
    UN = reshape(u_final, (d+4, nex))
    return UN
end

"""
    extract_trajectory(sol::ODESolution, d::Int, nex::Int)

Extract full trajectory from ODESolution and reshape to 3D array

# Returns
- `U_traj::Array{R,3}` - Trajectory array (d+4) × nex × (nt+1)
"""
function extract_trajectory(sol, d::Int, nex::Int)
    nt = length(sol.u) - 1
    R = eltype(sol.u[1])
    U_traj = zeros(R, d+4, nex, nt+1)

    for k in 1:(nt+1)
        U_traj[:, :, k] = reshape(sol.u[k], (d+4, nex))
    end

    return U_traj
end

