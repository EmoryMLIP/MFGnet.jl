export MeanFieldGame
"""
    MeanFieldGame{R<:Real}

Mean Field Game optimization problem solver

# Objective
min_Θ α₁∫L(x,-∇Φ)dx + α₂∫F(ρ)dt + α₃G(ρ(T)) + α₄∫|HJ residual|dt + α₅|HJ terminal|

subject to: ∂ₜu = odefun(u,Θ,t), u(0) = [x₀; 0; 0; 0; 0]

# Fields
- `F` - interaction term functional or array
- `G` - terminal cost functional or array
- `X0::AbstractArray{R}` - initial particle positions (training points)
- `rho0` - initial density function ρ₀(x)
- `w::Vector{R}` - quadrature weights for Monte Carlo integration
- `rho0x::Vector{R}` - precomputed ρ₀(X0) for efficiency
- `Φ` - potential function approximator (e.g., PotentialNN)
- `α::Vector{R}` - penalty weights [α₁,α₂,α₃,α₄,α₅] for objective terms
- `stepper` - time integration scheme (e.g., RK1Step)
- `tspan::Vector{R}` - time interval [t₀, T]
- `nt` - number of time steps for ODE integration
- `UN` - state [X; log(det); costL; costF; costHJ] after forward solve
- `cs` - cost components [costL, costF, costG, costHJ, costHJfinal]
"""
mutable struct MeanFieldGame{R}
    F # function or array for interaction term
    G # function or array for terminal cost
    X0::AbstractArray{R} # training points
    rho0 # function to compute initial density rho0
    w::AbstractVector{R} # quadrature weights for X0
    rho0x::AbstractVector{R} # rho0(X0), stored for efficiency
    Φ # machine learning model for the potential
    α::Vector{R}  # vector containing penalties for objective functions [L, HJ]
    stepper # time integrator
    tspan::Vector{R} # time interval
    nt # number of time steps in ODE solve
    UN  # [X0,ldet,costL,costF,costHJ]
    cs  # [costL, costF, costG, costHJ,costHJfinal]
end

MeanFieldGame(F,G,X0::AbstractArray{R},rho0,w;
                Φ=PotentialNN(),α=ones(R,5),
                rho0x=rho0(X0),stepper=RK1Step(),tspan=R.([0.; 1.]),nt=2) where R <: Real =
            MeanFieldGame(F,G,X0,rho0,w,rho0x,Φ,α,stepper,tspan,nt,[],zeros(R,5))

"""
Pad X0 with zeros to initialize running costs
"""
function initUN(J,X0::AbstractArray{R}) where R <: Real
    J.UN = [X0; typeof(X0)(zeros(4,size(X0,2)))]
    return J.UN
end

"""
    (J::MeanFieldGame)(Θ; use_diffeq=false, diffeq_config=nothing)

Evaluate MFG objective function

# Arguments
- `Θ`: Neural network parameters

# Keyword Arguments
- `use_diffeq::Bool`: Use DifferentialEquations.jl (default: false, uses legacy solver)
- `diffeq_config`: Configuration for DiffEq solver (AdaptiveConfig, RK4Config, or RK1Config)
                    If nothing, creates default RK4Config matching legacy behavior

# Returns
- `Jc::Real`: Objective value = α₁·costL + α₂·costF + α₃·costG + α₄·costHJ + α₅·costHJf
"""
function (J::MeanFieldGame{R})(Θ; use_diffeq::Bool=false, diffeq_config=nothing) where R <: Real
    (d,nex) = size(J.X0)

    if use_diffeq
        # Use DifferentialEquations.jl path
        if isnothing(diffeq_config)
            # Default: match legacy RK4 behavior
            diffeq_config = RK4Config(J.nt, J.tspan)
        end

        # Solve ODE
        sol = solve_mfg_ode(J, Θ, diffeq_config)
        UN = extract_final_state(sol, d, nex)
        J.UN = UN
    else
        # Legacy path: use custom ODE solver
        h = (J.tspan[2]-J.tspan[1])/J.nt
        UN = initUN(J,J.X0)

        tk = J.tspan[1]
        for k=1:J.nt
            UN = step(J.stepper,odefun,J,UN,Θ,tk,tk+h)
            tk +=h
        end
        J.UN = UN
    end

    # Compute costs (same for both paths)
    costL = dot(vec(UN[end-2,:]),J.w)
    costF = dot(vec(UN[end-1,:]),J.w)
    costG = dot(J.G(UN),J.w)

    # Compute HJB penalty
    costHJ = dot(vec(UN[end,:]),J.w)
    phi1 = vec(J.Φ([UN[1:d,:]; fill(R(1.0),1,size(J.X0,2))],Θ))
    costHJf = dot(abs.(phi1 - J.α[3].*vec(getDeltaG(J.G,UN))),J.w)

    cs = [costL, costF, costG, costHJ, costHJf]
    Jc = dot(J.α,cs)

    # Store intermediate results for plotting and printing
    J.cs = cs .* J.α
    return Jc
end

# Backward compatibility: Original method without keyword arguments
function (J::MeanFieldGame{R})(Θ::Tuple) where R <: Real
    return J(Θ; use_diffeq=false, diffeq_config=nothing)
end
