export Gcomb, Gls, Gkl, Gls2, Gpref, getDeltaG
"""
    Gcomb

Combination of multiple terminal cost functionals

# Formula
G(U) = ∑ᵢ Gᵢ(U)

# Fields
- `Gs::Vector` - array of terminal cost functionals to sum
"""
mutable struct Gcomb
    Gs::Array
end

function (G::Gcomb)(U)
    return sum(g(U) for g in G.Gs)
end
function Base.show(io::IO, G::Gcomb)
  print(io, G.Gs[1])
  for k=2:length(G.Gs)
      print(io, " + $(G.Gs[k])")
  end
end

"""
    Gls

Least-squares terminal cost functional

Penalizes L² distance between terminal density ρ(T) and target ρ₁

# Formula
G(U) = μ/2 ∫(ρ(x,T) - ρ₁(x))² dx

# Fields
- `rho0` - initial density function ρ₀
- `rho1` - target terminal density function ρ₁
- `rho0x::Vector` - precomputed ρ₀(X₀) values
- `rho1x::Vector` - precomputed ρ₁(X₀) values
- `mu::Real` - penalty parameter μ
"""
mutable struct Gls
    rho0
    rho1
    rho0x::AbstractVector # = rho0(X0), stored for efficiency
    rho1x::AbstractVector # = rho1(X0), stored for efficiency
    mu::Real      # = penalty parameter
end

function Base.show(io::IO, G::Gls)
  print(io, "$(G.mu) ⋅ Gls(U)")
end

function (G::Gls)(U::AbstractArray{R}) where R <: Real
    d = spatial_dim(U)

    # Add numerical safeguards to avoid division by zero and overflow/underflow
    ε = sqrt(eps(R))  # ~1e-8 for Float64
    rho0x_safe = max.(G.rho0x, ε)

    # Clamp U[d+1,:] to avoid extreme values in exp (prevents overflow)
    U_clamped = clamp.(U[d+1,:], -R(100), R(100))

    # Compute exp once and reuse (performance improvement!)
    detDy = exp.(-U_clamped)               # det(Dy) from log-determinant
    rho_T = rho0x_safe ./ detDy            # Terminal density via change of variables
    rho_target = G.rho1(spatial_positions(U))  # Target density at terminal positions

    # Least-squares cost: μ/2 ∫(ρ(T) - ρ₁)² · (ρ₀/ρ(T)) dx
    diff = rho_T - rho_target
    weight = detDy ./ rho0x_safe

    return (G.mu / 2) .* diff.^2 .* weight
end

function getDeltaG(G::Gls,U::AbstractArray{R}) where R <: Real
    d = spatial_dim(U)

    # Add numerical safeguards
    ε = sqrt(eps(R))
    rho0x_safe = max.(G.rho0x, ε)

    # Clamp to avoid overflow
    U_clamped = clamp.(U[d+1,:], -R(100), R(100))
    detDy = exp.(U_clamped)

    return G.mu .* (rho0x_safe ./ detDy .- G.rho1(spatial_positions(U)))
end

"""
    Gkl

Kullback-Leibler divergence terminal cost functional

Penalizes KL divergence KL(ρ(T)||ρ₁) between terminal and target densities

# Formula
G(U) = μ ∫ρ(x,T) log(ρ(x,T)/ρ₁(x)) dx

# Fields
- `rho0` - initial density function ρ₀
- `rho1` - target terminal density function ρ₁
- `rho0x::Vector` - precomputed ρ₀(X₀) values
- `rho1x::Vector` - precomputed ρ₁(X₀) values
- `mu::Real` - penalty parameter μ
"""
mutable struct Gkl
    rho0
    rho1
    rho0x::Vector # = rho0(X0), stored for efficiency
    rho1x::Vector # = rho1(X0), stored for efficiency
    mu::Real      # = penalty parameter
end
function (G::Gkl)(U::AbstractArray{R}) where R <: Real
    d = spatial_dim(U)

    # Add numerical safeguards to avoid log(0) = -Inf
    ε = sqrt(eps(R))  # ~1e-8 for Float64
    rho0x_safe = max.(G.rho0x, ε)
    rho1_vals = G.rho1(spatial_positions(U))
    rho1_safe = max.(rho1_vals, ε)

    return G.mu .* (log.(rho0x_safe) .- U[d+1,:] .- log.(rho1_safe))
end

function Base.show(io::IO, G::Gkl)
  print(io, "$(G.mu) ⋅ Gkl(U)")
end

function getDeltaG(G::Gkl,U::AbstractArray{R})  where R <: Real
    d = spatial_dim(U)

    # Add numerical safeguards to avoid log(0) = -Inf
    ε = sqrt(eps(R))
    rho0x_safe = max.(G.rho0x, ε)
    rho1_vals = G.rho1(spatial_positions(U))
    rho1_safe = max.(rho1_vals, ε)

    return G.mu .* (one(R) .+ log.(rho0x_safe) .- U[d+1,:] .- log.(rho1_safe))
end




"""
    Gpref

Preference terminal cost functional

Penalizes deviation from preferred terminal positions via function Pref(x)

# Formula
G(U) = μ ∫Pref(x(T)) dx

# Fields
- `Pref::Function` - preference function mapping positions to costs
- `rho0x::Vector` - precomputed ρ₀(X₀) values
- `mu::Real` - penalty parameter μ
"""
mutable struct Gpref
    Pref::Function  # preference function
    rho0x::Vector   #
    mu::Real        # = penalty parameter
end
function (G::Gpref)(U)
    return G.mu .* G.Pref(spatial_positions(U))
end

function getDeltaG(G::Gpref,U::AbstractArray{R})  where R <: Real
    return G.mu .* G.Pref(spatial_positions(U))
end
