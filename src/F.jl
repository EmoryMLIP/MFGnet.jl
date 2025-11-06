export Fcomb, getDeltaF, Fp, Fe, F0

"""
    Fcomb

Combination of multiple interaction functionals

# Formula
F(U,t) = ∑ᵢ Fᵢ(U,t)

# Fields
- `Fs::Vector` - array of interaction functionals to sum
"""
mutable struct Fcomb
    Fs::Array
end

function (F::Fcomb)(U,t)
    return sum(f(U,t) for f in F.Fs)
end
function getDeltaF(F::Fcomb,U,t)
    return sum(getDeltaF(f,U,t) for f in F.Fs)
end

function Base.show(io::IO, F::Fcomb)
  print(io, F.Fs[1])
  for k=2:length(F.Fs)
      print(io, " + $(F.Fs[k])")  # Fixed: was F.Fs[1], should be F.Fs[k]
  end
end

"""
    Fp{R<:Real}

Spatial potential interaction functional

Represents agents' spatial preferences via potential Q(x,t)

# Fields
- `Q` - spatial potential function (defines preferred agent locations)
- `rho0` - initial density function
- `rho0x::Vector{R}` - precomputed ρ₀(X) values
- `λ::R` - penalty weight for this term
"""
mutable struct Fp{R}
    Q # function for spatial potential (spacial preference for agents)
    rho0
    rho0x::AbstractVector{R}
    λ::R # vector of length 3 containing penalties for three terms in F
end

function (F::Fp{R})(U::AbstractArray{R},t::R) where R <: Real
    nex = size(U, 2)
    X = spatial_positions(U)
    return F.λ .* F.Q([X; fill(t,1,nex)])
end

"""
L2 derivative of running costs F
"""
function getDeltaF(F::Fp{R},U::AbstractArray{R},t::R) where R <: Real
    nex = size(U, 2)
    X = spatial_positions(U)
    return F.λ .* F.Q([X; fill(t,1,nex)])
end


"""
    Fe{R<:Real}

Entropy interaction functional

Measures relative entropy H(ρ|ρ₀) = ∫ρ log(ρ/ρ₀)dx

# Fields
- `rho0` - reference density function ρ₀
- `rho0x::Vector{R}` - precomputed log(ρ₀(X)) values
- `λ::R` - penalty weight for entropy term
"""
mutable struct Fe{R}
    rho0
    rho0x::AbstractVector{R}
    λ::R # vector of length 3 containing penalties for three terms in F
end

function (F::Fe{R})(U::AbstractArray{R},t::R) where R <: Real
    # Add numerical safeguard: clamp densities to avoid log(0) = -Inf
    ε = sqrt(eps(R))  # ~1e-8 for Float64, ~1e-4 for Float32
    rho0x_safe = max.(F.rho0x, ε)
    # U[end-2,:] contains log determinant component
    return F.λ .* (log.(rho0x_safe) .- vec(U[end-2,:]))
end

function getDeltaF(F::Fe{R},U::AbstractArray{R},t::R) where R <: Real
    # Add numerical safeguard: clamp densities to avoid log(0) = -Inf
    ε = sqrt(eps(R))
    rho0x_safe = max.(F.rho0x, ε)
    # Derivative includes +1 correction term
    return F.λ .* (log.(rho0x_safe) .- vec(U[end-2,:]) .+ one(R))
end

"""
    F0

Zero interaction functional (no running cost)

Used when there is no interaction term in the MFG objective
"""
struct F0
end

function (F::F0)(U::AbstractArray{R},t) where R <: Real
    return R(0.0) * U[1,:]
end

function getDeltaF(F::F0,U::AbstractArray{R},t) where R <: Real
    return R(0.0) * U[1,:]
end
