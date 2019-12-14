export Fcomb, getDeltaF, Fp, Fe, F0

"""
combine different F's
"""
mutable struct Fcomb
    Fs::Array
end

function (F::Fcomb)(U,t)
    res = F.Fs[1](U,t)
    for k=2:length(F.Fs)
        res += F.Fs[k](U,t)
    end
    return res
end
function getDeltaF(F::Fcomb,U,t)
    res = getDeltaF(F.Fs[1],U,t)
    for k=2:length(F.Fs)
        res += getDeltaF(F.Fs[k],U,t)
    end
    return res
end

function Base.show(io::IO, F::Fcomb)
  print(io, F.Fs[1])
  for k=2:length(F.Fs)
      print(io, " + $(F.Fs[1])")
  end
end

"""
F for potential
"""
mutable struct Fp{R}
    Q # function for spatial potential (spacial preference for agents)
    rho0
    rho0x::AbstractVector{R}
    λ::R # vector of length 3 containing penalties for three terms in F
end

function (F::Fp{R})(U::AbstractArray{R},t::R) where R <: Real
    (d,nex)     = size(U)
    d -= 4
    return F.λ .* F.Q([U[1:d,:]; fill(t,1,nex)])
end

"""
L2 derivative of running costs F
"""
function getDeltaF(F::Fp{R},U::AbstractArray{R},t::R) where R <: Real
    (d,nex) = size(U)
    d      -= 4
    return F.λ.*F.Q([U[1:d,:]; fill(t,1,nex)])
end


"""
F for entropy
"""
mutable struct Fe{R}
    rho0
    rho0x::AbstractVector{R}
    λ::R # vector of length 3 containing penalties for three terms in F
end

function (F::Fe{R})(U::AbstractArray{R},t::R) where R <: Real
    (d,nex)     = size(U)
    d -= 4
    return F.λ .* (log.(F.rho0x) - vec(U[end-2,:]))
end

function getDeltaF(F::Fe{R},U::AbstractArray{R},t::R) where R <: Real
    (d,nex) = size(U)
    d      -= 4
    return F.λ.*(log.(F.rho0x) - vec(U[end-2,:]) .+ R(1))
end

struct F0
end

function (F::F0)(U::AbstractArray{R},t) where R <: Real
    return R(0.0) * U[1,:]
end

function getDeltaF(F::F0,U::AbstractArray{R},t) where R <: Real
    return R(0.0) * U[1,:]
end
