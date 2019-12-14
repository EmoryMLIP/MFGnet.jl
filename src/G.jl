export Gcomb, Gls, Gkl, Gls2, Gpref
"""
combine different G's
"""
mutable struct Gcomb
    Gs::Array
end

function (G::Gcomb)(U)
    res = G.Gs[1](U)
    for k=2:length(G.Gs)
        res += G.Gs[k](U)
    end
    return res
end
function Base.show(io::IO, G::Gcomb)
  print(io, G.Gs[1])
  for k=2:length(G.Gs)
      print(io, " + $(G.Gs[k])")
  end
end

"""
Least-Squares Terminal Cost
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

    d   = size(U,1)-4
    return G.mu*R(0.5)* ( G.rho0x./ exp.(-U[d+1,:]) - G.rho1(U[1:d,:]) ).^2 .* (exp.(-U[d+1,:])./G.rho0x)

end

function getDeltaG(G::Gls,U::AbstractArray)
    (d,nex) = size(U)
    d      -= 4
    detDy = exp.(U[d+1,:])
    return G.mu.*(G.rho0x ./detDy - G.rho1(U[1:d,:]))
end

"""
KL Divergence Terminal Cost
"""
mutable struct Gkl
    rho0
    rho1
    rho0x::Vector # = rho0(X0), stored for efficiency
    rho1x::Vector # = rho1(X0), stored for efficiency
    mu::Real      # = penalty parameter
end
function (G::Gkl)(U)
    (d,nex) = size(U)
    d -= 4
    return G.mu .* (log.(G.rho0x) - U[d+1,:] - log.(G.rho1(U[1:d,:])))
end
function Base.show(io::IO, G::Gkl)
  print(io, "$(G.mu) ⋅ Gkl(U)")
end

function getDeltaG(G::Gkl,U::AbstractArray{R})  where R <: Real
    (d,nex) = size(U)
    d      -= 4
    return G.mu.*(R(1.0) .+ log.(G.rho0x) - U[d+1,:] - log.(G.rho1(U[1:d,:])))
end




"""
 Preference Terminal Cost
"""
mutable struct Gpref
    Pref::Function  # preference function
    rho0x::Vector   #
    mu::Real        # = penalty parameter
end
function (G::Gpref)(U)
    (d,nex) = size(U)
    d -= 4
    return G.mu .* G.Pref(U[1:d,:])
end

function getDeltaG(G::Gpref,U::AbstractArray{R})  where R <: Real
    (d,nex) = size(U)
    d      -= 4
    return G.mu.*(G.Pref(U[1:d,:]))
end
