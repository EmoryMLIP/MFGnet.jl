export Gcomb, Gls, Gkl, Gls2, Gpref, getDeltaG
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

    # Add numerical safeguards to avoid division by zero and overflow/underflow
    ε = sqrt(eps(R))  # ~1e-8 for Float64
    rho0x_safe = max.(G.rho0x, ε)

    # Clamp U[d+1,:] to avoid extreme values in exp
    U_clamped = clamp.(U[d+1,:], -R(100), R(100))  # Prevents overflow in exp
    exp_neg_U = exp.(-U_clamped)

    rho1_vals = G.rho1(U[1:d,:])

    return G.mu*R(0.5)* ( rho0x_safe ./ exp_neg_U - rho1_vals ).^2 .* (exp_neg_U ./ rho0x_safe)
end

function getDeltaG(G::Gls,U::AbstractArray{R}) where R <: Real
    (d,nex) = size(U)
    d      -= 4

    # Add numerical safeguards
    ε = sqrt(eps(R))
    U_clamped = clamp.(U[d+1,:], -R(100), R(100))
    detDy = exp.(U_clamped)
    rho0x_safe = max.(G.rho0x, ε)

    return G.mu.*(rho0x_safe ./detDy - G.rho1(U[1:d,:]))
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
function (G::Gkl)(U::AbstractArray{R}) where R <: Real
    (d,nex) = size(U)
    d -= 4

    # Add numerical safeguards to avoid log(0) = -Inf
    ε = sqrt(eps(R))  # ~1e-8 for Float64
    rho0x_safe = max.(G.rho0x, ε)
    rho1_vals = G.rho1(U[1:d,:])
    rho1_safe = max.(rho1_vals, ε)

    return G.mu .* (log.(rho0x_safe) - U[d+1,:] - log.(rho1_safe))
end

function Base.show(io::IO, G::Gkl)
  print(io, "$(G.mu) ⋅ Gkl(U)")
end

function getDeltaG(G::Gkl,U::AbstractArray{R})  where R <: Real
    (d,nex) = size(U)
    d      -= 4

    # Add numerical safeguards to avoid log(0) = -Inf
    ε = sqrt(eps(R))
    rho0x_safe = max.(G.rho0x, ε)
    rho1_vals = G.rho1(U[1:d,:])
    rho1_safe = max.(rho1_vals, ε)

    return G.mu.*(R(1.0) .+ log.(rho0x_safe) - U[d+1,:] - log.(rho1_safe))
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
