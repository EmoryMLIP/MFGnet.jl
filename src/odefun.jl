

"""

right-hand side associated with

∂t u = f(u,Θ), where

u = (x,l,v,hj) with

x - particle position
l - log determinant
v - accumulated transport costs
hj - accumulates least-squares error of -∂Φ(x,t) + 0.5 * |∇Φ(x,t)|^2 along trajectory
"""
function odefun(J,U::AbstractArray{R},Θ,t::R) where R <: Real
    (d,nex) = size(U)
    d = d-4;
    XT = [ U[1:d,:]; fill(t,1,nex)]
    Phi = J.Φ(XT,Θ) # run fwd prop to populate N.tmp
    gradPhi = getGradPotential(J.Φ,XT,Θ)
    trH = getTraceHess(J.Φ,XT,Θ)
    dx = -(1/J.α[1])*gradPhi[1:d,:]
    dl = -(1/J.α[1])*trH
    dv = sum(dx.^2,dims=1)
    df = reshape(J.F(U,t),1,nex)
    hj = abs.(-reshape(gradPhi[end,:],1,:) -
                reshape(J.α[2].*getDeltaF(J.F,U,t),1,:) +
                R(0.5*J.α[1]) .* dv)
    return [dx;reshape(dl,1,nex);dv;df;hj]
end
