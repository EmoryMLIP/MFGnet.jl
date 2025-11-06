"""
    odefun(J::MeanFieldGame, U, Θ, t)

Right-hand side of MFG characteristic ODE: ∂ₜu = f(u,Θ,t)

# State Vector
U = [x; l; v; f; hj] where:
- x: particle position in ℝ^d
- l: log determinant (for density via change of variables)
- v: accumulated transport cost ∫½|v(t)|² dt
- f: accumulated interaction cost ∫F(ρ,t) dt
- hj: accumulated Hamilton-Jacobi residual (PDE violation)

# Dynamics
- dx/dt = -α₁⁻¹∇ₓΦ(x,t)              (optimal velocity)
- dl/dt = -α₁⁻¹tr(∇²Φ)               (density evolution)
- dv/dt = ½|dx/dt|²                  (instantaneous transport cost)
- df/dt = F(ρ,t)                     (interaction cost)
- dhj/dt = |∂ₜΦ + δF - α₁·½|∇Φ|²|   (HJB equation residual)

# Returns
dU/dt: Time derivative of state vector
"""
function odefun(J,U::AbstractArray{R},Θ,t::R) where R <: Real
    nex = size(U, 2)
    d = spatial_dim(U)  # Spatial dimension

    # Augment position with time for potential evaluation
    XT = [spatial_positions(U); fill(t,1,nex)]

    # Evaluate potential and its derivatives
    Phi = J.Φ(XT,Θ)  # Forward pass to populate cached values
    gradPhi = getGradPotential(J.Φ,XT,Θ)  # ∇Φ(x,t) = [∇ₓΦ; ∂ₜΦ]
    trH = getTraceHess(J.Φ,XT,Θ)           # tr(∇²Φ) for density evolution

    # Optimal velocity: dx/dt = -α₁⁻¹∇ₓΦ (gradient descent on potential)
    dx = -(1/J.α[1]) * gradPhi[1:d,:]

    # Log-determinant evolution: dl/dt = -α₁⁻¹tr(∇²Φ) (continuity equation)
    dl = -(1/J.α[1]) * trH

    # Transport cost rate: dv/dt = ½|velocity|²
    dv = 0.5 .* sum(dx.^2, dims=1)

    # Interaction cost rate: df/dt = F(ρ,t)
    df = reshape(J.F(U,t),1,nex)

    # Hamilton-Jacobi residual: ∂ₜΦ + δF - α₁·½|∇Φ|² (should be zero on solution)
    hj = abs.(-reshape(gradPhi[end,:],1,:) -
                reshape(J.α[2].*getDeltaF(J.F,U,t),1,:) +
                J.α[1] .* dv)

    return [dx;reshape(dl,1,nex);dv;df;hj]
end
