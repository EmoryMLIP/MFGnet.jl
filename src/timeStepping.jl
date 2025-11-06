export RK1Step, RK4Step, step, integrate, integrate2

struct RK1Step
end
struct RK4Step
end



# changes: added d as input and
function step(stepper::RK1Step,odefun,J,U::AbstractArray{R},Θ,tk::R,tkp1::R) where R <: Real
    # Forward Euler: U_{n+1} = U_n + h*f(U_n, t_n)
    return U + (tkp1-tk) .* odefun(J, U, Θ, tk)
end

function step(stepper::RK4Step,odefun,J,U0::AbstractArray{R},Θ,tk::R,tkp1::R) where R<:Real
    # Classic 4th-order Runge-Kutta method
    h = tkp1 - tk

    # Compute all four stages with correct intermediate values
    k1 = odefun(J, U0, Θ, tk)
    k2 = odefun(J, U0 + (h/2) .* k1, Θ, tk + h/2)
    k3 = odefun(J, U0 + (h/2) .* k2, Θ, tk + h/2)
    k4 = odefun(J, U0 + h .* k3, Θ, tk + h)

    # Weighted combination: U_{n+1} = U_n + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    return U0 + (h/6) .* (k1 + 2 .* k2 + 2 .* k3 + k4)
end

"""
numerical integration of ODE
"""
function integrate(stepper,odefun,J,U::AbstractArray{R},Θ,tspan::AbstractArray{R},N::Int) where R <: Real
    h   = (tspan[2]-tspan[1])/N
    tk  = tspan[1]
    for k=1:N
        U = step(stepper,odefun,J, U, Θ,tk,tk+h)
        tk += h
    end
    return U
end

"""
numerical integration of ODE, with storing intermediate states
"""
function integrate2(stepper,odefun,J,U0::AbstractArray{R},Θ,tspan::AbstractArray{R},N::Int) where R <: Real
    h = (tspan[2]-tspan[1])/N
    tk = tspan[1]
    UArray = zeros(R,tuple([size(U0)... N+1]...))
    UArray[:,:,1] = U0
    for k=1:N
        temp = step(stepper,odefun, J, UArray[:,:,k], Θ , tk, tk+h)
        UArray[:,:,k+1] = temp
        tk += h
    end
    return UArray
end
