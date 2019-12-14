export RK1Step, RK4Step

struct RK1Step
end
struct RK4Step
end



# changes: added d as input and
function step(stepper::RK1Step,odefun,J,U::AbstractArray{R},Θ,tk::R,tkp1::R) where R <: Real
    U  += (tkp1-tk) .* odefun(J, U, Θ, tk)
    return U
end

function step(stepper::RK4Step,odefun,J,U0::AbstractArray{R},Θ,tk::R,tkp1::R) where R<:Real
    h    = tkp1 - tk

    K  = h .* odefun(J, U0, Θ, tk)
    U  = U0 + R.(1/6) .* K

    K  = h .* odefun(J, U0 + 0.5 .* K, Θ, tk+h/2)
    U  = U + R.(2/6) .* K

    K  = h .* odefun(J, U0 + 0.5 .* K, Θ, tk+h/2)
    U  = U + R.(2/6) .* K

    K  = h .* odefun(J, U0 + K, Θ, tk+h)
    U  = U + R.(1/6) .* K

    return U
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
    d   = size(U0,1)-4
    X0  = (U0[1:d,:])
    for k=1:N
        temp = step(stepper,odefun, J, UArray[:,:,k], Θ , tk, tk+h)

        if typeof(temp) <: TrackedArray
            UArray[:,:,k+1] = temp.data
        else
            UArray[:,:,k+1] = temp
        end
        tk += h
    end
    return UArray
end
