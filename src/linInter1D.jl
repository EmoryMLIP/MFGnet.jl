
function linInter1D(tk::R,T::R,Θ::Tuple{AbstractArray{R},AbstractArray{R}}) where R <: Real
    Θ1 = linInter1D(tk,T,Θ[1])
    Θ2 = linInter1D(tk,T,Θ[2])
    return (Θ1,Θ2)
end


function linInter1D(tk::R,T::R,Θ::Tuple) where R <: Real
    Θk = (linInter1D(tk,T,Θk) for Θk in Θ)
    return tuple(Θk...)
end

"""
    linInter1D(tk::R, T::R, Θ::AbstractArray{R,2}) where R <: Real

Linear interpolation in time for 2D parameter arrays.

# Arguments
- `tk::R`: Query time point
- `T::R`: Final time (defines the time span [0,T])
- `Θ::AbstractArray{R,2}`: Parameter array of size (d, Nt) where Nt is number of time points

# Returns
- Interpolated parameters at time tk

# Note
Time values outside [0,T] are clamped to the boundaries to ensure robustness
in numerical ODE integration where small floating-point errors might push tk
slightly outside the valid range.
"""
function linInter1D(tk::R,T::R,Θ::AbstractArray{R,2}) where R <: Real
    Nt = size(Θ,2)

    # Handle edge case: single time point
    if Nt < 2
        return Θ[:,1]
    end

    # Clamp time to valid range [0, T] for numerical robustness
    tk_clamped = clamp(tk, zero(R), T)

    H = T/(Nt-1)  # assume nodal discretization for Θ
    idl = Int64(floor(tk_clamped/H)) + 1

    # Ensure index is within bounds
    idl = clamp(idl, 1, Nt)

    if idl == Nt
        # At or past final time point
        return Θ[:,Nt]
    else
        # Interpolate between idl and idl+1
        w = ((H*idl)-tk_clamped)/H
        w = clamp(w, zero(R), one(R))  # Ensure valid interpolation weight
        return w .* Θ[:,idl] + (one(R)-w) .* Θ[:,idl+1]
    end
end

"""
    linInter1D(tk::R, T::R, Θ::AbstractArray{R,3}) where R <: Real

Linear interpolation in time for 3D parameter arrays.

# Arguments
- `tk::R`: Query time point
- `T::R`: Final time (defines the time span [0,T])
- `Θ::AbstractArray{R,3}`: Parameter array of size (d1, d2, Nt) where Nt is number of time points

# Returns
- Interpolated parameters at time tk of size (d1, d2)

# Note
Time values outside [0,T] are clamped to the boundaries to ensure robustness.
"""
function linInter1D(tk::R,T::R,Θ::AbstractArray{R,3}) where R <: Real
    Nt = size(Θ,3)

    # Handle edge case: single time point
    if Nt < 2
        return Θ[:,:,1]
    end

    # Clamp time to valid range [0, T] for numerical robustness
    tk_clamped = clamp(tk, zero(R), T)

    H = T/(Nt-1)  # assume nodal discretization for Θ
    idl = Int64(floor(tk_clamped/H)) + 1

    # Ensure index is within bounds
    idl = clamp(idl, 1, Nt)

    if idl == Nt
        # At or past final time point
        Θk = Θ[:,:,Nt]
    else
        # Interpolate between idl and idl+1
        w = ((H*idl)-tk_clamped)/H
        w = clamp(w, zero(R), one(R))  # Ensure valid interpolation weight
        Θk = w .* Θ[:,:,idl] + (one(R)-w) .* Θ[:,:,idl+1]
    end
    return Θk
end
