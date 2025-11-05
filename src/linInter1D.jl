"""
    linInter1D(tk, T, Θ)

Linear interpolation of time-dependent parameters at time tk ∈ [0,T]

# Arguments
- `tk::Real`: Query time point
- `T::Real`: Final time
- `Θ`: Parameter array/tuple with time as last dimension

# Algorithm
Assumes uniform time grid with Nt nodes: t_i = (i-1)*H where H = T/(Nt-1)
For tk ∈ [t_i, t_{i+1}], returns weighted average: w*Θ[i] + (1-w)*Θ[i+1]

# Returns
Parameters interpolated at time tk
"""
function linInter1D(tk::R,T::R,Θ::Tuple{AbstractArray{R},AbstractArray{R}}) where R <: Real
    Θ1 = linInter1D(tk,T,Θ[1])
    Θ2 = linInter1D(tk,T,Θ[2])
    return (Θ1,Θ2)
end


function linInter1D(tk::R,T::R,Θ::Tuple) where R <: Real
    Θk = (linInter1D(tk,T,Θk) for Θk in Θ)
    return tuple(Θk...)
end

function linInter1D(tk::R,T::R,Θ::AbstractArray{R,2}) where R <: Real
    Nt   = size(Θ,2)
    H   = T/(Nt-1)  # Grid spacing: assume nodal discretization for Θ
    idl = Int64(floor(tk/H))+1  # idl = index_left: left boundary of interval
    w   = ((H*idl)-tk)/H        # w = weight for left node

    if idl==0
        return Θ[:,idl+1]
    elseif idl==Nt
        return Θ[:,idl]
    else
        return w .* Θ[:,idl] + (1-w).*Θ[:,idl+1]  # Linear interpolation
    end
end

function linInter1D(tk::R,T::R,Θ::AbstractArray{R,3}) where R <: Real
    Nt   = size(Θ,3)
    H   = T/(Nt-1)  # Grid spacing: assume nodal discretization for Θ
    idl = Int64(floor(tk/H))+1  # idl = index_left: left boundary of interval
    w   = ((H*idl)-tk)/H        # w = weight for left node

    if idl==0
        Θk = Θ[:,:,idl+1]
    elseif idl==Nt
        Θk = Θ[:,:,idl]
    else
        Θk =  w .* Θ[:,:,idl] + (1-w).*Θ[:,:,idl+1]  # Linear interpolation
    end
    return Θk
end
