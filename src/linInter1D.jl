
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
    H   = T/(Nt-1)  # assume nodal discretization for Θ
    idl = Int64(floor(tk/H))+1
    w   = ((H*idl)-tk)/H

    if idl==0
        return Θ[:,idl+1]
    elseif idl==Nt
        return Θ[:,idl]
    else
        return w .* Θ[:,idl] + (1-w).*Θ[:,idl+1]
    end
end

function linInter1D(tk::R,T::R,Θ::AbstractArray{R,3}) where R <: Real
    Nt   = size(Θ,3)
    H   = T/(Nt-1)  # assume nodal discretization for Θ
    idl = Int64(floor(tk/H))+1
    w   = ((H*idl)-tk)/H

    if idl==0
        Θk = Θ[:,:,idl+1]
    elseif idl==Nt
        Θk = Θ[:,:,idl]
    else
        Θk =  w .* Θ[:,:,idl] + (1-w).*Θ[:,:,idl+1]
    end
    return Θk
end
