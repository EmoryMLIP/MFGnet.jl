export ResNN

"""
ResNN

Residual Neural Network structure
"""
mutable struct ResNN{R<:Real}
    layer::SingleLayer   # description of layer
    ts::Vector{R}      # time points
    tmp::Array{AbstractArray{R},2}     # storage for intermediates
end

ResNN(layer=SingleLayer(),ts::Vector{R}=[0.0 0.5 1.0]) where R<:Real =
        ResNN(layer,ts,Array{AbstractArray{R}}(undef,length(ts)-1,2))

nLayers(N::ResNN) = length(N.ts)-1

"""
evaluate layer for current weights Θ=(K,b)
"""
function (N::ResNN{R})(S::AbstractArray{R},Θ) where R <: Real
    T = maximum(N.ts)
    for k=1:nLayers(N)
        N.tmp[k,1] = S
        hk = N.ts[k+1]-N.ts[k]
        Θk = linInter1D(N.ts[k],T,Θ)
        S += hk .* N.layer(S,Θk)
    end
    return S
end

"""
compute matvec J_S N(S,Θ)'*Z
"""
function getJSTmv(N::ResNN{R},Z::AbstractVector{R},S::AbstractArray{R},Θ)  where R <: Real
    T = maximum(N.ts)
    hk = N.ts[end]-N.ts[end-1]
    Θk = linInter1D(N.ts[end-1],T,Θ)
    N.tmp[end,2] = Z
    Z = Z .+ hk .* getJSTmv(N.layer,Z,N.tmp[end,1],Θk)

    for k=nLayers(N)-1:-1:1
        N.tmp[k,2] = Z
        hk = N.ts[k+1]-N.ts[k]
        Θk = linInter1D(N.ts[k],T,Θ)
        Z +=  hk .* getJSTmv(N.layer,Z,N.tmp[k,1],Θk)
    end
    return Z
end

function getJSTmv(N::ResNN{R},Z::AbstractArray{R},S::AbstractArray{R},Θ) where R <: Real
    T = maximum(N.ts)
    for k=nLayers(N):-1:1
        N.tmp[k,2] = Z
        hk = N.ts[k+1]-N.ts[k]
        Θk = linInter1D(N.ts[k],T,Θ)
        Z +=  hk .* getJSTmv(N.layer,Z,N.tmp[k,1],Θk)
    end
    return Z
end

"""
compute

(I+JS') d2Z (I+JS), where JS is Jacobian of layer
= (I + hk .* JS') ((I+ hk .* JS')*d2Z)'

"""
function getJSTd2ZJSmv(N::ResNN{R},d2Z::AbstractArray{R},hk::R,s::AbstractVector{R},Θ) where R <: Real
    t1 = d2Z + hk .* getJSTmv(N.layer,d2Z,s,Θ)
    return t1' + hk .* getJSTmv(N.layer,t1',s,Θ)
end

function getJSTd2ZJSmv(N::ResNN{R},d2Z::AbstractArray{R,3},hk,S::AbstractArray{R,2},Θ) where R <: Real
    t1 = d2Z + hk .* getJSTmv(N.layer,d2Z,S,Θ)
    t1 = permutedims(t1,(2,1,3))
    return t1 + hk .* getJSTmv(N.layer,t1,S,Θ)
end


"""
compute matvec J_S(J_S N(S,Θ)'*Z(S))

here we use product rule

J_S N(S,Θ)'*dZ + J_S(N(S,Θ)'*Zfix)
"""
function getJSJSTmv(N::ResNN{R},dZ::AbstractVector{R},S::AbstractArray{R},Θ) where R <: Real
    T = maximum(N.ts)

    Θk = linInter1D(N.ts[end-1],T,Θ)
    hk = N.ts[end]-N.ts[end-1]
    d2Z =  hk .* getJSJSTmv(N.layer,dZ,N.tmp[end,1],Θk)
    dZ  = dZ .+ hk .* getJSTmv(N.layer,dZ,N.tmp[end,1],Θk)

    for k=nLayers(N)-1:-1:1
        Θk = linInter1D(N.ts[k],T,Θ)
        hk = N.ts[k+1]-N.ts[k]
        d2Z1 =  hk .* getJSJSTmv(N.layer,dZ,N.tmp[k,1],Θk)
        d2Z2 = getJSTd2ZJSmv(N,d2Z, hk, N.tmp[k,1],Θk)
        d2Z = d2Z1 + d2Z2
        if k>1
            dZ  += hk .* getJSTmv(N.layer,dZ,N.tmp[k,1],Θk)
        end
    end
    return d2Z
end

function getJSJSTmv(N::ResNN{R},dZ::AbstractArray{R},d2Z::AbstractArray{R},S::AbstractArray{R},Θ) where R <: Real
    T = maximum(N.ts)
    for k=nLayers(N):-1:1
        Θk = linInter1D(N.ts[k],T,Θ)
        hk = N.ts[k+1]-N.ts[k]
        d2Z1 =  hk .* getJSJSTmv(N.layer,dZ,N.tmp[k,1],Θk)
        d2Z2 = getJSTd2ZJSmv(N,d2Z, hk, N.tmp[k,1],Θk)
        d2Z = d2Z1 + d2Z2
        if k>1
            dZ  += hk .* getJSTmv(N.layer,dZ,N.tmp[k,1],Θk)
        end
    end
    return d2Z
end

function getGradAndHessian(N::ResNN{R},dZ::AbstractArray{R},S::AbstractArray{R},Θ) where R <: Real
    # Here, there is no Hessian from the following layer, so we assume it is zero
    T = maximum(N.ts)

    Θk = linInter1D(N.ts[end-1],T,Θ)
    hk = N.ts[end]-N.ts[end-1]
    N.tmp[end,2] = dZ
    ddZ, d2Z = getGradAndHessian(N.layer,dZ,N.tmp[end,1],Θk)
    dZ  = dZ .+ hk .* ddZ
    d2Z = hk.*d2Z

    for k=nLayers(N)-1:-1:1
        N.tmp[k,2] = dZ
        Θk = linInter1D(N.ts[k],T,Θ)
        hk = N.ts[k+1]-N.ts[k]
        ddZ, d2Z1 =  getGradAndHessian(N.layer,dZ,N.tmp[k,1],Θk)
        d2Z2 = getJSTd2ZJSmv(N,d2Z, hk, N.tmp[k,1],Θk)
        d2Z = hk .* d2Z1 + d2Z2
        dZ  += hk .* ddZ
    end
    return dZ, d2Z
end

function getGradAndHessian(N::ResNN{R},dZ::AbstractArray{R},d2Z::AbstractArray{R},S::AbstractArray{R},Θ) where R <: Real
    T = maximum(N.ts)
    for k=nLayers(N):-1:1
        N.tmp[k,2] = dZ
        Θk = linInter1D(N.ts[k],T,Θ)
        hk = N.ts[k+1]-N.ts[k]
        ddZ,d2Z1 =   getGradAndHessian(N.layer,dZ,N.tmp[k,1],Θk)
        d2Z2 = getJSTd2ZJSmv(N,d2Z, hk, N.tmp[k,1],Θk)
        d2Z = hk .*d2Z1 + d2Z2
        dZ  += hk .* ddZ
    end
    return dZ,d2Z
end


function getTraceHess(N::ResNN,S::AbstractArray{R},Θ) where R <: Real
    T  = maximum(N.ts)
    Θk = linInter1D(N.ts[1],T,Θ)
    hk = N.ts[2]-N.ts[1]

    trH1,Jac = getTraceHessAndGrad(N.layer,N.tmp[1,2],N.tmp[1,1],Θk)
    Jac =  Matrix(R(1.0)*I,size(Jac,1),size(Jac,2)) .+ hk .* Jac
    trH2, Jac = getTraceHessAndGrad(N,[],Jac,N.tmp[2,1],Θ,2)
    return hk*trH1 + trH2
end

function getTraceHess(N::ResNN,w,Jac::AbstractArray{R},S::AbstractArray{R},Θ,k::Int=1) where R <: Real
    # FEATURE: second input, w, is not used here.
    T  = maximum(N.ts)
    Θk = linInter1D(N.ts[k],T,Θ)
    hk = N.ts[k+1]-N.ts[k]
    if k < nLayers(N)
        trH1,Jt = getTraceHessAndGrad(N.layer,N.tmp[k,2],Jac,N.tmp[k,1],Θk)
        Jac = Jac + hk .* Jt
        trH2, Jac = getTraceHessAndGrad(N,[],Jac,N.tmp[k+1,1],Θ,k+1)
        return hk .* trH1 + trH2, Jac
    else
        trH1 = getTraceHess(N.layer,N.tmp[k,2],Jac,N.tmp[k,1],Θk)
        return hk .* trH1
    end
end

function getTraceHessAndGrad(N::ResNN,S::AbstractArray{R},Θ) where R <: Real
    T  = maximum(N.ts)
    Θk = linInter1D(N.ts[1],T,Θ)
    hk = N.ts[2]-N.ts[1]

    trH1,Jac = getTraceHessAndGrad(N.layer,N.tmp[1,2],N.tmp[1,1],Θk)
    Jac =  Matrix(R(1.0)*I,size(Jac,1),size(Jac,2)) .+ hk .* Jac
    trH2, Jac = getTraceHessAndGrad(N,[],Jac,N.tmp[2,1],Θ,2)
    return hk*trH1 + trH2, Jac
end

function getTraceHessAndGrad(N::ResNN,w,Jac::AbstractArray{R},S::AbstractArray{R},Θ,k::Int=1) where R <: Real
    # FEATURE: second input, w, is not used here.
    T  = maximum(N.ts)
    Θk = linInter1D(N.ts[k],T,Θ)
    hk = N.ts[k+1]-N.ts[k]
    trH1,Jt = getTraceHessAndGrad(N.layer,N.tmp[k,2],Jac,N.tmp[k,1],Θk)
    Jac = Jac + hk .* Jt
    if k < nLayers(N)
        trH2, Jac = getTraceHessAndGrad(N,[],Jac,N.tmp[k+1,1],Θ,k+1)
        return hk .* trH1 + trH2, Jac
    else
        return hk .* trH1, Jac
    end
end
