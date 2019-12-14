export NN

"""
Neural Network structure
"""
mutable struct NN
    layers
    tmp
end

NN(layers=[SingleLayer();SingleLayer()]) = NN(layers,Array{AbstractArray}(undef,length(layers),2))

nLayers(N::NN) = length(N.layers)

"""
evaluate layer for current weights Θ=(K,b)
"""
function (N::NN)(S::AbstractArray{R},Θ) where R <: Real
    for k=1:nLayers(N)
        N.tmp[k,1] = S
        S = N.layers[k](S,Θ[k])
    end
    return S
end

"""
compute matvec J_S N(S,Θ)'*Z
"""
function getJSTmv(N::NN,Z::AbstractArray{R},S::AbstractArray{R},Θ) where R <: Real
    for k=nLayers(N):-1:1
        N.tmp[k,2] = Z
        Z = getJSTmv(N.layers[k],Z,N.tmp[k,1],Θ[k])
    end
    return Z
end

function getGradAndHessian(N::NN,dZ::AbstractArray{R},S::AbstractArray{R},Θ) where R <: Real
    N.tmp[end,2] = dZ
    dZ, d2Z = getGradAndHessian(N.layers[end],dZ,N.tmp[end,1],Θ[end])
    # dZ  = getJSTmv(N.layers[end],dZ,N.tmp[end],Θ[end])

    for k=nLayers(N)-1:-1:1
        N.tmp[k,2] = dZ
        dZ,d2Z = getGradAndHessian(N.layers[k],dZ,d2Z,N.tmp[k,1],Θ[k])
        # dZ  = getJSTmv(N.layers[k],dZ,N.tmp[k],Θ[k])
    end
    return dZ, d2Z
end

"""
compute matvec J_S(J_S N(S,Θ)'*Z(S))

here we use product rule

J_S N(S,Θ)'*dZ + J_S(N(S,Θ)'*Zfix)
"""
function getJSJSTmv(N::NN,dZ::AbstractArray{R},S::AbstractArray{R},Θ) where R <: Real
    d2Z = getJSJSTmv(N.layers[end],dZ,N.tmp[end,1],Θ[end])
    dZ  = getJSTmv(N.layers[end],dZ,N.tmp[end,1],Θ[end])

    for k=nLayers(N)-1:-1:1
        d2Z = getJSJSTmv(N.layers[k],dZ,d2Z,N.tmp[k,1],Θ[k])
        dZ  = getJSTmv(N.layers[k],dZ,N.tmp[k,1],Θ[k])
    end
    return d2Z,dZ
end

function getJSJSTmv(N::NN,dZ::AbstractArray{R},d2Z::AbstractArray{R},S::AbstractArray{R},Θ) where R <: Real
    for k=nLayers(N):-1:1
        d2Z = getJSJSTmv(N.layers[k],dZ,d2Z,N.tmp[k,1],Θ[k])
        # if k>1
            dZ  = getJSTmv(N.layers[k],dZ,N.tmp[k,1],Θ[k])
        # end
    end
    return d2Z,dZ
end

function getHessmv(N::NN,dZ::AbstractArray{R},S::AbstractArray{R},Θ,k=1) where R <: Real
    H1 = getHessmv(N.layers[k],N.tmp[k,2],dZ,N.tmp[k,1],Θ[k])
    if k < nLayers(N)
        dZ = getJSmv(N.layers[k],dZ,N.tmp[k,1],Θ[k])
        dZ = getHessmv(N,dZ,N.tmp[k+1,1],Θ,k+1)
        dZ = getJSTmv(N.layers[k],dZ,N.tmp[k,1],Θ[k])
        return H1 + dZ
    else
        return H1
    end
end

function getDiagHess(N::NN,dZ::AbstractArray{R},S::AbstractArray{R},Θ,k=1) where R <: Real
    H1 = getDiagHess(N.layers[k],N.tmp[k,2],dZ,N.tmp[k,1],Θ[k])
    if k < nLayers(N)
        dZ = getJSmv(N.layers[k],dZ,N.tmp[k,1],Θ[k])
        H2 = getDiagHess(N,dZ,N.tmp[k+1,1],Θ,k+1)
        # sum(dZ.*getHessmv(N,dZ,N.tmp[k+1,1],Θ,k+1),dims=1)
        # dZ = getJSTmv(N.layers[k],dZ,N.tmp[k,1],Θ[k])
        return H1 + H2
    else
        return H1
    end
end

function getTraceHess(N::NN,S::AbstractArray{R},Θ) where R <: Real
    trH1,Jac = getTraceHessAndGrad(N.layers[1],N.tmp[1,2],N.tmp[1,1],Θ[1])
    trH2 = getTraceHess(N,[],Jac,N.tmp[2,1],Θ,2)
    return trH1 + trH2
end

function getTraceHess(N::NN,w,Jac::AbstractArray{R},S::AbstractArray{R},Θ,k::Int=1) where R <: Real
    # FEATURE: second input, w, is not used here.
    if k < nLayers(N)
        trH1,Jac = getTraceHessAndGrad(N.layers[k],N.tmp[k,2],Jac,N.tmp[k,1],Θ[k])
        trH2,Jac = getTraceHessAndGrad(N,[],Jac,N.tmp[k+1,1],Θ,k+1)
        return trH1 + trH2
    else
        trH1 = getTraceHess(N.layers[k],N.tmp[k,2],Jac,N.tmp[k,1],Θ[k])
        return trH1
    end
end

function getTraceHessAndGrad(N::NN,S::AbstractArray{R},Θ) where R <: Real
    trH1,Jac = getTraceHessAndGrad(N.layers[1],N.tmp[1,2],N.tmp[1,1],Θ[1])
    trH2, Jac = getTraceHessAndGrad(N,[],Jac,N.tmp[2,1],Θ,2)
    return trH1 + trH2, Jac
end

function getTraceHessAndGrad(N::NN,w,Jac::AbstractArray{R},S::AbstractArray{R},Θ,k::Int=1) where R <: Real
    # FEATURE: second input, w, is not used here.
    trH1,Jac = getTraceHessAndGrad(N.layers[k],N.tmp[k,2],Jac,N.tmp[k,1],Θ[k])
    if k < nLayers(N)
        trH2,Jac = getTraceHessAndGrad(N,[],Jac,N.tmp[k+1,1],Θ,k+1)
        return trH1 + trH2, Jac
    else
        return trH1, Jac
    end
end
