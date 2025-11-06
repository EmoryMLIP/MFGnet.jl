export NN

"""
    NN

Multi-layer neural network: S = Lₙ ∘ Lₙ₋₁ ∘ ... ∘ L₁(S₀)

# Architecture
Sequential composition of SingleLayer or ResNN blocks

# Fields
- `layers::Vector{Union{SingleLayer,ResNN}}` - ordered sequence of network layers
- `tmpS` - cached intermediate states for backward pass
- `tmpZ` - cached intermediate adjoint variables for gradient computation
"""
mutable struct NN
    layers::Array{Union{SingleLayer,ResNN},1}
    tmpS
	tmpZ
end

NN(layers=[SingleLayer();SingleLayer()]) = NN(layers,(),())

nLayers(N::NN) = length(N.layers)

"""
    (N::NN)(S, Θ)

Forward pass through multi-layer network

Computes S = Lₙ ∘ Lₙ₋₁ ∘ ... ∘ L₁(S₀) by sequential layer composition
"""
function (N::NN)(S::AbstractArray{R},Θ) where R <: Real
	# Pre-allocate storage for intermediate states (needed for backward pass)
	N.tmpS = Vector{typeof(S)}(undef, nLayers(N))
	for k=1: nLayers(N)
		N.tmpS[k] = S                         # Cache input to layer k
		S = N.layers[k](S,Θ[k]) :: Array{R,2} # Apply layer k
    end
    return S
end

"""
    getJSTmv(N::NN, Z, S, Θ)

Backward pass: chain rule through all layers

Computes gradient via reverse composition: Z₀ = J_L₁' ∘ J_L₂' ∘ ... ∘ J_Lₙ'(Zₙ)
"""
function getJSTmv(N::NN,Z::AbstractArray{R},S::AbstractArray{R},Θ) where R <: Real
	# Pre-allocate storage for adjoint variables at each layer
	N.tmpZ = Vector{typeof(Z)}(undef, nLayers(N))
    for k=nLayers(N):-1:1                          # Backward through layers
        N.tmpZ[k] = Z                              # Cache adjoint before layer k
        Z = getJSTmv(N.layers[k],Z,N.tmpS[k],Θ[k]) # Backprop through layer k
    end
    return Z
end

function getGradAndHessian(N::NN,dZ::AbstractArray{R},S::AbstractArray{R},Θ) where R <: Real
	# Pre-allocate vector for better performance (avoid tuple appending)
	N.tmpZ = Vector{typeof(dZ)}(undef, nLayers(N))
	N.tmpZ[end] = dZ
    dZ, d2Z = getGradAndHessian(N.layers[end],dZ,N.tmpS[end],Θ[end])
    # dZ  = getJSTmv(N.layers[end],dZ,N.tmp[end],Θ[end])

    for k=nLayers(N)-1:-1:1
        N.tmpZ[k] = dZ
        dZ,d2Z = getGradAndHessian(N.layers[k],dZ,d2Z,N.tmpS[k],Θ[k])
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
    d2Z = getJSJSTmv(N.layers[end],dZ,N.tmpS[end],Θ[end])
    dZ  = getJSTmv(N.layers[end],dZ,N.tmpS[end],Θ[end])

    for k=nLayers(N)-1:-1:1
        d2Z = getJSJSTmv(N.layers[k],dZ,d2Z,N.tmpS[k],Θ[k])
        dZ  = getJSTmv(N.layers[k],dZ,N.tmpS[k],Θ[k])
    end
    return d2Z,dZ
end

function getJSJSTmv(N::NN,dZ::AbstractArray{R},d2Z::AbstractArray{R},S::AbstractArray{R},Θ) where R <: Real
    for k=nLayers(N):-1:1
        d2Z = getJSJSTmv(N.layers[k],dZ,d2Z,N.tmpS[k],Θ[k])
        # if k>1
            dZ  = getJSTmv(N.layers[k],dZ,N.tmpS[k],Θ[k])
        # end
    end
    return d2Z,dZ
end

function getHessmv(N::NN,dZ::AbstractArray{R},S::AbstractArray{R},Θ,k=1) where R <: Real
    H1 = getHessmv(N.layers[k],N.tmpZ[k],dZ,N.tmpS[k],Θ[k])
    if k < nLayers(N)
        dZ = getJSmv(N.layers[k],dZ,N.tmpS[k],Θ[k])
        dZ = getHessmv(N,dZ,N.tmpS[k+1],Θ,k+1)
        dZ = getJSTmv(N.layers[k],dZ,N.tmpS[k],Θ[k])
        return H1 + dZ
    else
        return H1
    end
end

function getDiagHess(N::NN,dZ::AbstractArray{R},S::AbstractArray{R},Θ,k=1) where R <: Real
    H1 = getDiagHess(N.layers[k],N.tmpZ[k],dZ,N.tmpS[k],Θ[k])
    if k < nLayers(N)
        dZ = getJSmv(N.layers[k],dZ,N.tmpS[k],Θ[k])
        H2 = getDiagHess(N,dZ,N.tmpS[k+1],Θ,k+1)
        # sum(dZ.*getHessmv(N,dZ,N.tmp[k+1,1],Θ,k+1),dims=1)
        # dZ = getJSTmv(N.layers[k],dZ,N.tmp[k,1],Θ[k])
        return H1 + H2
    else
        return H1
    end
end

function getTraceHess(N::NN,S::AbstractArray{R},Θ) where R <: Real
    trH1,Jac = getTraceHessAndGrad(N.layers[1],N.tmpZ[1],N.tmpS[1],Θ[1])
    trH2 = getTraceHess(N,[],Jac,N.tmpS[2],Θ,2)
    return trH1 + trH2
end

function getTraceHess(N::NN,w,Jac::AbstractArray{R},S::AbstractArray{R},Θ,k::Int=1) where R <: Real
    # FEATURE: second input, w, is not used here.
    if k < nLayers(N)
        trH1,Jac = getTraceHessAndGrad(N.layers[k],N.tmpZ[k],Jac,N.tmpS[k],Θ[k])
        trH2,Jac = getTraceHessAndGrad(N,[],Jac,N.tmpS[k+1],Θ,k+1)
        return trH1 + trH2
    else
        trH1 = getTraceHess(N.layers[k],N.tmpZ[k],Jac,N.tmpS[k],Θ[k])
        return trH1
    end
end

function getTraceHessAndGrad(N::NN,S::AbstractArray{R},Θ) where R <: Real
    trH1,Jac = getTraceHessAndGrad(N.layers[1],N.tmpZ[1],N.tmpS[1],Θ[1])
    trH2, Jac = getTraceHessAndGrad(N,[],Jac,N.tmpS[2],Θ,2)
    return trH1 + trH2, Jac
end

function getTraceHessAndGrad(N::NN,w,Jac::AbstractArray{R},S::AbstractArray{R},Θ,k::Int=1) where R <: Real
    # FEATURE: second input, w, is not used here.
    trH1,Jac = getTraceHessAndGrad(N.layers[k],N.tmpZ[k],Jac,N.tmpS[k],Θ[k])
    if k < nLayers(N)
        trH2,Jac = getTraceHessAndGrad(N,[],Jac,N.tmpS[k+1],Θ,k+1)
        return trH1 + trH2, Jac
    else
        return trH1, Jac
    end
end
