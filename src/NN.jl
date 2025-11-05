export NN

"""
Neural Network structure

Type-parametric structure for composing layers with type-stable temporary storage.

# Type Parameter
- `R<:Real`: Numeric type for computations (typically Float32 or Float64)

# Fields
- `layers::Vector{Union{SingleLayer,ResNN}}`: Vector of network layers
- `tmpS::Union{Tuple{},Vector{Any}}`: Temporary storage for forward pass states
- `tmpZ::Union{Tuple{},Vector{Any}}`: Temporary storage for backward pass states

# Note
Temporary storage is initialized as empty tuples and allocated during computation.
The use of Vector{Any} is necessary because intermediate layer outputs may have
different dimensions, making a fully type-stable design impractical without
compile-time dimension information.
"""
mutable struct NN{R<:Real}
    layers::Vector{Union{SingleLayer,ResNN}}
    tmpS  # Type: Union{Tuple,Vector{Any}}, but cannot be annotated due to dynamic assignment patterns
	tmpZ  # Type: Union{Tuple,Vector{Any}}, but cannot be annotated due to dynamic assignment patterns
end

NN(layers=[SingleLayer();SingleLayer()]) = NN{Float64}(layers,(),())
NN{R}(layers) where R<:Real = NN{R}(layers,(),())

nLayers(N::NN) = length(N.layers)

"""
    (N::NN)(S::AbstractArray{R}, Θ) -> Array{R,2}

Evaluate neural network forward pass.

# Arguments
- `S::AbstractArray{R}`: Input features of size (d, nex)
- `Θ`: Parameters for each layer (length must equal nLayers(N))

# Returns
- Output features after all layers

# Throws
- `DimensionMismatchError`: If parameter count doesn't match layer count
- `InvalidParameterError`: If input contains non-finite values
- `ArgumentError`: If input is empty
"""
function (N::NN)(S::AbstractArray{R},Θ) where R <: Real
    # Input validation
    if size(S, 1) == 0 || size(S, 2) == 0
        throw(ArgumentError("Input S must be non-empty, got size $(size(S))"))
    end

    if !all(isfinite, S)
        throw(InvalidParameterError("S", "contains non-finite values (NaN or Inf)", "NN forward pass"))
    end

    nl = nLayers(N)
    if length(Θ) != nl
        throw(DimensionMismatchError(
            "$nl parameter sets (one per layer)",
            "$(length(Θ)) parameter sets",
            "NN forward pass"
        ))
    end

	# Pre-allocate vector for better performance (avoid tuple appending)
	# Use ignore_derivatives to avoid differentiation through cache mutations
	ChainRulesCore.ignore_derivatives() do
		N.tmpS = Vector{Any}(undef, nl)
	end
	for k=1:nl
		ChainRulesCore.ignore_derivatives() do
			N.tmpS[k] = S
		end
		S = N.layers[k](S,Θ[k]) :: Array{R,2}
    end
    return S
end

"""
compute matvec J_S N(S,Θ)'*Z
"""
function getJSTmv(N::NN,Z::AbstractArray{R},S::AbstractArray{R},Θ) where R <: Real
	# Pre-allocate vector - mutations wrapped in ignore since tmpZ is only for internal caching
	ChainRulesCore.ignore_derivatives() do
		N.tmpZ = Vector{Any}(nothing, nLayers(N))
	end
    for k=nLayers(N):-1:1
		ChainRulesCore.ignore_derivatives() do
			N.tmpZ[k] = Z
		end
        Z = getJSTmv(N.layers[k],Z,N.tmpS[k],Θ[k])
    end
    return Z
end

function getGradAndHessian(N::NN,dZ::AbstractArray{R},S::AbstractArray{R},Θ) where R <: Real
	# Pre-allocate vector - mutations wrapped in ignore since tmpZ is only for internal caching
	ChainRulesCore.ignore_derivatives() do
		N.tmpZ = Vector{Any}(nothing, nLayers(N))
		N.tmpZ[end] = dZ
	end
    dZ, d2Z = getGradAndHessian(N.layers[end],dZ,N.tmpS[end],Θ[end])
    # dZ  = getJSTmv(N.layers[end],dZ,N.tmp[end],Θ[end])

    for k=nLayers(N)-1:-1:1
		ChainRulesCore.ignore_derivatives() do
			N.tmpZ[k] = dZ
		end
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
