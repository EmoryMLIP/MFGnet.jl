export ResNN

"""
ResNN

Residual Neural Network structure

# Type Parameter
- `R<:Real`: Numeric type for computations and time points

# Fields
- `layer::SingleLayer`: Description of the repeated layer
- `ts::Vector{R}`: Time points for residual connections
- `tmpS::Union{Tuple{},Vector{Any}}`: Temporary storage for forward pass states
- `tmpZ::Union{Tuple{},Vector{Any}}`: Temporary storage for backward pass states

# Note
Temporary storage is initialized as empty tuples and allocated during computation.
"""
mutable struct ResNN{R<:Real}
    layer::SingleLayer   # description of layer
    ts::Vector{R}      # time points
    tmpS    # Type: Union{Tuple,Vector{Any}}, but cannot be annotated due to dynamic assignment patterns
    tmpZ    # Type: Union{Tuple,Vector{Any}}, but cannot be annotated due to dynamic assignment patterns
end

ResNN(layer=SingleLayer(),ts::Vector{R}=[0.0, 0.5, 1.0]) where R<:Real =
        ResNN(layer,ts,(),())

nLayers(N::ResNN) = length(N.ts)-1

"""
    (N::ResNN{R})(S::AbstractArray{R}, Θ) -> AbstractArray{R}

Evaluate residual neural network forward pass.

# Arguments
- `S::AbstractArray{R}`: Input features
- `Θ`: Time-dependent parameters

# Returns
- Output features after all residual layers

# Throws
- `InvalidParameterError`: If input contains non-finite values
- `ArgumentError`: If input is empty or time points are invalid
"""
function (N::ResNN{R})(S::AbstractArray{R},Θ) where R <: Real
    # Input validation
    if size(S, 1) == 0 || size(S, 2) == 0
        throw(ArgumentError("Input S must be non-empty, got size $(size(S))"))
    end

    if !all(isfinite, S)
        throw(InvalidParameterError("S", "contains non-finite values (NaN or Inf)", "ResNN forward pass"))
    end

    if length(N.ts) < 2
        throw(InvalidParameterError("ts", "must have at least 2 time points, got $(length(N.ts))", "ResNN"))
    end

    T = maximum(N.ts)
	# Pre-allocate vector for better performance (avoid tuple appending)
	# Use ignore_derivatives to avoid differentiation through cache mutations
	ChainRulesCore.ignore_derivatives() do
		N.tmpS = Vector{Any}(undef, nLayers(N))
	end
    for k=1:nLayers(N)
		ChainRulesCore.ignore_derivatives() do
			N.tmpS[k] = S
		end
        hk = R(N.ts[k+1]-N.ts[k])
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
    hk = R(N.ts[end]-N.ts[end-1])
    Θk = linInter1D(N.ts[end-1],T,Θ)
    # Pre-allocate vector - mutations wrapped in ignore since tmpZ is only for internal caching
    ChainRulesCore.ignore_derivatives() do
        N.tmpZ = Vector{Any}(nothing, nLayers(N)+1)
        N.tmpZ[nLayers(N)] = Z
    end
    Z = Z .+ hk .* getJSTmv(N.layer,Z,N.tmpS[end],Θk)

    for k=nLayers(N)-1:-1:1
		ChainRulesCore.ignore_derivatives() do
			N.tmpZ[k] = Z
		end
        hk = N.ts[k+1]-N.ts[k]
        Θk = linInter1D(N.ts[k],T,Θ)
        Z +=  hk .* getJSTmv(N.layer,Z,N.tmpS[k],Θk)
    end
    return Z
end

function getJSTmv(N::ResNN{R},Z::AbstractArray{R},S::AbstractArray{R},Θ) where R <: Real
    T = maximum(N.ts)
	# Pre-allocate vector - mutations wrapped in ignore since tmpZ is only for internal caching
	ChainRulesCore.ignore_derivatives() do
		N.tmpZ = Vector{Any}(nothing, nLayers(N)+1)
		N.tmpZ[nLayers(N)+1] = 1
	end
    for k=nLayers(N):-1:1
		ChainRulesCore.ignore_derivatives() do
			N.tmpZ[k] = Z
		end
        hk = N.ts[k+1]-N.ts[k]
        Θk = linInter1D(N.ts[k],T,Θ)
        Z +=  hk .* getJSTmv(N.layer,Z,N.tmpS[k],Θk)
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
    d2Z =  hk .* getJSJSTmv(N.layer,dZ,N.tmpS[end],Θk)
    dZ  = dZ .+ hk .* getJSTmv(N.layer,dZ,N.tmpS[end],Θk)

    for k=nLayers(N)-1:-1:1
        Θk = linInter1D(N.ts[k],T,Θ)
        hk = N.ts[k+1]-N.ts[k]
        d2Z1 =  hk .* getJSJSTmv(N.layer,dZ,N.tmpS[k],Θk)
        d2Z2 = getJSTd2ZJSmv(N,d2Z, hk, N.tmpS[k],Θk)
        d2Z = d2Z1 + d2Z2
        if k>1
            dZ  += hk .* getJSTmv(N.layer,dZ,N.tmpS[k],Θk)
        end
    end
    return d2Z
end

function getJSJSTmv(N::ResNN{R},dZ::AbstractArray{R},d2Z::AbstractArray{R},S::AbstractArray{R},Θ) where R <: Real
    T = maximum(N.ts)
    for k=nLayers(N):-1:1
        Θk = linInter1D(N.ts[k],T,Θ)
        hk = N.ts[k+1]-N.ts[k]
        d2Z1 =  hk .* getJSJSTmv(N.layer,dZ,N.tmpS[k],Θk)
        d2Z2 = getJSTd2ZJSmv(N,d2Z, hk, N.tmpS[k],Θk)
        d2Z = d2Z1 + d2Z2
        if k>1
            dZ  += hk .* getJSTmv(N.layer,dZ,N.tmpS[k],Θk)
        end
    end
    return d2Z
end

function getGradAndHessian(N::ResNN{R},dZ::AbstractArray{R},S::AbstractArray{R},Θ) where R <: Real
    # Here, there is no Hessian from the following layer, so we assume it is zero
    T = maximum(N.ts)

    Θk = linInter1D(N.ts[end-1],T,Θ)
    hk = N.ts[end]-N.ts[end-1]
    ChainRulesCore.ignore_derivatives() do
        N.tmpZ = append(dZ,1)
    end
    ddZ, d2Z = getGradAndHessian(N.layer,dZ,N.tmpS[end],Θk)
    dZ  = dZ .+ hk .* ddZ
    d2Z = hk.*d2Z

    for k=nLayers(N)-1:-1:1
		ChainRulesCore.ignore_derivatives() do
			N.tmpZ = append(dZ,N.tmpZ)
		end
        Θk = linInter1D(N.ts[k],T,Θ)
        hk = N.ts[k+1]-N.ts[k]
        ddZ, d2Z1 =  getGradAndHessian(N.layer,dZ,N.tmpS[k],Θk)
        d2Z2 = getJSTd2ZJSmv(N,d2Z, hk, N.tmpS[k],Θk)
        d2Z = hk .* d2Z1 + d2Z2
        dZ  += hk .* ddZ
    end
    return dZ, d2Z
end

function getGradAndHessian(N::ResNN{R},dZ::AbstractArray{R},d2Z::AbstractArray{R},S::AbstractArray{R},Θ) where R <: Real
    T = maximum(N.ts)
    for k=nLayers(N):-1:1
        ChainRulesCore.ignore_derivatives() do
            N.tmpZ[k] = dZ
        end
        Θk = linInter1D(N.ts[k],T,Θ)
        hk = N.ts[k+1]-N.ts[k]
        ddZ,d2Z1 =   getGradAndHessian(N.layer,dZ,N.tmpS[k],Θk)
        d2Z2 = getJSTd2ZJSmv(N,d2Z, hk, N.tmpS[k],Θk)
        d2Z = hk .*d2Z1 + d2Z2
        dZ  += hk .* ddZ
    end
    return dZ,d2Z
end


function getTraceHess(N::ResNN,S::AbstractArray{R},Θ) where R <: Real
    T  = maximum(N.ts)
    Θk = linInter1D(N.ts[1],T,Θ)
    hk = N.ts[2]-N.ts[1]

    trH1,Jac = getTraceHessAndGrad(N.layer,N.tmpZ[1],N.tmpS[1],Θk)
    Jac =  Matrix{R}(I, size(Jac,1), size(Jac,2)) .+ hk .* Jac
    trH2, Jac = getTraceHessAndGrad(N,[],Jac,N.tmpS[2],Θ,2)
    return hk*trH1 + trH2
end

function getTraceHess(N::ResNN,w,Jac::AbstractArray{R},S::AbstractArray{R},Θ,k::Int=1) where R <: Real
    # FEATURE: second input, w, is not used here.
    T  = maximum(N.ts)
    Θk = linInter1D(N.ts[k],T,Θ)
    hk = N.ts[k+1]-N.ts[k]
    if k < nLayers(N)
        trH1,Jt = getTraceHessAndGrad(N.layer,N.tmpZ[k],Jac,N.tmpS[k],Θk)
        Jac = Jac + hk .* Jt
        trH2, Jac = getTraceHessAndGrad(N,[],Jac,N.tmpS[k+1],Θ,k+1)
        return hk .* trH1 + trH2, Jac
    else
        trH1 = getTraceHess(N.layer,N.tmpZ[k],Jac,N.tmpS[k],Θk)
        return hk .* trH1
    end
end

function getTraceHessAndGrad(N::ResNN,S::AbstractArray{R},Θ) where R <: Real
    T  = maximum(N.ts)
    Θk = linInter1D(N.ts[1],T,Θ)
    hk = N.ts[2]-N.ts[1]

    trH1,Jac = getTraceHessAndGrad(N.layer,N.tmpZ[1],N.tmpS[1],Θk)
    Jac =  Matrix{R}(I, size(Jac,1), size(Jac,2)) .+ hk .* Jac
    trH2, Jac = getTraceHessAndGrad(N,[],Jac,N.tmpS[2],Θ,2)
    return hk*trH1 + trH2, Jac
end

function getTraceHessAndGrad(N::ResNN,w,Jac::AbstractArray{R},S::AbstractArray{R},Θ,k::Int=1) where R <: Real
    # FEATURE: second input, w, is not used here.
    T  = maximum(N.ts)
    Θk = linInter1D(N.ts[k],T,Θ)
    hk = N.ts[k+1]-N.ts[k]
    trH1,Jt = getTraceHessAndGrad(N.layer,N.tmpZ[k],Jac,N.tmpS[k],Θk)
    Jac = Jac + hk .* Jt
    if k < nLayers(N)
        trH2, Jac = getTraceHessAndGrad(N,[],Jac,N.tmpS[k+1],Θ,k+1)
        return hk .* trH1 + trH2, Jac
    else
        return hk .* trH1, Jac
    end
end
