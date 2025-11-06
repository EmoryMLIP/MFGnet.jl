export PotentialNN, getGradAndHessian

"""
    PotentialNN

Neural network approximation of MFG potential function

# Formula
Φ(x,t) = w'σ(K[x;t]+b) + ½[x;t]'A[x;t] + c'[x;t] + z

# Parameters
Θ = (w, ΘN, A, c, z) where:
- w: output weights for neural network
- ΘN: parameters for neural network N
- A: quadratic term matrix (symmetrized)
- c: linear term vector
- z: scalar bias

# Fields
- `N` - underlying neural network (NN, ResNN, or SingleLayer)
- `Q` - cached projection matrix for trace computation
"""
mutable struct PotentialNN
    N
    Q
end

PotentialNN() = PotentialNN(NN(),[])
PotentialNN(N) = PotentialNN(N,[])

"""
    (Φ::PotentialNN)(XT, Θ)

Evaluate potential function Φ(x,t) = w'σ(K[x;t]+b) + ½[x;t]'A[x;t] + c'[x;t] + z

# Parameters
Θ = (w, ΘN, A, c, z) where A is symmetrized as ½(A+A')
"""
function (Φ::PotentialNN)(XT::AbstractArray{R},Θ) where R <: Real
    (w,ΘN,A,c,z) = Θ
    A = symmetrize(A)                      # Symmetrize quadratic term
    return w' *  Φ.N(XT,ΘN) + 0.5*sum((A*XT).*XT,dims=1) + c'*XT .+ z
end

"""
    getGradPotential(Φ::PotentialNN, XT, Θ)

Compute gradient ∇Φ(x,t) = J_N'w + Ax + c

Returns gradient vector for computing optimal velocity v = -∇_x Φ
"""
function getGradPotential(Φ::PotentialNN,XT::AbstractArray{R},Θ) where R <: Real
    (w,ΘN,A,c,z) = Θ
    A = symmetrize(A)
    nex = size(XT,2)
    # t1 = Φ.N(XT,ΘN) # run fwd prop to populate N.tmp
    G1 = getJSTmv(Φ.N,w,XT,ΘN)
    return G1 + A*XT .+ c
end

function getVelocity(Φ::PotentialNN,XT::AbstractArray{R},Θ) where R <: Real
    n  = size(XT,1)-1
    gradPhi = getGradPotential(Φ,XT,Θ)
    vel  = gradPhi[1:end-1,:]
    dΦdt = gradPhi[end,:]
    return vel,dΦdt
end

function getHessian(Φ::PotentialNN,XT::AbstractVector{R},Θ) where R <: Real
    (w,ΘN,A,c,z) = Θ
    A = symmetrize(A)
    # t1 = Φ.N(XT,ΘN) # run fwd prop to populate N.tmp
    H1,G = getJSJSTmv(Φ.N,w,XT,ΘN)
    return H1 .+ A, G+ A*XT .+ c
end


function getHessian(Φ::PotentialNN,XT::AbstractArray{R},Θ) where R <: Real
    nex = size(XT,2)
    (w,ΘN,A,c,z) = Θ
    A = symmetrize(A)

    # t1 = Φ.N(XT,ΘN) # run fwd prop to populate N.tmp
    H1,G = getJSJSTmv(Φ.N,w,XT,ΘN)
    return H1 .+ A, G+ A*XT .+ c
end

"""
compute gradient and Hessian of Φ w.r.t. input features
"""
function getGradAndHessian(Φ::PotentialNN,XT::AbstractArray{R},Θ) where R <: Real
    nex = size(XT,2)
    (w,ΘN,A,c,z) = Θ
    A = symmetrize(A)
    G,H = getGradAndHessian(Φ.N,w,XT,ΘN)
    return G+ A*XT .+ c, H .+ A
end


function getHessMatVec(Φ::PotentialNN,V::AbstractArray{R},XT::AbstractArray{R},Θ) where R <: Real
    H,G = getHessian(Φ,XT,Θ)
    res = zero(R)*XT
    for k=1:size(XT,2)
        res[:,k] = H[:,:,k]*V[:,k]
    end
    return res
end


function getTrace(H::AbstractArray{R}) where R <: Real
    d = size(H,1)
    q = getq(H)
    trH = q'*reshape(H,length(q),:)
    return trH
end

getQ(XT::AbstractArray{R}) where R<: Real = Array(one(R)*I,size(XT,1),size(XT,1)-1)

function getQ(Φ::PotentialNN,XT::AbstractArray{R}) where R<: Real
    d = size(XT,1)
    if isempty(Φ.Q) || size(Φ.Q,1) !== d
        Φ.Q = getQ(XT)
    end
    return Φ.Q
end


function getTraceHess(Φ::PotentialNN,XT::AbstractArray{R},Θ) where R <: Real
    (w,ΘN,A,c,z) = Θ
    A = symmetrize(A)

    d = size(XT,1)
    Q = getQ(Φ,XT)
    trH1 = getTraceHess(Φ.N,w,Q,XT,ΘN)
    return trH1 .+ tr(Q'*A*Q)
end
