export SingleLayer

"""
singleLayer

σ(K*s+b)

where K,b are trainable weights

"""
struct SingleLayer
end

mσ(x::AbstractArray{R}) where R<:Real = log.(exp.(x)+exp.(-x))
mdσ(x::AbstractArray{R}) where R<:Real = tanh.(x)
md2σ(x::AbstractArray{R}) where R<:Real = one(eltype(x)) .- tanh.(x).^2

"""
evaluate layer for current weights Θ=(K,b)
"""
function (N::SingleLayer)(S::AbstractArray{R},Θ::Tuple{AbstractArray{R,2},AbstractArray{R,1}}) where R <: Real
    (K,b) = Θ
    return mσ(K*S .+ b)
end

"""
compute matvec J_S N(S,Θ)'*Z
"""
function getJSTmv(N::SingleLayer,Z::AbstractArray{R},S::AbstractArray{R},Θ::Tuple{AbstractArray{R,2},AbstractArray{R,1}}) where R <: Real
    (K,b) = Θ
    return K'*(mdσ(K*S .+ b) .* Z)
end

function getJSTmv(N::SingleLayer,Z::AbstractArray{R,3},S::AbstractArray{R},Θ::Tuple{AbstractArray{R,2},AbstractArray{R,1}}) where R <: Real
    (d,nex) = size(S)
    (K,b) = Θ
    t1 = reshape(mdσ(K*S .+ b),size(K,1),1,nex)
    t2 = K'*reshape(t1 .* Z, size(K,1),:)
    return reshape(t2,size(K,2),size(Z,2),nex)
end

"""
compute hessian matvec
"""

function getTraceHessAndGrad(N::SingleLayer, w::AbstractArray{R},S::AbstractArray{R},Θ::Tuple{AbstractArray{R,2},AbstractArray{R,1}}) where R <: Real
    (d,nex) = size(S)
    (K,b) = Θ
    t1 = K*S .+ b
    Jac = reshape(mdσ(t1),size(K,1),1,nex) .* K
    return vec(sum(reshape(md2σ(t1) .* w,size(K,1),:,nex).*(K.^2),dims=(1,2))), Jac
end

function getTraceHessAndGrad(N::SingleLayer, w::AbstractArray{R},Jac::AbstractArray{R,2},S::AbstractArray{R},Θ::Tuple{AbstractArray{R,2},AbstractArray{R,1}}) where R <: Real
    (d,nex) = size(S)
    (K,b) = Θ
    t1 = K*S .+ b;
    Jac = K * Jac
    trH =  vec(sum(reshape(md2σ(t1) .* w,size(K,1),:,nex).*(Jac).^2,dims=(1,2)))
    Jac = reshape(mdσ(t1),size(K,1),1,nex) .* Jac
    return trH,Jac
end

function getTraceHessAndGrad(N::SingleLayer, w::AbstractArray{R},Jac::AbstractArray{R,3},S::AbstractArray{R},Θ::Tuple{AbstractArray{R,2},AbstractArray{R,1}}) where R <: Real
    (d,nex) = size(S)
    (K,b) = Θ
    t1 = K*S .+ b;
    Jac = reshape(K* reshape(Jac,size(K,2),:), size(K,1),:,nex)
    trH =  vec(sum(reshape(md2σ(t1) .* w,size(K,1),:,nex).*(Jac).^2,dims=(1,2)))
    Jac = reshape(mdσ(t1),size(K,1),1,nex) .* Jac
    return trH,Jac
end


function getTraceHess(N::SingleLayer, w::AbstractArray{R},S::AbstractArray{R},Θ::Tuple{AbstractArray{R,2},AbstractArray{R,1}}) where R <: Real
    (d,nex) = size(S)
    (K,b) = Θ
    t1 = K*S .+ b
    return vec(sum(reshape(md2σ(t1) .* w,size(K,1),:,nex).*(K.^2),dims=(1,2)))
end

function getTraceHess(N::SingleLayer, w::AbstractArray{R},Jac::AbstractArray{R,2},S::AbstractArray{R},Θ::Tuple{AbstractArray{R,2},AbstractArray{R,1}}) where R <: Real
    (d,nex) = size(S)
    (K,b) = Θ
    t1 = K*S .+ b;
    Jac = K * Jac
    trH =  vec(sum(reshape(md2σ(t1) .* w,size(K,1),:,nex).*(Jac).^2,dims=(1,2)))
    return trH
end

function getTraceHess(N::SingleLayer, w::AbstractArray{R},Jac::AbstractArray{R,3},S::AbstractArray{R},Θ::Tuple{AbstractArray{R,2},AbstractArray{R,1}}) where R <: Real
    (d,nex) = size(S)
    (K,b) = Θ
    t1 = K*S .+ b;
    Jac = reshape(K* reshape(Jac,size(K,2),:), size(K,1),:,nex)
    trH =  vec(sum(reshape(md2σ(t1) .* w,size(K,1),:,nex).*(Jac).^2,dims=(1,2)))
    return trH
end


function getDiagHess(N::SingleLayer, w::AbstractArray{R}, Z::AbstractArray{R},S::AbstractArray{R},Θ::Tuple{AbstractArray{R,2},AbstractArray{R,1}}) where R <: Real
    (d,nex) = size(S)
    (K,b) = Θ
    return sum((md2σ(K*S .+ b) .* w).*(K*Z).^2,dims=1)
end

function getHessmv(N::SingleLayer, w::AbstractArray{R}, Z::AbstractArray{R},S::AbstractArray{R},Θ::Tuple{AbstractArray{R,2},AbstractArray{R,1}}) where R <: Real
    (d,nex) = size(S)
    (K,b) = Θ
    return K'*(md2σ(K*S .+ b) .* w .* (K*Z))
end
function getHessmv(N::SingleLayer, w::AbstractArray{R}, Z::AbstractArray{R,3},S::AbstractArray{R},Θ::Tuple{AbstractArray{R,2},AbstractArray{R,1}}) where R <: Real
    (d,nex) = size(S)
    (K,b) = Θ
    t1 = reshape(md2σ(K*S .+ b).*w,size(K,1),1,nex)
    t2 = t1 .* reshape(K*reshape(Z,size(K,2),:),size(K,1),:,nex)
    t2 = K'* reshape(t2,size(K,1),:)
    return reshape(t2,size(K,2),size(Z,2),nex)
end



"""
compute matvec J_S N(S,Θ)*Z
"""
function getJSmv(N::SingleLayer,Z::AbstractArray{R},S::AbstractArray{R},Θ::Tuple{AbstractArray{R,2},AbstractArray{R,1}}) where R <: Real
    (K,b) = Θ
    return mdσ(K*S .+ b) .* (K * Z)
end

"""
compute matvec J_S N(S,Θ)*Z
"""
function getJSmv(N::SingleLayer,Z::AbstractArray{R,3},S::AbstractArray{R,2},Θ::Tuple{AbstractArray{R,2},AbstractArray{R,1}}) where R <: Real
    (d,nex) = size(S)
    (K,b) = Θ
    # K * Z
    KZ = K*reshape(Z,size(K,2),:)
    return reshape(mdσ(K*S .+ b),size(K,1),1,nex) .* reshape(KZ,size(K,1),size(Z,2),nex)
end


"""
compute matvec J_S(J_S N(S,Θ)'*Z(S))

here we use product rule

J_S N(S,Θ)'*dZ + J_S(N(S,Θ)'*Zfix)
"""
function getJSJSTmv(N::SingleLayer,dz::AbstractVector{R},d2z::AbstractArray{R},s::AbstractVector{R},Θ::Tuple{AbstractArray{R,2},AbstractArray{R,1}}) where R <: Real
    (K,b) = Θ
    t1 = K*s + b
    ndσ =  mdσ(t1)
    return K'* ( Diagonal(md2σ(t1) .* dz) +
                               ndσ .* d2z .* ndσ') *K
end

function getJSJSTmv(N::SingleLayer,dZ::AbstractArray{R},d2Z::AbstractArray{R},S::AbstractArray{R},Θ::Tuple{AbstractArray{R,2},AbstractArray{R,1}}) where R <: Real
    (K,b) = Θ
    t1 = K* S .+ b
    (d,nex) = size(t1)

    H1 = getJSJSTmv(N,dZ,S,Θ)

    t2 = mdσ(t1)
    H2 = reshape(t2,d,1,:) .* d2Z .* reshape(t2,1,d,:)
    s1 = K' * reshape(H2,size(K,1),:)
    s1 = permutedims(reshape(s1,size(K,2),size(K,1),nex),(2,1,3))
    s2 = K'*reshape(s1,size(K,1),:)
    return H1 + permutedims(reshape(s2,size(K,2),size(K,2),nex),(2,1,3))
end

function getJSJSTmv(N::SingleLayer,dZ::AbstractArray{R},S::AbstractArray{R,2},Θ::Tuple{AbstractArray{R,2},AbstractArray{R,1}}) where R<:Real
    (K,b) = Θ
    t1 = K* S .+ b
    (d,nex) = size(t1)

    dZ2  = reshape(dZ .* md2σ(t1),size(dZ,1),1,nex)
    KtdZK = K'*reshape(dZ2.*K,size(K,1),:)
    return reshape(KtdZK,size(K,2),size(K,2),nex)
end

function getGradAndHessian(N::SingleLayer,dZ::AbstractArray{R},S::AbstractArray{R,2},Θ::Tuple{AbstractArray{R,2},AbstractArray{R,1}}) where R<:Real
    # Here, no d2Z is give, so we assume it is zero
    (K,b) = Θ
    t1 = K* S .+ b
    (d,nex) = size(t1)

    t2  = reshape(dZ .* md2σ(t1),size(dZ,1),1,nex)
    KtdZK = K'*reshape(t2.*K,size(K,1),:)
    H =  reshape(KtdZK,size(K,2),size(K,2),nex)

    return K'*(mdσ(t1) .* dZ),H
end

function getGradAndHessian(N::SingleLayer,dZ::AbstractArray{R},d2Z::AbstractArray{R},S::AbstractArray{R},Θ::Tuple{AbstractArray{R,2},AbstractArray{R,1}}) where R <: Real
    (K,b) = Θ
    t1 = K * S .+ b
    (d,nex) = size(t1)


    t2  = reshape(dZ .* md2σ(t1),size(dZ,1),1,nex)
    KtdZK = K'*reshape(t2.*K,size(K,1),:)
    H1 =  reshape(KtdZK,size(K,2),size(K,2),nex)

    dσt = mdσ(t1)
    t3 = reshape(dσt,d,1,:) .* d2Z .* reshape(dσt,1,d,:)
    s1 = K' * reshape(t3,size(K,1),:)
    s1 = permutedims(reshape(s1,size(K,2),size(K,1),nex),(2,1,3))
    s2 = K'*reshape(s1,size(K,1),:)
    H2  =  permutedims(reshape(s2,size(K,2),size(K,2),nex),(2,1,3))
    return K'*(dσt .* dZ), H1+H2
end


function getJSJSTmv(N::SingleLayer,d2Z::AbstractArray{R,3},S::AbstractArray{R,2},Θ::Tuple{AbstractArray{R,2},AbstractArray{R,1}}) where R <: Real
    (d,nex) = size(S)
    (K,b) = Θ
    # K * Z
    t1 = reshape(mdσ(K*S .+ b),size(K,1),1,nex)
    t1 = K'*reshape(t1 .* d2Z, size(K,1),:)
    t1 = reshape(t1,size(K,2),size(d2Z,2),nex)
    permutedims!(t1,t1,(2,1,3))
    t1 = K'*reshape(t1 .* d2Z, size(K,1),:)
    t1 = reshape(t1,size(K,2),size(d2Z,2),nex)
    return t1
end


function getJSJSTmv(N::SingleLayer,d2Z::AbstractArray{R,3},S::AbstractArray{R,2},Θ::Tuple{AbstractArray{R,2},AbstractArray{R,1}},hk::R) where R <: Real
    (d,nex) = size(S)
    (K,b) = Θ
    # K * Z
    t1 = reshape(mdσ(K*S .+ b),size(K,1),1,nex)
    t1 = K'*reshape(t1 .* d2Z, size(K,1),:)
    t1 = d2Z + hk .* reshape(t1,size(K,2),size(d2Z,2),nex)
    permutedims!(t1,t1,(2,1,3))
    t1 = K'*reshape(t1 .* d2Z, size(K,1),:)
    t1 = d2Z + hk .* reshape(t1,size(K,2),size(d2Z,2),nex)
    return t1
end
