export Gaussian, GaussianMixture, sample

"""
Defines Gaussian PDF

p(x) = α/(sqrt.((2*π)^d * prod(σ) )) *  exp.(-sum(((x .- μ)./sqrt.(σ)).^2,dims=1)./2)

"""
struct Gaussian{R<:Real, A <: AbstractVector{R}}
    d::Int
    σ::A
    μ::A
    α::R
end
Gaussian(R,d::Int) = Gaussian(d,ones(R,d),zeros(R,d),one(R))
Gaussian(d::Int) = Gaussian(d,ones(d),zeros(d),1.0)
Gaussian(d::Int,σ::AbstractVector{R},μ::AbstractVector{R},α=one(R)) where R<: Real = Gaussian(d,σ,μ,α)

mean(G::Gaussian) = G.μ
std(G::Gaussian) = G.μ

function (G::Gaussian)(X::AbstractArray{R,2}) where R <: Real
    t1 = G.α./( (R(2*pi))^(G.d/2) * sqrt(prod(G.σ)))
    t2 = exp.(-sum(((X .- G.μ)./sqrt.(G.σ)).^2,dims=1)./R(2))
    return vec(t1 .* t2)
end

function sample(G::Gaussian{R,Vector{R}}, n) where R <: Real
    X = randn(R,G.d,n)
    X = sqrt.(G.σ) .* X .+ G.μ
    return X
end

struct GaussianMixture{R<:Real, A <: AbstractVector{R}}
    Gs::Array{Gaussian{R,A}}
end

function (GM::GaussianMixture{R})(X::AbstractArray{R,2}) where R <: Real
    p = GM.Gs[1](X)
    for k=2:length(GM.Gs)
        p += GM.Gs[k](X)
    end
    return p
end

function sample(GM::GaussianMixture{R,Vector{R}},n::Int) where R <: Real
    # determine proportion of samples from each Gaussian
    totalMass = zero(R)
    for k=1:length(GM.Gs)
        totalMass += GM.Gs[k].α
    end
    X = zeros(R,GM.Gs[1].d,0)
    for k=1:length(GM.Gs)-1
        X = [X sample(GM.Gs[k],Int(round(n*GM.Gs[k].α/totalMass)))]
    end
    X = [X sample(GM.Gs[end],n-size(X,2))]
    return X
end
