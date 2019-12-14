using Revise
using Test
using LinearAlgebra
using MFGnet
using Printf

d = 2
m = 8
nex = 4
nTh = 3

Rs = [Float64;Float32]

for k=1:2
    @testset "derivative check $(Rs[k])" begin
    R = Rs[k]
    L = SingleLayer()
    Φ = PotentialNN(L)
    s = randn(R,d+1,nex)
    s[d+1,:] .= R(0)
    K = randn(R,m,d+1)
    b = randn(R,m)
    ΘN = (K,b)
    w = randn(R,m)
    A = randn(R,d+1,d+1)
    c = randn(R,d+1)
	b = randn(R,1)
    Θ = (w,ΘN,A,c,b)

    Φ0 = Φ(s,Θ)
    @test eltype(Φ0) == R
    ds = randn(R,size(s))
    dΦ,d2Φ = getGradAndHessian(Φ,s,Θ)
    @test eltype(dΦ) == R
    @test eltype(d2Φ)==R
    dΦds = sum(ds.*dΦ,dims=1)

    curv = zeros(1,nex)
    for j=1:nex
        curv[1,j] = ds[:,j]'*d2Φ[:,:,j]*ds[:,j]
    end
    trH = MFGnet.getTraceHess(Φ,s,Θ)
    for j=1:nex
        @test abs(trH[j] - tr(d2Φ[1:end-1,1:end-1,j]))/abs(tr(d2Φ[1:end-1,1:end-1,j])) < sqrt(eps(R))
    end

    for j=1:15
        h = R(2.0^(-j))
        Φt = Φ(s + h .* ds,Θ)

        E0 = norm(Φ0-Φt)
        E1 = norm(Φ0+h*dΦds - Φt)
        E2 = norm(Φ0+h*dΦds+0.5*h^2*curv - Φt)

        @printf("h=%1.3e\t\tE0=%1.3e\tE1=%1.3e\tE1=%1.3e\n",h,E0,E1,E2)
    end
    @test true
    end
end
