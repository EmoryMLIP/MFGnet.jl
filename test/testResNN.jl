using Revise
using Test
using LinearAlgebra
using MFGnet
using Printf


d = 2
nex = 4
nTh = 3

Rs = [Float64;Float32]

for k=1:2
    @testset "derivative check $(Rs[k])" begin
    R = Rs[k]
    L = SingleLayer()
    N = ResNN(L,Vector(range(R(0.0),stop=R(1.0),length=5)))

    s = randn(R,d,nex)
    K = randn(R,d,d,nTh)
    b = randn(R,d,nTh)
    Θ = (K,b)
    w = randn(R,d)
    z = N(s,Θ)
    @test eltype(z) == R
    Φ = w'*z
    ds = randn(R,size(s))
    dΦ,d2Φ = getGradAndHessian(N,w,s,Θ)
    @test eltype(dΦ) == R
    @test eltype(d2Φ)==R
    dΦds = sum(ds.*dΦ,dims=1)

    curv = zeros(1,nex)
    for j=1:nex
        curv[1,j] = ds[:,j]'*d2Φ[:,:,j]*ds[:,j]
    end
    trH = MFGnet.getTraceHess(N,s,Θ)
    trHt, = MFGnet.getTraceHessAndGrad(N,s,Θ)
    @test norm(trH-trHt)/norm(trH) < sqrt(eps(R))
    for j=1:nex
        @test abs(trH[j] - tr(d2Φ[:,:,j]))/abs(tr(d2Φ[:,:,j])) < sqrt(eps(R))
    end

    for j=1:15
        h = R(2.0^(-j))
        Φt = w'*N(s + h .* ds,Θ)

        E0 = norm(Φ-Φt)
        E1 = norm(Φ+h*dΦds - Φt)
        E2 = norm(Φ+h*dΦds+0.5*h^2*curv - Φt)

        @printf("h=%1.3e\t\tE0=%1.3e\tE1=%1.3e\tE1=%1.3e\n",h,E0,E1,E2)
    end
    @test true
    end
end
