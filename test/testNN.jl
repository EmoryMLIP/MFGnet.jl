using Test
using Revise
using LinearAlgebra
using MFGnet
using Printf

d = 3
m1 = 4
m2 = 6
m3 = 8
nex = 10;
L = SingleLayer()
N = NN([L;L;L])

Rs = [Float64,Float32]

for k = 1:length(Rs)
    R = Rs[k]
    s = randn(R,d,nex)
    K1 = randn(R,m1,d)
    b1 = randn(R,m1)
    Θ1 = (K1,b1)
    K2 = randn(R,m2,m1)
    b2 = randn(R,m2)
    Θ2 = (K2,b2)
    K3 = randn(R,m3,m2)
    b3 = randn(R,m3)
    Θ3 = (K3,b3)
    w = randn(R,m3)
    Θ = (Θ1,Θ2,Θ3)

    @testset "derivative check, mult. dispatch test $(Rs[k]), and trace" begin
    z = N(s,Θ)
    @test eltype(z) == R
    Φ = w'*z
    ds = randn(R,size(s))
    dΦ,d2Φ = getGradAndHessian(N,w,s,Θ)
    @test eltype(dΦ) == R
    dΦds = sum(ds.*dΦ,dims=1)

    d2Φ2 = MFGnet.getJSJSTmv(N,w,s,Θ)
    curv = zeros(R,1,nex)
    for j=1:nex
        curv[1,j] = ds[:,j]'*d2Φ[:,:,j]*ds[:,j]
    end
    MFGnet.getJSTmv(N,w,s,Θ)
    d2Φ2dt = MFGnet.getHessmv(N,ds,s,Θ)
    curvt = sum(ds.*d2Φ2dt,dims=1)

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

    d2 = d
    W = randn(R,m3,d2,nex)
    JTW = MFGnet.getJSTmv(N,W,s,Θ)
    @test eltype(JTW) == R

    # test JSJSTmv
    dZ = w*ones(R,1,nex)
    @test eltype(d2Φ[1])==R
    @test eltype(d2Φ2[1])==R
    @test norm(vec(d2Φ) - vec(d2Φ2[1])) <= sqrt(eps(R))

end
end
