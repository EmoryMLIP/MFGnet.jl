using Test
using Revise
using LinearAlgebra
using MFGnet
using Printf

function initializeWeights(d,m,ar=x->x)
    (K1,b1) = (ar(.01*randn(m,d+1)),ar(0.1*randn(m)))
    Θ1 = (K1,b1)
    (w0,A0,c0,z0) = (ar(ones(m)/sqrt(m)),ar(zeros(d+1,d+1)),ar(zeros(d+1)),ar([1.0]))
    return (w0,Θ1,A0,c0,z0)
end

Rs = [Float64,Float32]
for k = 1:length(Rs)
    R = Rs[k]

    d = 2
    m = 10
    nex = 20;

    N = SingleLayer()
    ar(x) = R.(x)
    (w,Θ,A,c,z0) = initializeWeights(d,m,ar)

    s = randn(R,d+1,nex)

    z = N(s,Θ)
    @test eltype(z) == R
    Φ = w'*z
    ds = randn(R,size(s))
    dΦ,d2Φ = getGradAndHessian(N,w,s,Θ)
    @test eltype(dΦ) == R
    @test eltype(d2Φ) == R
    dΦds = sum(ds.*dΦ,dims=1)
    curv = zeros(R,1,nex)
    for j=1:nex
        curv[1,j] = ds[:,j]'*d2Φ[:,:,j]*ds[:,j]
    end
    curvt = MFGnet.getHessmv(N,w,ds,s,Θ)
    trH = MFGnet.getTraceHess(N,w,s,Θ)
    trHt, = MFGnet.getTraceHessAndGrad(N,w,s,Θ)
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

    @testset "adjoint and mult. dispatch test  $(Rs[k])" begin
    d2 = 2
    V = randn(R,d+1,d2,nex)
    W = randn(R,m,d2,nex)
    S = randn(R,d+1,nex)
    JTW = MFGnet.getJSTmv(N,W,S,Θ)
    @test eltype(JTW) == R
    JV = MFGnet.getJSmv(N,V,S,Θ)
    @test eltype(JV) == R

    sumJS = R(0.0)
    sumJST = R(0.0)
    for j=1:d2
        # global sumJS, sumJST
        MFGnet.getJSmv(N,V[:,j,:],S,Θ)
        sumJS   = sumJS + norm(JV[:,j,:] - MFGnet.getJSmv(N,V[:,j,:],S,Θ))
        sumJST  = sumJST +  norm(JTW[:,j,:] - MFGnet.getJSTmv(N,W[:,j,:],S,Θ))
    end
    @test sumJS <= eps(R)
    @test sumJST <= eps(R)

    # adjoint test
    vJTw = sum(V.*JTW,dims=1)
    wJv  = sum(W.*JV, dims=1)
    @test sum(vJTw - wJv) <= sqrt(eps(R))


    # test JSJSTmv
    S = randn(R,d+1,nex)
    dZ = randn(R,m,1)
    d2Z = randn(R,m,m,1)

    H1 = MFGnet.getJSJSTmv(N,dZ,S,Θ)
    @test eltype(H1)==R
    H2 = MFGnet.getJSJSTmv(N,dZ,zeros(R,m,m),S,Θ)
    @test eltype(H2)==R
    @test norm(vec(H1) - vec(H2)) <= sqrt(eps(R))
    end

end
