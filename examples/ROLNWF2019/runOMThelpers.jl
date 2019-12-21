function getPotentialResNet(nTh=2,T=1.0,nY=nTh,R=Float64)
    L = SingleLayer()
    RN = ResNN(L,R.(range(0.0,stop=T,length=nY)))
    N = NN([L;RN])
    return PotentialNN(N)
end

function initializeWeights(d,m,nTh,ar=x->x)
    (K1,b1) = (ar(0.01*randn(m,d+1)),ar(0.1*randn(m)))
    Θ1 = (K1,b1)
    (K2,b2) = (ar(0.01*repeat(randn(m,m),1,1,nTh)),ar(0.1*repeat(randn(m),1,nTh)))
    Θ2 = (K2,b2)
    ΘN = (Θ1, Θ2)
    (w0,A0,c0,z0) = (ones(m),ar(zeros(d+1,d+1)),ar(zeros(d+1)),ar(zeros(1)))
    return (w0,ΘN,A0,c0,z0)
end
