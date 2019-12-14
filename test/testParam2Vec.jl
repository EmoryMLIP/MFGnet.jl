using Flux, Zygote
using Revise
using MFGnet
using Test

R = Float32
m = 16
d = 2
nTh = 2

(K1,b1) = (R(.1)*randn(R,m,d+1),R(.1)*randn(R,m))
Θ1 = (K1,b1)
(K2,b2) = (R(.1)*repeat(randn(R,m,m),1,1,nTh),R(.1)*repeat(randn(R,m),1,nTh))
Θ2 = (K2,b2)
ΘN = (Θ1, Θ2)
(w0,A0,c0) = (ones(R,m)/R(sqrt(m)),zeros(R,d+1,d+1),zeros(R,d+1))
Θ = (w0,ΘN,A0,c0)

As = param(copy(A0))
K1s = param(copy(K1))
K2s = param(copy(K2))
b1s = param(copy(b1))
b2s = param(copy(b2))
ws = param(copy(w0))
cs = param(copy(c0))


parms = (ws,((K1s,b1s),(K2s,b2s)),As,cs)
Θopt  = Array(MFGnet.param2vec(parms))

@test MFGnet.getParmsType(parms)==R
@test typeof(Θopt[1])==R
# println("Param2Vec and getParmsType Test Passed!")
# 