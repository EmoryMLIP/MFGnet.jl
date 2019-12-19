using Plots
using jInv.Mesh
using Flux
using Printf
using Statistics
using LinearAlgebra
using JLD
using MFGnet
include("viewers.jl")
include("runOMThelpers.jl")

iter        = 500
d           = 100
nSamples    = 192

file = "OMT-BFGS-d-"*string(d)*"-nSamples-"*string(nSamples)*"-iter"*string(iter)*".jld"

res  = load(file)
settings = res["settings"]
(m,α,nTrain,R,d,domain,nTh,m,nt,T) = settings

His = res["His"]

objTrain = sum(His[:,1:5],dims=2)
objVal   = sum(His[:,6:10],dims=2)

costLTrain          = His[:,1]

costFTrain          = His[:,2]
costGTrain          = His[:,3]
costHJBTrain        = His[:,4]
costHJBFinalTrain   = His[:,5]

costLVal            = His[:,6]
costFVal            = His[:,7]
costGVal            = His[:,8]
costHJBVal          = His[:,9]
costHJBFinalVal     = His[:,10]

costLTrain = costLTrain[1:iter]
costFTrain = costFTrain[1:iter]
costGTrain = costGTrain[1:iter]
costHJBTrain = costHJBTrain[1:iter]
costHJBFinalTrain = costHJBFinalTrain[1:iter]

costLVal            = costLVal[1:iter]
costFVal            = costFVal[1:iter]
costGVal            = costGVal[1:iter]
costHJBVal          = costHJBVal[1:iter]
costHJBFinalVal     = costHJBFinalVal[1:iter]

objTrain = objTrain[1:iter]
objVal  = objVal[1:iter]

stepper = RK4Step()

if !@isdefined(ar)
    ar = x-> R.(x)
end
Θbest = res["Θbest"]

# rho1 = Gaussian(d)
Gs = Array{Gaussian{R,Vector{R}}}(undef,8)
ang = range(0,stop=2*pi, length=length(Gs)+1)[1:end-1]
sig = R.(.3*ones(d))
for k=1:length(ang)
    μk = R.(4*([cos(ang[k]); sin(ang[k]);zeros(d-2)]))
    Gs[k] = Gaussian(d,sig,μk,R(1.0/length(Gs)))
end
rho0 = GaussianMixture(Gs)
rho1 = Gaussian(d,sig,zeros(d),1.0)

M      = getRegularMesh(domain, [64, 64])
X0     = Matrix(getCellCenteredGrid(M)')

X0  = ar([X0; zeros(d-2,size(X0,2))])

Φ = getPotentialResNet(nTh, T, nTh, R)
Θ0 = initializeWeights(d,m,nTh)

# setup validation
rho0x = rho0(X0)
rho1x = rho1(X0)

vol     = (domain[2]-domain[1])^d
w         = (vol/64^2) * rho0x
sum(w)


F = F0()
G = Gkl(rho0,rho1,rho0x,rho1x,R.(1.0))
J = MeanFieldGame(F,G,X0,rho0,w,Φ=Φ,stepper=stepper,nt=nt,α=α)

J.rho0x = rho0(J.X0)
J.G.rho0x = rho0(J.X0)
J.G.rho1x = rho1(J.X0)

Jc  = J(Θbest)

## plots:
p1= viewImage2D(J.rho0x,M,aspect_ratio=:equal)
title!("rho0(x)")
p2= viewImage2D(J.G.rho1x,M,aspect_ratio=:equal)
title!("rho1(x)")
detDy = exp.(J.UN[d+1,:])
rho1y = (J.G.rho1(J.UN[1:d,:]).*detDy)
p3= viewImage2D(rho1y ,M,aspect_ratio=:equal,clims=(minimum(J.rho0x),maximum(J.rho0x)))
# p3= viewImage2D(rho1y ,M,aspect_ratio=:equal)
title!("rho1(y).*det")

diff = J.rho0x - rho1y
p4 = viewImage2D(abs.(diff),M,aspect_ratio=:equal,clims=(minimum(J.rho0x),maximum(J.rho0x)))
# p4 = viewImage2D(abs.(res),M,aspect_ratio=:equal)
title!("rho0x-rho1y.*det") # where's the determinant

p5 = viewImage2D(detDy,M,aspect_ratio=:equal)
title!("det")

p6 = plot(log.(objTrain),linewidth=3,legend=false)
plot!(p6, log.(objVal),linewidth=3,legend=false)
title!("log objective values")

xc = [ J.UN[1:d,:]; fill(1.0,1,size(J.X0,2))]
Φ1 = vec(J.Φ(xc,Θbest) )
p7 = viewImage2D(Φ1,M,aspect_ratio=:equal)
title!("Phi(z,1)")


deltaG = J.α[3]*MFGnet.getDeltaG(J.G,J.UN)
p8 = viewImage2D(deltaG,M,aspect_ratio=:equal)
title!("deltaG(z)")

p9 = viewImage2D(abs.(Φ1-deltaG),M,aspect_ratio=:equal)
title!("Phi1 - deltaG(z)")


# ###### Create new MFG for characteristics
Xt    = sample(rho0,10)
if d>2
    Xt[3:d,:] .= R(0.0)
end
wt     = 1/size(Xt,2) * ones(size(Xt,2)); sum(wt)
rho0xt = rho0(Xt)
rho1xt = rho1(Xt)

Ft = F0()
Gt = Gkl(rho0,rho1,rho0xt,rho1xt,1.0)
Jt = MeanFieldGame(Ft,Gt,Xt,rho0,wt,Φ=Φ,stepper=stepper,nt=16,α=α)

ntv = Jt.nt
Ut = MFGnet.integrate2(Jt.stepper,MFGnet.odefun,Jt,ar([Xt;zeros(4,size(Xt,2))]),Θbest,ar([0.0 T]),ntv)
Y0  = ar([Ut[1:d,:,end];zeros(4,size(Xt,2))])
U0 = MFGnet.integrate2(Jt.stepper,MFGnet.odefun,Jt,Y0,Θbest,ar([T 0.0]),ntv)

for j=1:size(Xt,2)
    uj = Ut[1:d,j,:]
    vj = U0[1:d,j,:]
    plot!(p1,uj[1,:],uj[2,:],legend=false,linecolor=:white,linewidth=0.4)
    plot!(p1,vj[1,:],vj[2,:],legend=false,linecolor=:red,linewidth=0.4)
end
# title!("characteristics, fwd+inv")


pt = plot(p1,p2,p3,p4,p5,p6,p7,p8,p9)
display(pt)




## plot histories
p1 = plot(log.(abs.(objTrain)), linewidth=3,legend=false)
plot!(p1, log.(abs.(objVal)),linewidth=3,legend=false)
title!("log obj")

p2 = plot(log.((costLTrain)), linewidth=3,legend=false)
plot!(p2,log.((costLVal)), linewidth=3,legend=false)
title!("log L cost")

p3 = plot(((costFTrain)), linewidth=3,legend=false)
plot!(p3,((costFVal)), linewidth=3,legend=false)
title!("F cost")

p4 = plot(log.(abs.(costGTrain)), linewidth=3,legend=false)
plot!(p4,log.(abs.(costGVal)), linewidth=3,legend=false)
title!("G cost")

p5 = plot(log.((costHJBTrain)), linewidth=3,legend=false)
plot!(p5,log.((costHJBVal)), linewidth=3,legend=false)
title!("log HJB cost")

p6 = plot(log.((costHJBFinalTrain)), linewidth=3,legend=false)
plot!(p6,log.((costHJBFinalVal)), linewidth=3,legend=false)
title!("log HJB Final cost")

pt = plot(p1,p2,p3,p4,p5,p6)
display(pt)
