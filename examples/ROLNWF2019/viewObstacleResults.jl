using Plots
using jInv.Mesh
using Flux
using Printf
using Statistics
using LinearAlgebra
using JLD
using MFGnet
include("../examples/ROLNWF2019/viewers.jl")
include("../examples/ROLNWF2019/runOMThelpers.jl")

iter = 500
d   = 2

file = "Obstacle-BFGS-d-"*string(d)*"-iter"*string(iter)*".jld"
res  = load(file)
settings = res["settings"]
(m,α,nTrain,R,d,mu0,sig0,mu1,sig1,domain,nTh,m,nt,T,Qheight,sigQ,muFe,muFp) = settings

tspan = [0.0, T]
stepper = RK4Step()

ar = x-> R.(x)
Θopt = res["Θbest"]

rho0 = Gaussian(d,sig0,mu0)
rho1 = Gaussian(d,sig1,mu1)

Q1   = Gaussian(2,sigQ,zeros(R,2),R(Qheight))
Q(x) = vec(Q1(x[1:2,:]))


M      = getRegularMesh(domain,[64, 64])
X0     = Matrix(getCellCenteredGrid(M)')
X0  = ar([X0; zeros(d-2,size(X0,2))])

Φ = getPotentialResNet(nTh, T, nTh, R)

# setup validation
rho0x = rho0(X0)
rho1x = rho1(X0)

vol  = (domain[4]-domain[3])^d
w    = (vol/64^2) * rho0x
sum(w)

F1  = Fp(Q,rho0,rho0x,ar(muFp))
F2  = Fe(rho0,rho0x,ar(muFe))
F   = Fcomb([F1;F2])
G = Gkl(rho0,rho1,rho0x,rho1x,ar(1.0))
J = MeanFieldGame(F,G,X0,rho0,w,Φ=Φ,stepper=stepper,nt=16,α=α,tspan=tspan)

Jc  = J(Θopt)

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

residual = J.rho0x - rho1y
p4 = viewImage2D(abs.(residual),M,aspect_ratio=:equal,clims=(minimum(J.rho0x),maximum(J.rho0x)))
# p4 = viewImage2D(abs.(residual),M,aspect_ratio=:equal)
title!("rho0x-rho1y.*det") 

p5 = viewImage2D(detDy,M,aspect_ratio=:equal)
title!("det")

xc = [ J.X0; fill(0.0,1,size(J.X0,2))]

p6 = viewImage2D(Q(J.X0), M, aspect_ratio=:equal)
title!("Potential Q(x)")

xc = [ J.UN[1:d,:]; fill(1.0,1,size(J.X0,2))]
Φ1 = vec(J.Φ(xc,Θopt) )
p7 = viewImage2D(Φ1,M,aspect_ratio=:equal)
title!("Phi(z,1)")


deltaG = J.α[3]*MFGnet.getDeltaG(J.G,J.UN)
p8 = viewImage2D(deltaG,M,aspect_ratio=:equal)
# p8 = viewImage2D(deltaG,M,aspect_ratio=:equal,clims=(minimum(Φ1),maximum(Φ1)))
title!("deltaG(z)")
p9 = viewImage2D(log10.(abs.(Φ1-deltaG)),M,aspect_ratio=:equal)
title!("log Phi1 - deltaG(z)")


################################################################################
####### Create new MFG for characteristics
################################################################################
Xt    = sample(rho0,5)
if d>2
    Xt[3:d,:] .= R(0.0)
end
Xt      = Xt .- mu0
e     = ones(d); e[1] = -1
Xt      = [Xt e.*Xt]
Xt      = Xt .+ mu0
rho0xt = rho0(Xt)
rho1xt = rho1(Xt)
wt     = 1/size(Xt,2) * ones(size(Xt,2))
F1t  = Fp(Q,rho0,rho0xt,1.0)
F2t  = Fe(rho0,rho0xt,1.0)
Ft   = Fcomb([F1t;F2t])
Gt = Gkl(rho0,rho1,rho0xt,rho1xt,1.0)
Jt = MeanFieldGame(Ft,Gt,Xt,rho0,wt,Φ=Φ,stepper=RK4Step(),nt=nt,α=α)

ntv = 16
Ut = MFGnet.integrate2(Jt.stepper,MFGnet.odefun,Jt,ar([Xt;zeros(4,size(Xt,2))]),Θopt,ar([0.0 T]),ntv)
Y0  = ar([Ut[1:d,:,end];zeros(4,size(Xt,2))])
U0 = MFGnet.integrate2(Jt.stepper,MFGnet.odefun,Jt,Y0,Θopt,ar([T 0.0]),ntv)

for j=1:size(Xt,2)
    uj = Ut[1:d,j,:]
    vj = U0[1:d,j,:]
    plot!(p6,uj[1,:],uj[2,:],legend=false,linecolor=:white,linewidth=1.5)
    plot!(p6,vj[1,:],vj[2,:],legend=false,linecolor=:red,linewidth=1.5)
end
title!("characteristics, fwd+inv")
################################################################################
################################################################################


pt = plot(p1,p2,p3,p4,p5,p6,p7,p8,p9)
display(pt)


################################################################################
################################################################################
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

## plot histories
p1 = plot(log.(abs.(objTrain)), linewidth=3,legend=false)
plot!(p1, log.(abs.(objVal)),linewidth=3,legend=false)
title!("log obj")

p2 = plot(log.((costLTrain)), linewidth=3,legend=false)
plot!(p2,log.((costLVal)), linewidth=3,legend=false)
title!("log L")

p3 = plot(log.(abs.(costFTrain)), linewidth=3,legend=false)
plot!(p3,log.(abs.(costFVal)), linewidth=3,legend=false)
title!("log |F|")

p4 = plot(log.(abs.(costGTrain)), linewidth=3,legend=false)
plot!(p4,log.(abs.(costGVal)), linewidth=3,legend=false)
title!("log G")

p5 = plot(log.((costHJBTrain)), linewidth=3,legend=false)
plot!(p5,log.((costHJBVal)), linewidth=3,legend=false)
title!("log HJB")

p6 = plot(log.((costHJBFinalTrain)), linewidth=3,legend=false)
plot!(p6,log.((costHJBFinalVal)), linewidth=3,legend=false)
title!("log HJB Final")

pt = plot(p1,p2,p3,p4,p5,p6)
display(pt)
