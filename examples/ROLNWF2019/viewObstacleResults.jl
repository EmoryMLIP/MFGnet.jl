using Plots
using jInv.Mesh
using Flux
using Printf
using Statistics
using LinearAlgebra
using JLD
using MFGnet
using MAT

include("../ROLNWF2019/viewers.jl")
include("../ROLNWF2019/runOMThelpers.jl")

for d=[2,10,50,100]
	for level=[1,2]
		for iter=[100,200,300,400,500]
			dir = pwd()
			fname ="Obstacle-Multilevel-d-$d-level-$level-iter-$iter"

imgdir = dir * "/" * fname *"/"
mkdir(imgdir)
println("fname=$fname")
file = fname *".jld"

res  = load(file)
settings = res["settings"]
(m,α,nTrain,R,d,mu0,sig0,mu1,sig1,domain,nTh,m,nt,T,Qheight,sigQ,muFe,muFp,Tth) = settings
println("nTrain=$nTrain")
tspan = [0.0, T]
stepper = RK4Step()

ar = x-> R.(x)
Θbest = res["Θbest"]
println("number of model parameters: $(length(MFGnet.param2vec(Θbest)))")

rho0 = Gaussian(d,sig0,mu0)
rho1 = Gaussian(d,sig1,mu1)

Q1   = Gaussian(2,sigQ,zeros(R,2),R(Qheight))
Q    = x -> vec(Q1(x[1:2,:]))


domain = [-3.0 3.0 -4.5 4.5]

M      = getRegularMesh(domain,[128, 128])
X0     = Matrix(getCellCenteredGrid(M)')
X0  = ar([X0; zeros(d-2,size(X0,2))])

Φ = getPotentialResNet(nTh, T, nTh, R)

# setup validation
rho0x = rho0(X0)
rho1x = rho1(X0)

vol  = (domain[4]-domain[3])^d
w    = (vol/M.nc) * rho0x
sum(w)

F1  = Fp(Q,rho0,rho0x,ar(muFp))
F2  = Fe(rho0,rho0x,ar(muFe))
F   = Fcomb([F1;F2])
G = Gkl(rho0,rho1,rho0x,rho1x,ar(1.0))
J = MeanFieldGame(F,G,X0,rho0,w,Φ=Φ,stepper=stepper,nt=16,α=α,tspan=tspan)

Jc  = J(Θbest)

## plots:
Qx = Q(X0)
p0 =  viewImage2D(Qx,M,aspect_ratio=:equal)
title!("Qx(x)")
savefig(p0,imgdir * "Qx-n-$(M.n[1])x$(M.n[2]).png")

p1= viewImage2D(J.rho0x,M,aspect_ratio=:equal)
title!("rho0(x)")
savefig(p1,imgdir * "rho0x-n-$(M.n[1])x$(M.n[2]).png")

p2= viewImage2D(J.G.rho1x,M,aspect_ratio=:equal)
title!("rho1(x)")
savefig(p2,imgdir * "rho1x-n-$(M.n[1])x$(M.n[2]).png")

detDy = exp.(J.UN[d+1,:])
rho1y = (J.G.rho1(J.UN[1:d,:]).*detDy)
p3= viewImage2D(rho1y ,M,aspect_ratio=:equal,clims=(minimum(J.rho0x),maximum(J.rho0x)))
# p3= viewImage2D(rho1y ,M,aspect_ratio=:equal)
title!("rho1(y).*det")
savefig(p3,imgdir * "rho1y-n-$(M.n[1])x$(M.n[2]).png")


res0 = J.rho0x - rho1y
p4 = viewImage2D(abs.(res0),M,aspect_ratio=:equal,clims=(minimum(J.rho0x),maximum(J.rho0x)))
title!("rho0x-rho1y.*det") # where's the determinant
savefig(p4,imgdir * "absDiffRho-n-$(M.n[1])x$(M.n[2]).png")

p5 = viewImage2D(detDy,M,aspect_ratio=:equal)
title!("det")
savefig(p5,imgdir * "det-n-$(M.n[1])x$(M.n[2]).png")


xc = [ J.X0; fill(0.0,1,size(J.X0,2))]
Φ0 = J.Φ(xc,Θbest)
p6 = viewImage2D(Φ0, M, aspect_ratio=:equal)
title!("Phi(x,0)")
savefig(p6,imgdir * "phi0-n-$(M.n[1])x$(M.n[2]).png")


xc = [ J.UN[1:d,:]; fill(1.0,1,size(J.X0,2))]
Φ1 = vec(J.Φ(xc,Θbest) )
p7 = viewImage2D(Φ1,M,aspect_ratio=:equal)
title!("Phi(z,1)")
savefig(p7,imgdir * "phi1-n-$(M.n[1])x$(M.n[2]).png")

deltaG = J.α[3]*MFGnet.getDeltaG(J.G,J.UN)
p8 = viewImage2D(deltaG,M,aspect_ratio=:equal)
title!("deltaG(z)")
savefig(p8,imgdir * "deltaG-n-$(M.n[1])x$(M.n[2]).png")

resHJB = abs.(Φ1-deltaG)
p9 = viewImage2D(resHJB,M,aspect_ratio=:equal)
title!("|Phi1 - deltaG(z)|")
savefig(p9,imgdir * "absDiffHJBfinal-n-$(M.n[1])x$(M.n[2]).png")

################################################################################
####### Create new MFG for characteristics
################################################################################
Xt = load("Obstacle-X0.jld")["Xt"]
if d>2
	Xt = [Xt; zeros(R,d-2,size(Xt,2))]
end
rho0xt = rho0(Xt)
rho1xt = rho1(Xt)
wt     = 1/size(Xt,2) * ones(size(Xt,2))
F1t  = Fp(Q,rho0,rho0xt,1.0)
F2t  = Fe(rho0,rho0xt,1.0)
Ft   = Fcomb([F1t;F2t])
Gt = Gkl(rho0,rho1,rho0xt,rho1xt,1.0)
Jt = MeanFieldGame(Ft,Gt,Xt,rho0,wt,Φ=Φ,stepper=RK4Step(),nt=2*nt,α=α)

ntv = Jt.nt
Ut = MFGnet.integrate2(Jt.stepper,MFGnet.odefun,Jt,ar([Xt;zeros(4,size(Xt,2))]),Θbest,ar([0.0 T]),ntv)
charFwd = copy(Ut);

Y0  = ar([Ut[1:d,:,end];zeros(4,size(Xt,2))])
U0 = MFGnet.integrate2(Jt.stepper,MFGnet.odefun,Jt,Y0,Θbest,ar([T 0.0]),ntv)
charBwd = copy(U0);

p10 = viewImage2D(J.rho0x ,M,aspect_ratio=:equal)
for j=1:size(Xt,2)
    uj = Ut[1:d,j,:]
    vj = U0[1:d,j,:]
    plot!(p10,uj[1,:],uj[2,:],legend=false,linecolor=:white,linewidth=1.5)
    plot!(p10,vj[1,:],vj[2,:],legend=false,linecolor=:red,linewidth=1.5)
end
title!("characteristics, fwd+inv")
savefig(p10,imgdir * "characteristics-n-$(M.n[1])x$(M.n[2]).png")
################################################################################
################################################################################

U0 = MFGnet.integrate(J.stepper,MFGnet.odefun,J,ar([J.X0;zeros(4,size(J.X0,2))]),Θbest,[T 0.0],ntv)
detDyInv = exp.(U0[d+1,:])
rho0z = rho0(U0[1:d,:]).*detDyInv
p11= viewImage2D(rho0z ,M,aspect_ratio=:equal)
title!("rho0(z).*detInv")
savefig(p11,imgdir * "rho0z-n-$(M.n[1])x$(M.n[2]).png")

res1 = rho1x - rho0z
p4 = viewImage2D(abs.(res1),M,aspect_ratio=:equal)
title!("| rho1(x)-rho0(z).*detInv |") # where's the determinant
savefig(p4,imgdir * "absDiffRho1-n-$(M.n[1])x$(M.n[2]).png")

X1,X2 = getFaceGrids(M)
XC = getCellCenteredGrid(M)
tt = range(0,stop=T,length=64)
X1t = t -> [Matrix(X1'); zeros(d-2,size(X1,1)); fill(t,1,size(X1,1))]
X2t = t -> [Matrix(X2'); zeros(d-2,size(X2,1)); fill(t,1,size(X2,1))]
XCt = t -> [Matrix(XC'); zeros(d-2,size(XC,1)); fill(t,1,size(XC,1))]
V1 = zeros(M.n[1]+1,M.n[2],length(tt))
V2 = zeros(M.n[1],M.n[2]+1,length(tt))
ΦOpt  = zeros(M.n[1],M.n[2],length(tt))
for k=1:length(tt)
    tk = tt[k]
    println("tk=$tk")
    Phi = J.Φ(X1t(tk),Θbest) # run fwd prop to populate N.tmp
    gradPhi = MFGnet.getGradPotential(J.Φ,X1t(tk),Θbest)
    V1[:,:,k] = reshape(-(1/J.α[1])*gradPhi[1,:],M.n[1]+1,M.n[2])
    Phi = J.Φ(X2t(tk),Θbest) # run fwd prop to populate N.tmp
    gradPhi = MFGnet.getGradPotential(J.Φ,X2t(tk),Θbest)
    V2[:,:,k] = reshape(-(1/J.α[1])*gradPhi[2,:],M.n[1],M.n[2]+1)

	ΦOpt[:,:,k] = reshape(J.Φ(XCt(tk),Θbest),M.n[1],M.n[2])
end

matwrite( fname * ".mat", Dict(
	"Qx" => Qx,
    "rho0x" => rho0x,
    "rho1x" => rho1x,
    "rho1y" => rho1y,
    "rho0z" => rho0z,
    "domain" => M.domain,
    "detDy" => detDy,
    "detDyInv" => detDyInv,
    "res0" => res0,
    "res1" => res1,
    "charFwd" => charFwd,
    "charBwd" => charBwd,
    "deltaG" => deltaG,
    "n" => M.n,
    "Phi0" => Φ0,
    "Phi1" => Φ1,
	"V1" => V1,
	"V2" => V2,
	"PhiOpt" => ΦOpt,
    "His" => res["His"]))
end

end
end
