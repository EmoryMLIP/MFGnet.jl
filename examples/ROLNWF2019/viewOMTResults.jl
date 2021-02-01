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

# @isdefined(d)  ? d = d : d = 2
# @isdefined(level)  ? level = level : level = 2
# @isdefined(iter)  ? iter = iter : iter = 500
for d=[2]
    for level=[1,2]
        for iter=[100 200 300 400 500]

dir = pwd()

# fname = "OMT-Multilevel-nt-2-m-16-d-2-level-2"
fname = "OMT-Multilevel-noHJB-nt-8-d-$d-level-$level-iter-$iter"
imgdir = dir * "/" * fname *"/"
println("fname=$fname")
mkdir(imgdir)
file = fname * ".jld"
res = load(file)
settings = res["settings"]
(m,α,nTrain,R,d,domain,nTh,m,nt,T) = settings

stepper = RK4Step()
if !@isdefined(ar)
    ar = x-> R.(x)
end
Θbest = res["Θbest"]
println("number of model parameters: $(length(MFGnet.param2vec(Θbest)))")

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

M   = getRegularMesh(domain, [128, 128])
X0  = Matrix(getCellCenteredGrid(M)')
X0  = ar([X0; zeros(d-2,size(X0,2))])

Φ = getPotentialResNet(nTh, T, nTh, R)

# setup validation
rho0x = rho0(X0)
rho1x = rho1(X0)

vol     = (domain[2]-domain[1])^d
w         = (vol/M.nc) * rho0x
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
p4 = viewImage2D(abs.(res0),M,aspect_ratio=:equal)
title!("| rho0(x)-rho1(y).*det |") # where's the determinant
savefig(p4,imgdir * "absDiffRho-n-$(M.n[1])x$(M.n[2]).png")

p5 = viewImage2D(detDy,M,aspect_ratio=:equal)
title!("det")
savefig(p5,imgdir * "det-n-$(M.n[1])x$(M.n[2]).png")

xc = [ J.X0; fill(0.0,1,size(J.X0,2))]
Φ0 = J.Φ(xc,Θbest)
p6 = viewImage2D(Φ0,M,aspect_ratio=:equal)
title!("Phi(x,0)")
savefig(p6,imgdir * "phi0-n-$(M.n[1])x$(M.n[2]).png")


xc = [ J.UN[1:d,:]; fill(T,1,size(J.X0,2))]
Φ1z = vec(J.Φ(xc,Θbest) )
p7 = viewImage2D(Φ1z,M,aspect_ratio=:equal)
title!("Phi(z,1)")
savefig(p7,imgdir * "phi1z-n-$(M.n[1])x$(M.n[2]).png")


deltaG = J.α[3]*MFGnet.getDeltaG(J.G,J.UN)
p8 = viewImage2D(deltaG,M,aspect_ratio=:equal)
title!("deltaG(z)")
savefig(p8,imgdir * "deltaG-n-$(M.n[1])x$(M.n[2]).png")

resHJB = abs.(Φ1z-deltaG)
p9 = viewImage2D(resHJB,M,aspect_ratio=:equal)
title!("|Phi1 - deltaG(z)|")
savefig(p9,imgdir * "absDiffHJBfinal-n-$(M.n[1])x$(M.n[2]).png")


# ###### Create new MFG for characteristics
Xc     = load("OT-X0.jld")["Xt"]
if d>2
    Xc = [Xc; zeros(d-2,size(Xc,2))]
    # [3:d,:] .= R(0.0)
end
wc     = 1/size(Xc,2) * ones(size(Xc,2))
rho0xc = rho0(Xc)
rho1xc = rho1(Xc)
Fc = F0()
Gc = Gkl(rho0,rho1,rho0xc,rho1xc,1.0)
Jc = MeanFieldGame(Fc,Gc,Xc,rho0,wc,Φ=Φ,stepper=RK4Step(),nt=4*J.nt,α=J.α,tspan=J.tspan)

p10 = viewImage2D(J.rho0x ,M,aspect_ratio=:equal)
Ut = MFGnet.integrate2(Jc.stepper,MFGnet.odefun,Jc,ar([Xc;zeros(4,size(Xc,2))]),Θbest,[0.0 T],Jc.nt)
charFwd = copy(Ut);
Y0  = [Ut[1:d,:,end];zeros(4,size(Xc,2))]
U0 = MFGnet.integrate2(Jc.stepper,MFGnet.odefun,Jc,Y0,Θbest,[T 0.0],Jc.nt)
charBwd = copy(U0);
for j=1:size(Xc,2)
    uj = Ut[1:d,j,:]
    vj = U0[1:d,j,:]
    plot!(p10,uj[1,:],uj[2,:],legend=false,linecolor=:white,linewidth=1.5)
    plot!(p10,vj[1,:],vj[2,:],legend=false,linecolor=:red,linewidth=1.5)
end
title!("characteristics, fwd+inv")
savefig(p10,imgdir * "characteristics-n-$(M.n[1])x$(M.n[2]).png")


U0 = MFGnet.integrate(J.stepper,MFGnet.odefun,J,ar([J.X0;zeros(4,size(J.X0,2))]),Θbest,[T 0.0],J.nt)
detDyInv = exp.(U0[d+1,:])
rho0z = rho0(U0[1:d,:]).*detDyInv
p11= viewImage2D(rho0z ,M,aspect_ratio=:equal)
title!("rho0(z).*detInv")
savefig(p11,imgdir * "rho0z-n-$(M.n[1])x$(M.n[2]).png")

res1 = rho1x - rho0z
p4 = viewImage2D(abs.(res1),M,aspect_ratio=:equal)
title!("| rho1(x)-rho0(z).*detInv |") # where's the determinant
savefig(p4,imgdir * "absDiffRho1-n-$(M.n[1])x$(M.n[2]).png")

xc = [ J.X0[1:d,:]; fill(T,1,size(J.X0,2))]
Φ1x = vec(J.Φ(xc,Θbest) )
p12 = viewImage2D(Φ1x,M,aspect_ratio=:equal)
title!("Phi(x,1)")
savefig(p12,imgdir * "phi1-n-$(M.n[1])x$(M.n[2]).png")


X1,X2 = getFaceGrids(M)
tt = range(0,stop=T,length=64)
X1t = t -> [Matrix(X1'); zeros(d-2,size(X1,1)); fill(t,1,size(X1,1))]
X2t = t -> [Matrix(X2'); zeros(d-2,size(X2,1)); fill(t,1,size(X2,1))]
XC = getCellCenteredGrid(M)
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
using MAT
matwrite( fname * ".mat", Dict(
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
    "Phi1z" => Φ1z,
    "Phi1" => Φ1x,
    "V1" => V1,
    "V2" => V2,
    "PhiOpt" => ΦOpt,
    "His" => res["His"]))
end

end

end
