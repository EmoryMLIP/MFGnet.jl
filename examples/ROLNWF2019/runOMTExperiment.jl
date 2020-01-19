using Flux, Zygote
using LinearAlgebra
using jInv.Mesh
using Printf
using Plots
using JLD
using Revise
using MFGnet
include("../ROLNWF2019/viewers.jl")
include("../ROLNWF2019/runOMThelpers.jl")

# set default experimental parameters
@isdefined(R)  ? R = R : R = Float64                                  # element type. Use Float32 or Float64
@isdefined(d)  ? d = d : d = 2                                        # dimension of problem
@isdefined(ar) ? ar = ar : ar = x-> R.(x)                             # function to change element type of arrays  and device (e.g., to cuArray)
@isdefined(m)           ? m=m               : m=16                    # width of network
@isdefined(nTh)         ? nTh=nTh           : nTh=2                   # number of nodes in ResNet discretization
@isdefined(nTrain)      ? nTrain=nTrain     : nTrain=64^2             # number of training samples
@isdefined(nVal)        ? nVal=nVal         : nVal=minimum([64^2,nTrain]) # number of validation samples
@isdefined(nt)          ? nt=nt             : nt=2                    # number of time steps for characteristics
@isdefined(stepper)     ? stepper=stepper   : stepper=RK4Step()       # time stepping for characteristics
@isdefined(T)           ? T=T               : T=1.0                   # final time for dynamical OT
@isdefined(alph)        ? α=alph             : α=[1.0;1.0;5.0;1.0;5.0] # weights for objective
@isdefined(sampleFreq)  ? sampleFreq = sampleFreq : sampleFreq = 25   # sample frequency
@isdefined(saveIter)    ? saveIter = saveIter     : saveIter   = 25   # iteration to save weights
@isdefined(maxIter)     ? maxIter  = maxIter      : maxIter = 200     # max number of iters
@isdefined(optim)       ? optim  = optim      : optim = :bfgs     # max number of iters

@isdefined(saveStr)     ? saveStr=saveStr   : saveStr = "OMT-BFGS-d-"*string(d)

sig = ar(.3*ones(R,d))
    rho1 = Gaussian(d,sig,zeros(R,d),R(1.0))
Gs = Array{Gaussian{R,Vector{R}}}(undef,8)
ang = range(0,stop=2*pi, length=length(Gs)+1)[1:end-1]
for k=1:length(ang)
    μk = ar(4*([cos(ang[k]); sin(ang[k]);zeros(d-2)]))
    Gs[k] = Gaussian(d,sig,μk,R(1.0/length(Gs)))
end
rho0 = GaussianMixture(Gs)

################################################################################
# Note: validation from grid
################################################################################
# setup samples
domain = 5.0*[-1 1 -1 1];
# M      = getRegularMesh(domain,[sqrt(nTrain),sqrt(nTrain)])
# X0val     = Matrix(getCellCenteredGrid(M)')
# X0val   = ar([X0val; zeros(d-2,size(X0val,2))]); nVal = size(X0val,2)
# rho0xVal = rho0(X0val)
# rho1xVal = rho1(X0val)
#
# vol     = 10^d
# wVal    = (vol/nVal) * rho0xVal
# sum(wVal)

################################################################################
# validation from samples
################################################################################
# nVal    = minimum([64^2,nTrain])
X0val   = sample(rho0,nVal)
wVal    = ar(1/nVal * ones(nVal))
println("sum(wVal)=$(sum(wVal))")
if norm(sum(wVal) - R(1)) > 0.1
    error("Double check the weights in validation loss.")
end
################################################################################
################################################################################
Φ = getPotentialResNet(nTh,1.0,nTh,R)
(w0,ΘN,A0,b0,z0) = initializeWeights(d,m,nTh,ar)
X0    = sample(rho0,nTrain)
w     = ar(1/nTrain * ones(nTrain))
println("sum(w)=$(sum(w))")
if norm(sum(w) - R(1)) > 0.1
    error("Double check the weights in training loss.")
end

α = R.(α)
tspan = R.([0.0, T])

Fv = F0()
Gv   = Gkl(rho0,rho1,rho0(X0val),rho1(X0val),R.(1.0))
Jv  = MeanFieldGame(Fv,Gv,X0val,rho0,wVal,Φ=Φ,stepper=stepper,nt=nt,α=α,tspan=tspan)

F = F0()
G = Gkl(rho0,rho1,rho0(X0),rho1(X0),ar(1.0))
J = MeanFieldGame(F,G,X0,rho0,w,Φ=Φ,stepper=stepper,nt=nt,α=α,tspan=tspan)

Θ = (w0,(ΘN),A0,b0,z0)

parms = MFGnet.myMap(x->param(x),Θ)
ps = Flux.params(parms)

println("\n\n ---------- BFGS OMT Driver -------------\n\n")
println("results stored in: $(pwd()*"/"*saveStr).jld")
println("sampleFreq = $(sampleFreq), α = $(α), nTh = $(nTh), m = $(m), nt = $(J.nt)")
println("DIMENSION = $(d), nTrain = $(nTrain), nVal = $(nVal), saveIter = $(saveIter), maxIter = $(maxIter)\n\n")
println("optim: $optim")

global bestLoss = Inf
global Θtemp = initializeWeights(d,m,nTh,ar)
global Θbest = initializeWeights(d,m,nTh,ar)

cb = function(J,Jv,iter,His,parms,doPlots=true)

    global bestLoss,  Θtemp,  Θbest

    θc = parms
    Jvc = Jv(parms)

    if Jvc < bestLoss
        bestLoss = Jvc.data
        # println("loss:", bestLoss)
        Θtemp = copy(MFGnet.param2vec(parms))
        Θbest = MFGnet.vec2param!(Θtemp,Θbest)
        # println("Θopt:", Θopt[1][1:10])
    end

    Hisk = [J.cs[1].data J.cs[2].data J.cs[3].data J.cs[4].data J.cs[5].data Jv.cs[1].data Jv.cs[2].data Jv.cs[3].data Jv.cs[4].data Jv.cs[5].data]
    His[iter,:] = Hisk

    if iter==1
        str1 = @sprintf("iter\tobj\t\tcostL\t\tcostF\t\tcostG\t\tcostHJ\t\tcostHJfinal\t\t\tobjVal\t\tcostLVal\tcostFVal\tcostGVal\tcostHJVal\tcostHJfinalVal\n")
        str2 = @sprintf("%05d\t%1.3e\t%1.3e\t%1.3e\t%1.3e\t%1.3e\t%1.3e\t\t\t%1.3e\t%1.3e\t%1.3e\t%1.3e\t%1.3e\t%1.3e",
                iter,sum(His[iter,1:5]),His[iter,1],His[iter,2],His[iter,3],His[iter,4],His[iter,5],
                sum(His[iter,6:10]),His[iter,6],His[iter,7],His[iter,8],His[iter,9],His[iter,10])
        str = str1*str2
    else
        str = @sprintf("%05d\t%1.3e\t%1.3e\t%1.3e\t%1.3e\t%1.3e\t%1.3e\t\t\t%1.3e\t%1.3e\t%1.3e\t%1.3e\t%1.3e\t%1.3e",
                iter,sum(His[iter,1:5]),His[iter,1],His[iter,2],His[iter,3],His[iter,4],His[iter,5],
                sum(His[iter,6:10]),His[iter,6],His[iter,7],His[iter,8],His[iter,9],His[iter,10])
    end
    println(str)

    # save every saveIter iterations
    if mod(iter,saveIter)==0
        println("saving...")
        currentSaveStr = saveStr *"-iter"*string(iter)*".jld"
        settings = (m,α,nTrain,R,d,domain,nTh,m,nt,T)
        save(currentSaveStr,"Θbest",Θbest,"His",His,"settings",settings)
    end

    if mod(iter,sampleFreq)==0;
        println("resampling..")
        J.X0    = sample(rho0,nTrain)
        J.rho0x = rho0(J.X0)
        J.G.rho0x = rho0(J.X0)
        J.G.rho1x = rho1(J.X0)
    end

    if doPlots
        p1= viewImage2D(Jv.rho0x,M,aspect_ratio=:equal)
        title!("rho0(x)")
        p2= viewImage2D(Jv.G.rho1x,M,aspect_ratio=:equal)
        title!("rho1(x)")
        detDy = exp.(Jv.UN[d+1,:]).data
        rho1y = (Jv.G.rho1(Jv.UN[1:d,:]).*detDy).data
        p3= viewImage2D(rho1y ,M,aspect_ratio=:equal,clims=(minimum(Jv.rho0x),maximum(Jv.rho0x)))
        # p3= viewImage2D(rho1y ,M,aspect_ratio=:equal)
        title!("rho1(y).*det")

        res = Jv.rho0x - rho1y
        p4 = viewImage2D(log10.(abs.(res)),M,aspect_ratio=:equal)
        title!("rho0(x)-rho1(y).*det") # where's the determinant

        p5 = viewImage2D(detDy,M,aspect_ratio=:equal)
        title!("det")

        xc = [ Jv.X0; fill(0.0,1,size(Jv.X0,2))]
        Φ0 = Jv.Φ(xc,θc).data
        p6 = viewImage2D(Φ0,M,aspect_ratio=:equal)
        title!("Phi(x,0)")


        ################
        # p6 = viewImage2D(Q(Jv.X0), M, aspect_ratio=:equal)
        # title!("Potential Q(x)")

        xc = ar([ Jv.UN[1:d,:].data; fill(1.0,1,size(Jv.X0,2))])
        Φ1 = vec(Jv.Φ(xc,θc).data)
        p7 = viewImage2D(Φ1,M,aspect_ratio=:equal)
        title!("Phi(z,1)")


        deltaG = Jv.α[3]*MFGnet.getDeltaG(Jv.G,Jv.UN).data
        p8 = viewImage2D(deltaG,M,aspect_ratio=:equal)
        title!("deltaG(z)")
        p9 = viewImage2D(log10.(abs.(Φ1-deltaG)),M,aspect_ratio=:equal)
        title!("Phi1 - deltaG(z)")

        Xt    = sample(rho0,10)
        if d>2
            Xt[3:d,:] .= R(0.0)
        end
        wt     = 1/size(Xt,2) * ones(size(Xt,2))
        rho0xt = rho0(Xt)
        rho1xt = rho1(Xt)
        Ft = F0()
        Gt = Gkl(rho0,rho1,rho0xt,rho1xt,ar(1.0))
        Jt = MeanFieldGame(Ft,Gt,Xt,rho0,wt,Φ=Φ,stepper=stepper,nt=nt,α=α,tspan=tspan)
        ntv = Jt.nt
        Ut = MFGnet.integrate2(Jt.stepper,MFGnet.odefun,Jt,ar([Xt;zeros(4,size(Xt,2))]),θc,ar(tspan),ntv)
        # Y0  = ar([Ut[1:d,:,end];zeros(4,size(Xt,2))])
        # U0 = MFGnet.integrate2(Jt.stepper,MFGnet.odefun,Jt,Y0,Θopt,ar([T 0.0]),ntv)

        for j=1:size(Xt,2)
            uj = Ut[1:d,j,:]
            # vj = U0[1:d,j,:]
            plot!(p1,uj[1,:],uj[2,:],legend=false,linecolor=:white,linewidth=1.5)
            # plot!(p1,vj[1,:],vj[2,:],legend=false,linecolor=:red,linewidth=1.5)
        end

        pt = plot(p1,p2,p3,p4,p5,p6,p7,p8,p9)
        display(pt)
    end
end

His = zeros(maxIter,10)
cbBFGS = (iter)-> cb(J,Jv,iter,His,parms,false)
# cbBFGS(1)

if optim==:bfgs
    f   =(Θ)-> evalObj(J,Θ,parms,ps)
    fdf =(Θ)-> evalObjAndGrad(J,Θ,parms,ps)
    Θ0 = MFGnet.param2vec(parms)
    f(Θ0)
    fdf(Θ0)
    runtime = @elapsed Θopt,flag,His,X,H = MFGnet.bfgs(f,fdf,Θ0,maxIter=size(His,1),
                                               out=0,atol=1e-10,cb=cbBFGS)

else
    println("using ADAM")
    opt = ADAM()
    runtime = @elapsed for k=1:maxIter
        if mod(k,sampleFreq)==0
            Xt    = sample(J.rho0,nTrain)
            w     = (1/nTrain) * ones(R,nTrain)
            J.X0  = Xt;
            J.w   = w;
            J.G.rho0x = J.G.rho0(Xt)
            J.G.rho1x = J.G.rho1(Xt)
            J.rho0x = J.rho0(Xt)
        end

        Jc,back = Zygote.Tracker.forward(() -> J(parms), ps)
        grads = back(1)
        ng = 0.0
        for p in ps
            ng += norm(grads[p]).^2
            Zygote.Tracker.update!(opt,p, grads[p])
        end
        cbBFGS(k)
    end
end
println("average time per iteration: $(runtime/maxIter)")
Θopt = MFGnet.param2vec(parms)
Θopt = MFGnet.vec2param!(Θopt,Θ)

settings = (m,α,nTrain,R,d,domain,nTh,m,nt,T)

save(saveStr*".jld","Θopt", Θopt,"Θbest",Θbest,"His",His,"settings",settings)
