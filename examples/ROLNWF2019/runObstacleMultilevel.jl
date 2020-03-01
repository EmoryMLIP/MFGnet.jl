using Flux, Zygote
using LinearAlgebra
using jInv.Mesh
using Printf
using Plots
using JLD
using Revise
using MFGnet
include("viewers.jl")
include("runOMThelpers.jl")

# set default experimental parameters
@isdefined(R)  ? R = R : R = Float64                                  # element type. Use Float32 or Float64
@isdefined(d)  ? d = d : d = 2                                        # dimension of problem
@isdefined(ar) ? ar = ar : ar = x-> R.(x)                             # function to change element type of arrays  and device (e.g., to cuArray)
@isdefined(m)           ? m=m               : m=16                   # width of network
@isdefined(nTh)         ? nTh=nTh           : nTh=2                   # number of nodes in ResNet discretization
@isdefined(nTrain)      ? nTrain=nTrain     : nTrain=[32^2 64^2 128^2]    # number of training samples
@isdefined(Tnet)      ? Tnet=Tnet     : Tnet=1.0;
@isdefined(nVal)        ? nVal=nVal         : nVal=minimum([64^2,nTrain[end]]) # number of training samples
@isdefined(nt)          ? nt=nt             : nt=4                    # number of time steps for characteristics
@isdefined(stepper)     ? stepper=stepper   : stepper=RK4Step()       # time stepping for characteristics
@isdefined(T)           ? T=T               : T=1.0                   # final time for dynamical OT
@isdefined(alph)        ? α=alph            : α=[1.0,2.0,4.0,2.0,2.0] # weights for objective
@isdefined(sampleFreq)  ? sampleFreq = sampleFreq : sampleFreq = 25   # sample frequency
@isdefined(saveIter)    ? saveIter = saveIter     : saveIter   = 25   # iteration to save weights
@isdefined(maxIter)     ? maxIter  = maxIter      : maxIter = [200 200 200]     # max number of iters
@isdefined(optim)       ? optim  = optim        : optim = :bfgs       # optimization algorithm
@isdefined(sigQ)        ? sigQ = sigQ           : sigQ = R.([1.0; 0.5]) # variance of obstacle Gaussian
@isdefined(Qheight)     ? Qheight = Qheight     : Qheight = 50.0 # height/mass of obstacle Gaussian
@isdefined(muFp)        ? muFp = muFp           : muFp = 1.0     # penalty for F preference
@isdefined(muFe)        ? muFe = muFe           : muFe = 1e-2    # penalty for F entropy

@isdefined(saveStr)     ? saveStr=saveStr   : saveStr = "Obstacle-Multilevel-d-"*string(d)

mu0  = R.([0.0;3.0;zeros(R,d-2)])
sig0 = R.(0.3*ones(d))
rho0 = Gaussian(d,sig0,mu0)

mu1  = R.([0.0;-3.0;zeros(R,d-2)])
sig1 = R.(0.3*ones(d))
rho1 = Gaussian(d,sig1,mu1)

# sigQ = R.(0.5*ones(d))
# Q1   = Gaussian(d,sigQ,zeros(R,d),R(Qheight))
Q1   = Gaussian(2,sigQ,zeros(R,2),R(Qheight))
# Q(x) = vec((sqrt((2*pi)^d * prod(sigQ))).*Q1(x[1:d,:]))
Q(x) = Q1(x[1:2,:])
# println("sigQ !")

################################################################################
# Note: validation from grid
################################################################################
# setup samples
domain = 4*[-1 1 -1 1];

################################################################################
# validation from samples
################################################################################
# nVal    = minimum([64^2,nTrain])
X0val   = sample(rho0,nVal)
wVal    = 1/nVal * ones(nVal)

################################################################################
################################################################################
Φ = getPotentialResNet(nTh,Tnet,nTh,R)
(w0,ΘN,A0,b0,z0) = initializeWeights(d,m,nTh,ar)

α = ar.(α)
tspan = [0.0, T]

rho0xv = rho0(X0val)
rho1xv = rho1(X0val)

F1v  = Fp(Q,rho0,rho0xv,R.(muFp))
F2v  = Fe(rho0,rho0xv,R.(muFe))
Fv   = Fcomb([F1v;F2v])
Gv   = Gkl(rho0,rho1,rho0xv,rho1xv,R.(1.0))
Jv  = MeanFieldGame(Fv,Gv,X0val,rho0,wVal,Φ=Φ,stepper=stepper,nt=nt,α=α,tspan=tspan)

Θ = (w0,(ΘN),A0,b0,z0)

parms = MFGnet.myMap(x->param(x),Θ)
ps = Flux.params(parms)

global bestLoss = Inf
global Θtemp = initializeWeights(d,m,nTh,ar)
global Θbest = initializeWeights(d,m,nTh,ar)

cb = function(J,Jv,level,iter,His,parms,doPlots=true)
    global bestLoss
    global Θtemp
    global Θbest

    θc = parms
    # Jvc = Jv(θc)
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
        currentSaveStr = saveStr *"-level-$level-iter"*string(iter)*".jld"
        settings = (m,α,nTrain,R,d,mu0,sig0,mu1,sig1,domain,nTh,m,nt,T,Qheight,sigQ,muFe,muFp,Tnet)
        # save(currentSaveStr, "Θ", Θopt,"settings",settings, "His",His[1:iter,:])
        save(currentSaveStr,"Θbest",Θbest,"His",His,"settings",settings)
    end

    if mod(iter,sampleFreq)==0;
        # println("resampling (symmetric)..")
        # X0      = sample(rho0,Int(nTrain/2))
        # X0      = X0 .- mu0
        # X0      = [X0 -X0]
        # X0      = X0 .+ mu0
        println("regular sampling...")
        X0    = sample(rho0,nTrain[level])
        J.X0    = X0
        # J.X0    = [J.X0 -J.X0]
        J.rho0x = rho0(J.X0)
        J.G.rho0x = rho0(J.X0)
        J.G.rho1x = rho1(J.X0)
    end
end

for level=1:length(nTrain)
    nTr = nTrain[level]
    maxIt = maxIter[level]

    X0    = sample(rho0,nTr)
    w     = ar(1/nTr * ones(nTr))
    rho0x  = rho0(X0)
    rho1x  = rho1(X0)

    F1  = Fp(Q,rho0,rho0x,ar(muFp))
    F2  = Fe(rho0,rho0x,ar(muFe))
    F   = Fcomb([F1;F2])
    G = Gkl(rho0,rho1,rho0x,rho1x,ar(1.0))
    J = MeanFieldGame(F,G,X0,rho0,w,Φ=Φ,stepper=stepper,nt=nt,α=α,tspan=tspan)

    println("\n\n ---------- Multilevel Obstacle Driver : level=$level of $(length(nTrain)) -------------\n\n")
    println("results stored in: $(pwd())/$saveStr-level-$level.jld)")
    println("sampleFreq = $(sampleFreq), α = $(α), nTh = $(nTh), Tnet = $(Tnet), m = $(m), nt = $(J.nt), Qheight = $(Qheight), sigQ=$(sigQ)")
    println("DIMENSION = $(d), nTrain = $(nTr), nVal = $(nVal), saveIter = $(saveIter), maxIter = $(maxIt), muFe=$(muFe), muFp=$(muFp)\n\n")
    println("optim: $optim")



    His = zeros(maxIt,10)
    cbBFGS = (iter)-> cb(J,Jv,level,iter,His,parms,false)
    # cbBFGS(1)

    # # Optimize

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
    println("average time per iteration: $(runtime/maxIt)")
    Θopt = MFGnet.param2vec(parms)
    Θopt = MFGnet.vec2param!(Θopt,Θ)

    settings = (m,α,nTrain,R,d,mu0,sig0,mu1,sig1,domain,nTh,m,nt,T,Qheight,sigQ,muFe,muFp,Tnet)

save(saveStr*"-level-$level.jld","Θopt", Θopt,"Θbest",Θbest,"His",His,"settings",settings)
end
