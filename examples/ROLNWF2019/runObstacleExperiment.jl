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
@isdefined(nTrain)      ? nTrain=nTrain     : nTrain=16^2
@isdefined(nVal)        ? nVal=nVal         : nVal=minimum([64^2,nTrain]) # number of training samples
@isdefined(nt)          ? nt=nt             : nt=4                    # number of time steps for characteristics
@isdefined(stepper)     ? stepper=stepper   : stepper=RK4Step()       # time stepping for characteristics
@isdefined(T)           ? T=T               : T=1.0                   # final time for dynamical OT
@isdefined(alph)        ? α=alph            : α=[1.0,2.0,4.0,2.0,2.0] # weights for objective
@isdefined(sampleFreq)  ? sampleFreq = sampleFreq : sampleFreq = 25   # sample frequency
@isdefined(saveIter)    ? saveIter = saveIter     : saveIter   = 25   # iteration to save weights
@isdefined(maxIter)     ? maxIter  = maxIter      : maxIter = 200     # max number of iters
@isdefined(optim)       ? optim  = optim        : optim = :bfgs       # optimization algorithm
@isdefined(sigQ)        ? sigQ = sigQ           : sigQ = R.([1.0; 0.5]) # variance of obstacle Gaussian
@isdefined(Qheight)     ? Qheight = Qheight     : Qheight = 50.0 # height/mass of obstacle Gaussian
@isdefined(muFp)        ? muFp = muFp           : muFp = 1.0     # penalty for F preference
@isdefined(muFe)        ? muFe = muFe           : muFe = 1e-2    # penalty for F entropy
@isdefined(doPlots)     ? doPlots  = doPlots      : doPlots = true     # max number of iters

@isdefined(saveStr)     ? saveStr=saveStr   : saveStr = "Obstacle-BFGS-d-"*string(d)

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


if doPlots
    # validation points on regular grid
    domain = 5.0*[-1 1 -1 1];
    M      = getRegularMesh(domain,[sqrt(nVal),sqrt(nVal)])
    X0val     = Matrix(getCellCenteredGrid(M)')
    X0val   = ar([X0val; zeros(d-2,size(X0val,2))]); nVal = size(X0val,2)
    rho0xVal = rho0(X0val)
    rho1xVal = rho1(X0val)
    vol     = 10^d
    wVal    = (vol/nVal) * rho0xVal
else
    # validation from samples
    X0val   = sample(rho0,nVal)
    wVal    = ar(1/nVal * ones(nVal))
    println("sum(wVal)=$(sum(wVal))")
end

if norm(sum(wVal) - R(1)) > 0.1
    error("Double check the weights in validation loss.")
end

α = ar.(α)
tspan = [0.0, T]
Φ = getPotentialResNet(nTh,1.0,nTh,R)
(w0,ΘN,A0,b0,z0) = initializeWeights(d,m,nTh,ar)


rho0xv = rho0(X0val)
rho1xv = rho1(X0val)

F1v  = Fp(Q,rho0,rho0xv,R.(muFp))
F2v  = Fe(rho0,rho0xv,R.(muFe))
Fv   = Fcomb([F1v;F2v])
Gv   = Gkl(rho0,rho1,rho0xv,rho1xv,R.(1.0))
Jv  = MeanFieldGame(Fv,Gv,X0val,rho0,wVal,Φ=Φ,stepper=stepper,nt=nt,α=α,tspan=tspan)

Θ = (w0,(ΘN),A0,b0,z0)

parms = MFGnet.myMap(x->x,Θ)
ps = Flux.params(parms)

println("\n\n ---------- Obstacle Driver -------------\n\n")
println("results stored in: $(pwd()*"/"*saveStr)-level-1.jld")
println("sampleFreq = $(sampleFreq), α = $(α), nTh = $(nTh), m = $(m), nt = $(nt), Qheight = $(Qheight), sigQ=$(sigQ)")
println("DIMENSION = $(d), nTrain = $(nTrain), nVal = $(nVal), saveIter = $(saveIter), maxIter = $(maxIter), muFe=$(muFe), muFp=$(muFp)\n\n")
println("optim: $optim")

global bestLoss = Inf
global Θtemp = initializeWeights(d,m,nTh,ar)
global Θbest = initializeWeights(d,m,nTh,ar)

cb = function(J,Jv,iter,His,parms,doPlots=doPlots)
    global bestLoss
    global Θtemp
    global Θbest

    θc = parms
    # Jvc = Jv(θc)
    Jvc = Jv(parms)

    if Jvc < bestLoss
        bestLoss = Jvc
        # println("loss:", bestLoss)
        Θtemp = copy(MFGnet.param2vec(parms))
        Θbest = MFGnet.vec2param!(Θtemp,Θbest)
        # println("Θopt:", Θopt[1][1:10])
    end


    Hisk = [J.cs[1] J.cs[2] J.cs[3] J.cs[4] J.cs[5] Jv.cs[1] Jv.cs[2] Jv.cs[3] Jv.cs[4] Jv.cs[5]]
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
        settings = (m,α,nTrain,R,d,mu0,sig0,mu1,sig1,domain,nTh,m,nt,T,Qheight,sigQ,muFe,muFp)
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
        X0    = sample(rho0,size(J.X0,2))
        J.X0    = X0
        # J.X0    = [J.X0 -J.X0]
        J.rho0x = rho0(J.X0)
        J.G.rho0x = rho0(J.X0)
        J.G.rho1x = rho1(J.X0)
    end

    if doPlots
        p1= viewImage2D(Jv.rho0x,M,aspect_ratio=:equal)
        title!("rho0(x)")
        p2= viewImage2D(Jv.G.rho1x,M,aspect_ratio=:equal)
        title!("rho1(x)")
        detDy = exp.(Jv.UN[d+1,:])
        rho1y = (Jv.G.rho1(Jv.UN[1:d,:]).*detDy)
        # p3= viewImage2D(rho1y ,M,aspect_ratio=:equal,clims=(minimum(Jv.rho0x),maximum(Jv.rho0x)))
        p3= viewImage2D(rho1y ,M,aspect_ratio=:equal)
        title!("rho1(y).*det")

        res = Jv.rho0x - rho1y
        p4 = viewImage2D(abs.(res),M,aspect_ratio=:equal)
        title!("rho0(x)-rho1(y).*det") # where's the determinant

        p5 = viewImage2D(detDy,M,aspect_ratio=:equal)
        title!("det")

        xc = [ Jv.X0; fill(0.0,1,size(Jv.X0,2))]
        # Φ0 = Jv.Φ(xc,θc)
        # p6 = viewImage2D(Φ0,M,aspect_ratio=:equal)
        # title!("Phi(x,0)")


        ################
        p6 = viewImage2D(Q(Jv.X0), M, aspect_ratio=:equal)
        title!("Potential Q(x)")

        xc = ar([ Jv.UN[1:d,:]; fill(1.0,1,size(Jv.X0,2))])
        Φ1 = vec(Jv.Φ(xc,θc))
        p7 = viewImage2D(Φ1,M,aspect_ratio=:equal)
        title!("Phi(z,1)")


        deltaG = Jv.α[3]*MFGnet.getDeltaG(Jv.G,Jv.UN)
        p8 = viewImage2D(deltaG,M,aspect_ratio=:equal)
        title!("deltaG(z)")
        p9 = viewImage2D(abs.(Φ1-deltaG),M,aspect_ratio=:equal)
        title!("Phi1 - deltaG(z)")

        Xt    = sample(rho0,10)
        wt     = 1/size(Xt,2) * ones(size(Xt,2))
        rho0xt = rho0(Xt)
        rho1xt = rho1(Xt)
        F1t  = Fp(Q,rho0,rho0xt,ar(1.0))
        F2t  = Fe(rho0,rho0xt,ar(1e-1))
        Ft   = Fcomb([F1t;F2t])
        Gt = Gkl(rho0,rho1,rho0xt,rho1xt,ar(1.0))
        Jt = MeanFieldGame(Ft,Gt,Xt,rho0,wt,Φ=Φ,stepper=stepper,nt=nt,α=α,tspan=tspan)

        # p20 = viewImage2D(J.rho0x ,M,aspect_ratio=:equal)
        ntv = Jt.nt
        Ut = MFGnet.integrate2(Jt.stepper,MFGnet.odefun,Jt,ar([Xt;zeros(4,size(Xt,2))]),θc,ar(tspan),ntv)
        Y0  = ar([Ut[1:d,:,end];zeros(4,size(Xt,2))])
        # U0 = MFGnet.integrate2(Jt.stepper,MFGnet.odefun,Jt,Y0,Θopt,ar([T 0.0]),ntv)

        for j=1:size(Xt,2)
            uj = Ut[1:d,j,:]
            # vj = U0[1:d,j,:]
            plot!(p6,uj[1,:],uj[2,:],legend=false,linecolor=:white,linewidth=1.5)
            # plot!(p1,vj[1,:],vj[2,:],legend=false,linecolor=:red,linewidth=1.5)
        end



        pt = plot(p1,p2,p3,p4,p5,p6,p7,p8,p9)
        display(pt)
    end
end

for level=1:length(nTrain)
    X0    = sample(rho0,nTrain[level])
    w     = ar(1/nTrain[level] * ones(nTrain[level]))
    println("sum(w)=$(sum(w))")
    if norm(sum(w) - R(1)) > 0.1
        error("Double check the weights in training loss.")
    end
    F1  = Fp(Q,rho0(X0),rho1(X0),ar(muFp))
    F2  = Fe(rho0,rho0(X0),ar(muFe))
    F   = Fcomb([F1;F2])
    G = Gkl(rho0,rho1,rho0(X0),rho1(X0),ar(1.0))
    J = MeanFieldGame(F,G,X0,rho0,w,Φ=Φ,stepper=stepper,nt=nt,α=α,tspan=tspan)


    His = zeros(maxIter[level],10)
    cbBFGS = (iter)-> cb(J,Jv,iter,His,parms,doPlots)
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
                Xt    = sample(J.rho0,size(J.X0,2))
                J.X0  = Xt;
                J.w   = (1/nTrain) * ones(R,nTrain);
                J.G.rho0x = J.G.rho0(Xt)
                J.G.rho1x = J.G.rho1(Xt)
                J.rho0x = J.rho0(Xt)
            end

            Jc,back = Zygote.pullback(() -> J(parms), ps)
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

    settings = (m,α,nTrain[level],R,d,mu0,sig0,mu1,sig1,domain,nTh,m,nt,T,Qheight,sigQ,muFe,muFp)

    save(saveStr*"-level-$level.jld","Θopt", Θopt,"Θbest",Θbest,"His",His,"settings",settings)
end
