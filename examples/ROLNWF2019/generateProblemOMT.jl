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

    domain = [-5.5 5.5 -5.5 5.5]
    d = 2;

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

    M   = getRegularMesh(domain, [512, 512])
    X0  = Matrix(getCellCenteredGrid(M)')
    X0  = [X0; zeros(d-2,size(X0,2))]

    rho0x = rho0(X0)
    rho1x = rho1(X0)

    using MAT
    matwrite("OMTProblem2D.mat", Dict(
        "rho0x" => rho0x,
        "rho1x" => rho1x,
        "domain" => M.domain,
        "n" => M.n))
