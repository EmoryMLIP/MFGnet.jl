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

d   = 2

mu0 = [0.0;  3.0]
mu1 = [0.0; -3.0]
sig0 = [0.3;0.3];
sig1 = [0.3;0.3];
Qheight = 50;
sigQ = [1.0;0.5];
domain = [-3.0 3 -4.5 4.5];

rho0 = Gaussian(d,sig0,mu0)
rho1 = Gaussian(d,sig1,mu1)
Q    = Gaussian(2,sigQ,zeros(R,2),R(Qheight))


M      = getRegularMesh(domain,[512, 512])
X0     = Matrix(getCellCenteredGrid(M)')

# setup validation
rho0x = rho0(X0)
rho1x = rho1(X0)
Qx    = Q(X0)

matwrite("ObstacleProblem2D.mat", Dict(
	"Qx" => Qx,
    "rho0x" => rho0x,
    "rho1x" => rho1x,
    "domain" => M.domain,
    "n" => M.n))
