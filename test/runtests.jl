using Test
using LinearAlgebra
using Printf
using MFGnet

@testset "NN" begin
    include("testNN.jl")
end

@testset "param2vec" begin
    include("testParam2Vec.jl")
end

@testset "PotentialNN" begin
    include("testPotentialNN.jl")
end

@testset "PotentialResNN" begin
    include("testPotentialResNN.jl")
end

@testset "PotentialSingle" begin
    include("testPotentialSingle.jl")
end

@testset "ResNN" begin
    include("testResNN.jl")
end

@testset "singleLayer" begin
    include("testSingleLayer.jl")
end

@testset "linInter1D" begin
    include("testLinInter1D.jl")
end

# Phase 2 additions: Tests for previously untested modules

@testset "MeanFieldGame" begin
    include("testMFG.jl")
end

@testset "BFGS Optimizer" begin
    include("testBFGS.jl")
end

@testset "Time Stepping" begin
    include("testTimeStepping.jl")
end

@testset "Running Cost Functions (F)" begin
    include("testF.jl")
end

@testset "Terminal Cost Functions (G)" begin
    include("testG.jl")
end

@testset "Gaussians" begin
    include("testGaussians.jl")
end

@testset "Utils" begin
    include("testUtils.jl")
end
