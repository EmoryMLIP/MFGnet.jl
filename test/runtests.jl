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

