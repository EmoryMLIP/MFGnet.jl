using LinearAlgebra
using Revise
using Statistics
using Test
using MFGnet


@testset "Float64 3D Array" begin
    A = randn(32,16,2)
    At = MFGnet.linInter1D(0.5,1.,A)
    @test At ≈ mean(A,dims=3)
end

@testset "Float64 2D Array" begin
    B = randn(32,2)
    Bt = MFGnet.linInter1D(0.5,1.,B)
    @test Bt ≈ mean(B,dims=2)
end

@testset "tuple" begin
    A = randn(32,16,2)
    B = randn(32,2)
    AB = MFGnet.linInter1D(0.5,1.,(A,B))
    @test AB[1] ≈ mean(A,dims=3)
    @test AB[2] ≈ mean(B,dims=2)
end
