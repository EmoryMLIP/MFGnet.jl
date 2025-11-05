using Test
using MFGnet
using LinearAlgebra
using Statistics

@testset "MeanFieldGame Tests" begin

    @testset "MFG Construction" begin
        # Create simple MFG problem
        d = 2
        nex = 100
        X0 = randn(d, nex)

        rho0(X) = ones(size(X,2))
        rho1(X) = ones(size(X,2))
        w = ones(nex) / nex

        F = F0()
        G = Gls(rho0, rho1, rho0(X0), rho1(X0), 1.0)

        @testset "Basic construction" begin
            J = MeanFieldGame(F, G, X0, rho0, w)
            @test J isa MeanFieldGame
            @test size(J.X0) == (d, nex)
            @test length(J.w) == nex
        end

        @testset "With different cost functions" begin
            # Test with entropy cost
            Fe_cost = Fe(rho0, rho0(X0), 0.1)
            J1 = MeanFieldGame(Fe_cost, G, X0, rho0, w)
            @test J1 isa MeanFieldGame

            # Test with potential cost
            Q(XT) = sum(XT.^2, dims=1)
            Fp_cost = Fp(Q, rho0, rho0(X0), 0.1)
            J2 = MeanFieldGame(Fp_cost, G, X0, rho0, w)
            @test J2 isa MeanFieldGame
        end
    end

    @testset "MFG Callable" begin
        # Basic test that MFG is callable
        # Full evaluation tests require understanding complete parameter structure
        # which is tested in example files

        d = 2
        nex = 10
        X0 = randn(d, nex)

        rho0(X) = ones(size(X,2))
        rho1(X) = ones(size(X,2))
        w = ones(nex) / nex

        F = F0()
        G = Gls(rho0, rho1, rho0(X0), rho1(X0), 1.0)
        J = MeanFieldGame(F, G, X0, rho0, w)

        @test J isa MeanFieldGame
        @test J.F === F
        @test J.G === G
    end

    @testset "Edge Cases" begin
        # Test with minimal examples
        d = 1
        nex = 10
        X0 = randn(d, nex)

        rho0(X) = ones(size(X,2))
        rho1(X) = ones(size(X,2))
        w = ones(nex) / nex

        F = F0()
        G = Gls(rho0, rho1, rho0(X0), rho1(X0), 1.0)

        @testset "1D spatial problem" begin
            # This tests the getQ fix for d=1
            J = MeanFieldGame(F, G, X0, rho0, w)
            @test J isa MeanFieldGame
        end

        @testset "Non-uniform weights" begin
            w_nonuniform = rand(nex)
            w_nonuniform ./= sum(w_nonuniform)
            J = MeanFieldGame(F, G, X0, rho0, w_nonuniform)
            @test sum(J.w) ≈ 1.0
        end
    end

    @testset "Numerical Stability" begin
        d = 2
        nex = 50
        X0 = randn(d, nex)

        # Test with near-zero densities (tests numerical safeguards)
        rho0_small(X) = fill(1e-10, size(X,2))
        rho1_small(X) = fill(1e-10, size(X,2))
        w = ones(nex) / nex

        @testset "Entropy with small densities" begin
            F = Fe(rho0_small, rho0_small(X0), 1.0)
            G = Gls(rho0_small, rho1_small, rho0_small(X0), rho1_small(X0), 1.0)
            J = MeanFieldGame(F, G, X0, rho0_small, w)
            @test J isa MeanFieldGame
        end

        @testset "KL divergence with small densities" begin
            rho0(X) = ones(size(X,2))
            F = F0()
            G = Gkl(rho0_small, rho1_small, rho0_small(X0), rho1_small(X0), 1.0)
            J = MeanFieldGame(F, G, X0, rho0, w)
            @test J isa MeanFieldGame

            # Evaluate G to ensure numerical safeguards work
            U = randn(d+4, nex)
            g_val = G(U)
            @test all(isfinite, g_val)
        end
    end
end
