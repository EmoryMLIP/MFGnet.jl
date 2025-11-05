using Test
using MFGnet
using LinearAlgebra

@testset "Terminal Cost Function Tests" begin

    @testset "Gls (Least-Squares)" begin
        rho0(X) = ones(size(X,2))
        rho1(X) = ones(size(X,2))

        d, nex = 2, 10
        X0 = randn(d, nex)
        mu = 1.0

        G = Gls(rho0, rho1, rho0(X0), rho1(X0), mu)

        @testset "Evaluation" begin
            U = randn(d+4, nex)
            result = G(U)

            @test length(result) == nex
            @test all(isfinite, result)
        end

        @testset "Numerical stability" begin
            # Test with extreme values in U[d+1,:]
            U = randn(d+4, nex)
            U[d+1, :] .= 100.0  # Large values

            result = G(U)
            @test all(isfinite, result)  # Safeguards should prevent overflow
        end

        @testset "getDeltaG" begin
            U = randn(d+4, nex)
            delta = getDeltaG(G, U)

            @test length(delta) == nex
            @test all(isfinite, delta)
        end
    end

    @testset "Gkl (KL Divergence)" begin
        rho0(X) = ones(size(X,2))
        rho1(X) = ones(size(X,2))

        d, nex = 2, 10
        X0 = randn(d, nex)
        mu = 1.0

        G = Gkl(rho0, rho1, rho0(X0), rho1(X0), mu)

        @testset "Evaluation" begin
            U = randn(d+4, nex)
            result = G(U)

            @test length(result) == nex
            @test all(isfinite, result)
        end

        @testset "Numerical stability with small densities" begin
            rho0_small(X) = fill(1e-10, size(X,2))
            rho1_small(X) = fill(1e-10, size(X,2))
            G_small = Gkl(rho0_small, rho1_small, rho0_small(X0), rho1_small(X0), mu)

            U = randn(d+4, nex)
            result = G_small(U)

            @test all(isfinite, result)  # No -Inf from log(0)
            @test !any(isinf, result)
        end

        @testset "getDeltaG" begin
            U = randn(d+4, nex)
            delta = getDeltaG(G, U)

            @test length(delta) == nex
            @test all(isfinite, delta)
        end
    end

    @testset "Gpref (Preference)" begin
        Pref(X) = vec(sum(X.^2, dims=1))
        rho0(X) = ones(size(X,2))

        d, nex = 2, 10
        X0 = randn(d, nex)
        mu = 0.5

        G = Gpref(Pref, rho0(X0), mu)

        @testset "Evaluation" begin
            U = randn(d+4, nex)
            result = G(U)

            @test length(result) == nex
            @test all(isfinite, result)
        end

        @testset "getDeltaG" begin
            U = randn(d+4, nex)
            delta = getDeltaG(G, U)

            @test length(delta) == nex
            @test all(isfinite, delta)
        end
    end

    @testset "Gcomb (Combined Terminal Cost)" begin
        rho0(X) = ones(size(X,2))
        rho1(X) = ones(size(X,2))

        d, nex = 2, 10
        X0 = randn(d, nex)

        G1 = Gls(rho0, rho1, rho0(X0), rho1(X0), 0.5)
        G2 = Gkl(rho0, rho1, rho0(X0), rho1(X0), 0.3)

        @testset "Combination" begin
            Gc = Gcomb([G1, G2])

            U = randn(d+4, nex)
            result = Gc(U)
            expected = G1(U) + G2(U)

            @test result ≈ expected
        end
    end
end
