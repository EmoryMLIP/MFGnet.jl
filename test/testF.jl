using Test
using MFGnet
using LinearAlgebra

@testset "Running Cost Function Tests" begin

    @testset "F0 (Zero Cost)" begin
        F = F0()
        d, nex = 2, 10
        U = randn(d+4, nex)
        t = 0.5

        @testset "Evaluation" begin
            result = F(U, t)
            @test all(result .== 0.0)
            @test length(result) == nex
        end

        @testset "getDeltaF" begin
            delta = getDeltaF(F, U, t)
            @test all(delta .== 0.0)
            @test length(delta) == nex
        end
    end

    @testset "Fp (Potential Cost)" begin
        Q(XT) = vec(sum(XT.^2, dims=1))
        rho0(X) = ones(size(X,2))

        d, nex = 2, 10
        X0 = randn(d, nex)
        λ = 0.5

        F = Fp(Q, rho0, rho0(X0), λ)

        @testset "Evaluation" begin
            U = randn(d+4, nex)
            t = 0.3
            result = F(U, t)

            @test length(result) == nex
            @test all(isfinite, result)
        end

        @testset "getDeltaF" begin
            U = randn(d+4, nex)
            t = 0.3
            delta = getDeltaF(F, U, t)

            @test length(delta) == nex
            @test all(isfinite, delta)
        end
    end

    @testset "Fe (Entropy Cost)" begin
        rho0(X) = ones(size(X,2))

        d, nex = 2, 10
        X0 = randn(d, nex)
        λ = 0.1

        F = Fe(rho0, rho0(X0), λ)

        @testset "Evaluation" begin
            U = randn(d+4, nex)
            t = 0.5
            result = F(U, t)

            @test length(result) == nex
            @test all(isfinite, result)  # Should handle numerical issues
        end

        @testset "Numerical stability with small densities" begin
            # Test the numerical safeguards we added
            rho0_small(X) = fill(1e-10, size(X,2))
            F_small = Fe(rho0_small, rho0_small(X0), λ)

            U = randn(d+4, nex)
            t = 0.5
            result = F_small(U, t)

            @test all(isfinite, result)  # No -Inf or NaN
            @test !any(isinf, result)
        end

        @testset "getDeltaF" begin
            U = randn(d+4, nex)
            t = 0.5
            delta = getDeltaF(F, U, t)

            @test length(delta) == nex
            @test all(isfinite, delta)
        end
    end

    @testset "Fcomb (Combined Cost)" begin
        Q(XT) = vec(sum(XT.^2, dims=1))
        rho0(X) = ones(size(X,2))

        d, nex = 2, 10
        X0 = randn(d, nex)

        F1 = F0()
        F2 = Fp(Q, rho0, rho0(X0), 0.5)
        F3 = Fe(rho0, rho0(X0), 0.1)

        @testset "Combination of costs" begin
            Fc = Fcomb([F1, F2, F3])

            U = randn(d+4, nex)
            t = 0.5

            result = Fc(U, t)
            expected = F1(U, t) + F2(U, t) + F3(U, t)

            @test result ≈ expected
        end

        @testset "getDeltaF combination" begin
            Fc = Fcomb([F1, F2])

            U = randn(d+4, nex)
            t = 0.5

            delta = getDeltaF(Fc, U, t)
            expected = getDeltaF(F1, U, t) + getDeltaF(F2, U, t)

            @test delta ≈ expected
        end
    end
end
