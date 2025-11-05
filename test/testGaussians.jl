using Test
using MFGnet
using LinearAlgebra
using Statistics

@testset "Gaussian Tests" begin

    @testset "Gaussian Construction" begin
        d = 2
        σ = ones(d)
        μ = zeros(d)
        α = 1.0

        G = Gaussian(d, σ, μ, α)

        @test G.d == d
        @test G.σ == σ
        @test G.μ == μ
        @test G.α == α
    end

    @testset "Gaussian PDF Evaluation" begin
        d = 2
        σ = ones(d)
        μ = zeros(d)

        G = Gaussian(d, σ, μ)

        @testset "At mean" begin
            X = zeros(d, 1)
            pdf_val = G(X)

            @test length(pdf_val) == 1
            @test pdf_val[1] > 0  # Density at mean should be positive
            @test isfinite(pdf_val[1])
        end

        @testset "Multiple points" begin
            X = randn(d, 10)
            pdf_vals = G(X)

            @test length(pdf_vals) == 10
            @test all(pdf_vals .>= 0)  # PDF is non-negative
            @test all(isfinite, pdf_vals)
        end

        @testset "Symmetry" begin
            X1 = [1.0; 0.0]
            X2 = [-1.0; 0.0]
            pdf1 = G(reshape(X1, 2, 1))
            pdf2 = G(reshape(X2, 2, 1))

            @test pdf1 ≈ pdf2  # Symmetric around mean
        end
    end

    @testset "Gaussian Sampling" begin
        d = 2
        σ = ones(d)
        μ = [1.0, 2.0]
        n = 1000

        G = Gaussian(d, σ, μ)

        @testset "Sample size" begin
            X = sample(G, n)

            @test size(X, 1) == d
            @test size(X, 2) == n
        end

        @testset "Sample statistics" begin
            X = sample(G, 5000)

            sample_mean = vec(mean(X, dims=2))
            sample_var = vec(var(X, dims=2))

            @test sample_mean ≈ μ rtol=0.1  # Mean should be close
            @test sample_var ≈ σ rtol=0.2   # Variance should be close
        end
    end

    @testset "GaussianMixture" begin
        d = 2
        σ1 = ones(d)
        μ1 = [0.0, 0.0]
        α1 = 0.6

        σ2 = ones(d)
        μ2 = [3.0, 3.0]
        α2 = 0.4

        G1 = Gaussian(d, σ1, μ1, α1)
        G2 = Gaussian(d, σ2, μ2, α2)

        GM = GaussianMixture([G1, G2])

        @testset "Total mass" begin
            @test totalMass(GM) ≈ α1 + α2
        end

        @testset "Sampling" begin
            n = 1000
            X = sample(GM, n)

            @test size(X, 1) == d
            @test size(X, 2) == n
        end

        @testset "Mixture proportions" begin
            # With large enough sample, proportions should match weights
            n = 10000
            X = sample(GM, n)

            # Count samples closer to each mode
            dist1 = vec(sum((X .- μ1).^2, dims=1))
            dist2 = vec(sum((X .- μ2).^2, dims=1))

            prop1 = sum(dist1 .< dist2) / n
            expected_prop1 = α1 / (α1 + α2)

            @test prop1 ≈ expected_prop1 rtol=0.15
        end
    end

    @testset "Edge Cases" begin
        @testset "1D Gaussian" begin
            d = 1
            σ = [1.0]
            μ = [0.0]

            G = Gaussian(d, σ, μ)

            X = randn(1, 10)
            pdf_vals = G(X)

            @test all(isfinite, pdf_vals)
            @test all(pdf_vals .>= 0)
        end

        @testset "High-dimensional" begin
            d = 10
            σ = ones(d)
            μ = zeros(d)

            G = Gaussian(d, σ, μ)

            X = randn(d, 5)
            pdf_vals = G(X)

            @test all(isfinite, pdf_vals)
            @test all(pdf_vals .>= 0)
        end
    end
end
