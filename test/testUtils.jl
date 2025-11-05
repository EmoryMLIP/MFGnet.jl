using Test
using MFGnet
using LinearAlgebra
using Flux

@testset "Utils Tests" begin

    @testset "append functions" begin
        @testset "Tuple append" begin
            A = (1, 2, 3)
            B = 4
            result = append(A, B)

            @test result == (1, 2, 3, 4)
            @test result isa Tuple
        end

        @testset "Scalar append" begin
            A = 1
            B = 2
            result = append(A, B)

            @test result == (1, 2)
            @test result isa Tuple
        end

        @testset "Nested append" begin
            A = (1, 2)
            B = 3
            C = 4
            result = append(append(A, B), C)

            @test result == (1, 2, 3, 4)
        end
    end

    @testset "evalObjAndGrad" begin
        # Create a simple test problem
        d = 2
        nex = 20
        m = 5
        nTh = 2
        X0 = randn(d, nex)

        rho0(X) = ones(size(X,2))
        rho1(X) = ones(size(X,2))
        w = ones(nex) / nex

        F = F0()
        G = Gls(rho0, rho1, rho0(X0), rho1(X0), 1.0)
        J = MeanFieldGame(F, G, X0, rho0, w)

        # SingleLayer network: parms = (w0, ΘN, A0, c0, z0)
        Θ1 = (0.01*randn(m, d+1), 0.1*randn(m))
        Θ2 = (0.01*randn(m, m), 0.1*randn(m))
        parms = (ones(m), (Θ1, Θ2), zeros(d+1,d+1), zeros(d+1), zeros(1))
        parms = MFGnet.myMap(x->x, parms)  # Ensure correct structure for Flux
        # Note: Using deprecated Flux.params() to match evalObjAndGrad API
        ps = Flux.params(parms)
        Θvec = param2vec(parms)

        @testset "Objective evaluation" begin
            obj_val, grad_vec = evalObjAndGrad(J, Θvec, parms, ps)

            @test obj_val isa Real
            @test isfinite(obj_val)
            @test length(grad_vec) == length(Θvec)
            @test all(isfinite, grad_vec)
        end

        @testset "Gradient correctness" begin
            # Gradient should not be all zeros for non-trivial problem
            _, grad_vec = evalObjAndGrad(J, Θvec, parms, ps)

            @test !all(grad_vec .== 0.0)
        end
    end

    @testset "Type stability" begin
        # Test that append maintains type stability
        A = (1.0, 2.0)
        B = 3.0

        result = append(A, B)
        @test eltype(result) == Float64
    end
end
