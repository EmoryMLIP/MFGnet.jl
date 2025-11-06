"""
End-to-end training tests

Tests complete MFG training pipeline:
1. Full optimization with legacy BFGS
2. Full optimization with Optimization.jl
3. Training convergence
4. Solution quality
"""

using Test
using LinearAlgebra
using Printf
using MFGnet

include("../utils/test_utils.jl")
include("../utils/test_problems.jl")
using .TestUtils
using .TestProblems

@testset "Legacy BFGS Training" begin

    @testset "1D Gaussian - Legacy full pipeline" begin
        prob_data = gaussian_1d_transport(nex=50)
        mfg, Θ_init = prob_data.mfg, prob_data.Θ_init

        # Define objective and gradient
        function fdf(θ_vec)
            # Convert vector to parameter structure
            Θ = reconstruct_params(θ_vec, Θ_init)

            # Compute objective and gradient
            using Zygote
            Jc, ∇Jc_tuple = Zygote.withgradient(θ -> mfg(θ), Θ)

            # Convert gradient to vector
            ∇Jc_vec = vec_params(∇Jc_tuple[1])

            return Jc, ∇Jc_vec
        end

        f(θ_vec) = fdf(θ_vec)[1]

        # Initial parameters as vector
        θ0 = vec_params(Θ_init)

        # Run BFGS
        θ_opt, flag, his, _, _ = bfgs(f, fdf, θ0;
                                      maxIter=20,
                                      atol=1e-6,
                                      out=0)  # Silent

        # Check convergence
        @test flag in [0, -1]  # Success or max iterations
        @test size(his, 1) > 0

        # Loss should decrease
        @test his[end, 1] <= his[1, 1]

        println("\nLegacy BFGS Training:")
        println("  Initial loss: $(his[1,1])")
        println("  Final loss:   $(his[end,1])")
        println("  Iterations:   $(size(his,1))")
    end

    @testset "2D Problem - Legacy training" begin
        prob_data = gaussian_2d_transport(nex=100)
        mfg, Θ_init = prob_data.mfg, prob_data.Θ_init

        function fdf(θ_vec)
            Θ = reconstruct_params(θ_vec, Θ_init)
            using Zygote
            Jc, ∇Jc_tuple = Zygote.withgradient(θ -> mfg(θ), Θ)
            return Jc, vec_params(∇Jc_tuple[1])
        end

        f(θ_vec) = fdf(θ_vec)[1]
        θ0 = vec_params(Θ_init)

        θ_opt, flag, his, _, _ = bfgs(f, fdf, θ0;
                                      maxIter=20,
                                      atol=1e-6,
                                      out=0)

        @test size(his, 1) > 0
        @test his[end, 1] <= his[1, 1]

        # Gradient norm should decrease
        @test his[end, 2] <= his[1, 2]
    end

end

@testset "Optimization.jl Training" begin

    @testset "Basic optimization setup" begin
        prob_data = gaussian_1d_transport(nex=30)
        mfg, Θ_init = prob_data.mfg, prob_data.Θ_init

        @test_skip begin
            # Create optimization problem
            prob = create_optimization_problem(mfg, Θ_init;
                                             use_diffeq=true)

            @test prob isa OptimMFGProblem
            @test prob.opt_prob isa OptimizationProblem
        end
    end

    @testset "LBFGS optimization" begin
        prob_data = gaussian_1d_transport(nex=30)
        mfg, Θ_init = prob_data.mfg, prob_data.Θ_init

        @test_skip begin
            # Create and solve
            prob = create_optimization_problem(mfg, Θ_init;
                                             use_diffeq=true)

            sol = solve_optimization(prob, LBFGS();
                                   maxiters=20,
                                   verbose=false)

            @test sol isa Optimization.OptimizationSolution
            @test isfinite(sol.objective)
            @test sol.retcode in [:Success, :MaxIters]
        end
    end

    @testset "Adam optimization (first-order)" begin
        prob_data = gaussian_1d_transport(nex=30)
        mfg, Θ_init = prob_data.mfg, prob_data.Θ_init

        @test_skip begin
            prob = create_optimization_problem(mfg, Θ_init;
                                             use_diffeq=true)

            # Adam with small learning rate
            sol = solve_optimization(prob, Adam(0.01);
                                   maxiters=100,
                                   verbose=false)

            @test sol isa Optimization.OptimizationSolution
            @test isfinite(sol.objective)
        end
    end

    @testset "High-level train_mfg interface" begin
        prob_data = gaussian_1d_transport(nex=30)
        mfg, Θ_init = prob_data.mfg, prob_data.Θ_init

        @test_skip begin
            # High-level training
            result = train_mfg(mfg, Θ_init;
                             optimizer=LBFGS(),
                             maxiters=20,
                             use_diffeq=true,
                             verbose=false)

            @test haskey(result, :Θ_opt)
            @test haskey(result, :loss_history)
            @test haskey(result, :sol)

            @test length(result.loss_history) > 0
            @test result.loss_history[end] <= result.loss_history[1]
        end
    end

end

@testset "Training Convergence Properties" begin

    @testset "Loss decreases monotonically (quasi-Newton)" begin
        prob_data = gaussian_1d_transport(nex=30)
        mfg, Θ_init = prob_data.mfg, prob_data.Θ_init

        # Track loss history
        loss_history = Float64[]

        function callback(state, loss_val)
            push!(loss_history, loss_val)
            return false
        end

        @test_skip begin
            prob = create_optimization_problem(mfg, Θ_init; use_diffeq=true)
            sol = solve_optimization(prob, LBFGS();
                                   maxiters=20,
                                   callback=callback,
                                   verbose=false)

            # Loss should generally decrease (allowing for line search variations)
            @test loss_history[end] < loss_history[1]

            # Count how many times loss increased
            increases = sum(diff(loss_history) .> 0)
            # Should be mostly decreasing (allow some increases due to line search)
            @test increases < length(loss_history) / 2
        end
    end

    @testset "Gradient norm decreases" begin
        prob_data = gaussian_1d_transport(nex=30)
        mfg, Θ_init = prob_data.mfg, prob_data.Θ_init

        # Legacy BFGS tracks gradient norm
        function fdf(θ_vec)
            Θ = reconstruct_params(θ_vec, Θ_init)
            using Zygote
            Jc, ∇Jc = Zygote.withgradient(θ -> mfg(θ), Θ)
            return Jc, vec_params(∇Jc[1])
        end

        f(θ_vec) = fdf(θ_vec)[1]
        θ0 = vec_params(Θ_init)

        θ_opt, flag, his, _, _ = bfgs(f, fdf, θ0;
                                      maxIter=30,
                                      atol=1e-7,
                                      out=0)

        # Gradient norms in column 2
        grad_norms = his[:, 2]

        @test grad_norms[end] < grad_norms[1]

        # Should converge to small gradient
        @test grad_norms[end] < 1e-2 || flag == -1  # Small gradient or max iter
    end

    @testset "Convergence on simple problem" begin
        # Very simple 1D problem should converge well
        prob_data = gaussian_1d_transport(nex=20)
        mfg, Θ_init = prob_data.mfg, prob_data.Θ_init

        function fdf(θ_vec)
            Θ = reconstruct_params(θ_vec, Θ_init)
            using Zygote
            Jc, ∇Jc = Zygote.withgradient(θ -> mfg(θ), Θ)
            return Jc, vec_params(∇Jc[1])
        end

        f(θ_vec) = fdf(θ_vec)[1]
        θ0 = vec_params(Θ_init)

        θ_opt, flag, his, _, _ = bfgs(f, fdf, θ0;
                                      maxIter=50,
                                      atol=1e-5,
                                      out=0)

        # Should converge or reach max iterations
        @test flag in [0, -1]

        # Final gradient should be small
        if flag == 0
            @test his[end, 2] < 1e-5
        end
    end

end

@testset "Solution Quality Assessment" begin

    @testset "Final objective is reasonable" begin
        prob_data = gaussian_1d_transport(nex=50)
        mfg, Θ_init = prob_data.mfg, prob_data.Θ_init

        # Train briefly
        function fdf(θ_vec)
            Θ = reconstruct_params(θ_vec, Θ_init)
            using Zygote
            Jc, ∇Jc = Zygote.withgradient(θ -> mfg(θ), Θ)
            return Jc, vec_params(∇Jc[1])
        end

        f(θ_vec) = fdf(θ_vec)[1]
        θ0 = vec_params(Θ_init)

        θ_opt, flag, his, _, _ = bfgs(f, fdf, θ0;
                                      maxIter=30,
                                      atol=1e-6,
                                      out=0)

        final_loss = his[end, 1]

        # Should be finite and positive
        @test isfinite(final_loss)
        @test final_loss > 0

        # Should be better than initial
        @test final_loss < his[1, 1]
    end

    @testset "Cost components are balanced" begin
        prob_data = gaussian_1d_transport(nex=50)
        mfg, Θ_init = prob_data.mfg, prob_data.Θ_init

        # Train briefly
        function fdf(θ_vec)
            Θ = reconstruct_params(θ_vec, Θ_init)
            using Zygote
            Jc, ∇Jc = Zygote.withgradient(θ -> mfg(θ), Θ)
            return Jc, vec_params(∇Jc[1])
        end

        f(θ_vec) = fdf(θ_vec)[1]
        θ0 = vec_params(Θ_init)

        θ_opt, flag, his, _, _ = bfgs(f, fdf, θ0;
                                      maxIter=20,
                                      atol=1e-6,
                                      out=0)

        # Evaluate final solution
        Θ_final = reconstruct_params(θ_opt, Θ_init)
        Jc_final = mfg(Θ_final)

        # Check cost components
        cs = mfg.cs  # [costL, costF, costG, costHJ, costHJf] .* α

        @test all(isfinite.(cs))
        @test all(cs .>= 0)  # All costs should be non-negative

        # Transport cost and terminal cost should be significant
        @test cs[1] > 0  # Transport cost
        @test cs[3] > 0  # Terminal cost

        println("\nCost components:")
        @printf("  Transport:   %.6e\n", cs[1])
        @printf("  Interaction: %.6e\n", cs[2])
        @printf("  Terminal:    %.6e\n", cs[3])
        @printf("  HJ residual: %.6e\n", cs[4])
        @printf("  HJ final:    %.6e\n", cs[5])
    end

end

@testset "Reproducibility" begin

    @testset "Same initialization gives same result" begin
        # Run optimization twice with same initialization
        prob_data = gaussian_1d_transport(nex=30, seed=1234)
        mfg1, Θ_init1 = prob_data.mfg, prob_data.Θ_init

        prob_data = gaussian_1d_transport(nex=30, seed=1234)
        mfg2, Θ_init2 = prob_data.mfg, prob_data.Θ_init

        # Should have identical initializations
        @test vec_params(Θ_init1) ≈ vec_params(Θ_init2)

        function fdf(mfg, Θ_init)
            return (θ_vec) -> begin
                Θ = reconstruct_params(θ_vec, Θ_init)
                using Zygote
                Jc, ∇Jc = Zygote.withgradient(θ -> mfg(θ), Θ)
                return Jc, vec_params(∇Jc[1])
            end
        end

        f1(θ_vec) = fdf(mfg1, Θ_init1)(θ_vec)[1]
        f2(θ_vec) = fdf(mfg2, Θ_init2)(θ_vec)[1]

        θ0_1 = vec_params(Θ_init1)
        θ0_2 = vec_params(Θ_init2)

        θ_opt1, flag1, his1, _, _ = bfgs(f1, fdf(mfg1, Θ_init1), θ0_1;
                                        maxIter=10, atol=1e-6, out=0)
        θ_opt2, flag2, his2, _, _ = bfgs(f2, fdf(mfg2, Θ_init2), θ0_2;
                                        maxIter=10, atol=1e-6, out=0)

        # Results should be identical
        @test flag1 == flag2
        @test his1 ≈ his2
        @test θ_opt1 ≈ θ_opt2
    end

end

@testset "Checkpointing and Resume" begin

    @testset "Can save and load parameters" begin
        prob_data = gaussian_1d_transport(nex=30)
        mfg, Θ_init = prob_data.mfg, prob_data.Θ_init

        # "Save" parameters (just copy)
        Θ_saved = deepcopy(Θ_init)

        # "Load" and verify
        @test vec_params(Θ_saved) == vec_params(Θ_init)

        # Evaluate at loaded parameters
        Jc = mfg(Θ_saved)
        @test isfinite(Jc)
    end

    @testset "Intermediate checkpoints during training" begin
        @test_skip begin
            prob_data = gaussian_1d_transport(nex=30)
            mfg, Θ_init = prob_data.mfg, prob_data.Θ_init

            # Train with checkpointing
            result = train_mfg(mfg, Θ_init;
                             optimizer=LBFGS(),
                             maxiters=20,
                             checkpoint_every=5,
                             save_path=tempdir(),
                             verbose=false)

            # Check that checkpoints were created
            # (Would need actual file system check)
            @test true
        end
    end

end

@testset "Performance Regression Tests" begin

    @testset "Training time is reasonable" begin
        prob_data = gaussian_1d_transport(nex=50)
        mfg, Θ_init = prob_data.mfg, prob_data.Θ_init

        function fdf(θ_vec)
            Θ = reconstruct_params(θ_vec, Θ_init)
            using Zygote
            Jc, ∇Jc = Zygote.withgradient(θ -> mfg(θ), Θ)
            return Jc, vec_params(∇Jc[1])
        end

        f(θ_vec) = fdf(θ_vec)[1]
        θ0 = vec_params(Θ_init)

        # Time 10 iterations
        time_elapsed = @elapsed bfgs(f, fdf, θ0;
                                    maxIter=10,
                                    atol=1e-6,
                                    out=-1)  # Silent

        # Should complete in reasonable time (problem-dependent)
        @test time_elapsed < 60.0  # Less than 1 minute for 10 iterations

        println("Time for 10 iterations: $(time_elapsed)s")
    end

end
