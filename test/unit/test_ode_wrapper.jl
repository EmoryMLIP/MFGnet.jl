"""
Unit tests for ODE wrapper (DifferentialEquations.jl integration)

Tests:
1. ODEProblem construction
2. Single ODE step comparison
3. Full trajectory comparison (legacy vs DiffEq)
4. Adaptive vs fixed step accuracy
5. Different solver selection
"""

using Test
using LinearAlgebra
using MFGnet

# Import test utilities (will be available after module is loaded)
include("../utils/test_utils.jl")
include("../utils/test_problems.jl")
using .TestUtils
using .TestProblems

@testset "ODE Wrapper Construction" begin

    @testset "ODEProblem Creation - 1D" begin
        # Create simple 1D problem
        prob_data = gaussian_1d_transport(nex=20)
        mfg, Θ = prob_data.mfg, prob_data.Θ_init

        # Test: Can we create ODEProblem?
        # NOTE: This assumes create_ode_problem is implemented
        # For now, this is a placeholder showing the expected interface

        @test_skip begin
            # This will be implemented during migration
            prob = create_ode_problem(mfg, Θ)
            @test prob isa ODEMFGProblem
            @test prob.ode_prob isa ODEProblem
        end
    end

    @testset "ODEProblem Creation - 2D" begin
        prob_data = gaussian_2d_transport(nex=50)
        mfg, Θ = prob_data.mfg, prob_data.Θ_init

        @test_skip begin
            prob = create_ode_problem(mfg, Θ)
            @test prob isa ODEMFGProblem
            @test prob.ode_prob isa ODEProblem

            # Check dimensions
            u0 = prob.ode_prob.u0
            @test length(u0) == (2 + 4) * 50  # (d + 4) * nex
        end
    end

end

@testset "ODE Integration - Legacy vs DiffEq" begin

    @testset "1D Gaussian - RK4 Equivalence" begin
        prob_data = gaussian_1d_transport(nex=50)
        mfg, Θ = prob_data.mfg, prob_data.Θ_init

        # Legacy integration (RK4)
        Jc_legacy = mfg(Θ)
        UN_legacy = copy(mfg.UN)
        cs_legacy = copy(mfg.cs)

        # DifferentialEquations.jl integration
        # NOTE: This uses new interface that will be implemented
        @test_skip begin
            # Use RK4() solver for direct comparison
            Jc_diffeq = mfg(Θ, use_diffeq=true, solver=RK4(),
                           abstol=1e-10, reltol=1e-10, adaptive=false, dt=(1.0/mfg.nt))
            UN_diffeq = copy(mfg.UN)
            cs_diffeq = copy(mfg.cs)

            # Should be very close (not exact due to floating point)
            tol = MIGRATION_TOLERANCES
            compare_objectives(Jc_legacy, Jc_diffeq;
                             abstol=tol.objective_abstol,
                             reltol=tol.objective_reltol,
                             name="RK4 objective")

            compare_states(UN_legacy, UN_diffeq;
                         abstol=tol.state_abstol,
                         reltol=tol.state_reltol,
                         name="RK4 final state")

            # Cost components should also match
            @test isapprox(cs_legacy, cs_diffeq, rtol=1e-3)
        end
    end

    @testset "2D Gaussian - Numerical Equivalence" begin
        prob_data = gaussian_2d_transport(nex=100)
        mfg, Θ = prob_data.mfg, prob_data.Θ_init

        # Legacy (default RK1 with nt=10)
        Jc_legacy = mfg(Θ)
        UN_legacy = copy(mfg.UN)

        @test_skip begin
            # DiffEq with equivalent fixed-step Euler
            Jc_diffeq = mfg(Θ, use_diffeq=true, solver=Euler(),
                           adaptive=false, dt=(1.0/mfg.nt))
            UN_diffeq = copy(mfg.UN)

            tol = MIGRATION_TOLERANCES
            compare_objectives(Jc_legacy, Jc_diffeq;
                             abstol=tol.objective_abstol,
                             reltol=tol.objective_reltol)

            compare_states(UN_legacy, UN_diffeq;
                         abstol=tol.state_abstol,
                         reltol=tol.state_reltol)
        end
    end

end

@testset "Adaptive vs Fixed Step" begin

    @testset "Adaptive is more accurate" begin
        prob_data = gaussian_1d_transport(nex=50)
        mfg, Θ = prob_data.mfg, prob_data.Θ_init

        @test_skip begin
            # Fixed step (RK4 with 100 steps)
            Jc_fixed = mfg(Θ, use_diffeq=true, solver=RK4(),
                          adaptive=false, dt=0.01)

            # Adaptive (Tsit5 with tight tolerances)
            Jc_adaptive = mfg(Θ, use_diffeq=true, solver=Tsit5(),
                             abstol=1e-8, reltol=1e-6)

            # Both should give finite, reasonable results
            @test isfinite(Jc_fixed)
            @test isfinite(Jc_adaptive)

            # Adaptive should be at least as accurate
            # (hard to test without reference solution)
            @test abs(Jc_adaptive - Jc_fixed) / abs(Jc_fixed) < 0.1
        end
    end

    @testset "Adaptive is more efficient" begin
        prob_data = gaussian_2d_transport(nex=100)
        mfg, Θ = prob_data.mfg, prob_data.Θ_init

        @test_skip begin
            # Fixed: Many steps
            t_fixed = @elapsed mfg(Θ, use_diffeq=true, solver=RK4(),
                                   adaptive=false, dt=0.001)  # 1000 steps

            # Adaptive: Should take fewer steps
            t_adaptive = @elapsed mfg(Θ, use_diffeq=true, solver=Tsit5(),
                                      abstol=1e-6, reltol=1e-3)

            # Adaptive should be faster (or at least comparable)
            # This is a soft test - mainly checking it doesn't crash
            @test t_adaptive > 0
            @test t_fixed > 0
        end
    end

end

@testset "Solver Selection" begin

    @testset "Multiple solvers give consistent results" begin
        prob_data = gaussian_1d_transport(nex=50)
        mfg, Θ = prob_data.mfg, prob_data.Θ_init

        @test_skip begin
            solvers = [
                ("Tsit5", Tsit5()),
                ("Vern7", Vern7()),
                ("DP5", DP5()),
            ]

            results = Dict()
            for (name, solver) in solvers
                Jc = mfg(Θ, use_diffeq=true, solver=solver,
                        abstol=1e-8, reltol=1e-6)
                results[name] = Jc
            end

            # All should be close to each other
            vals = collect(values(results))
            for i in 1:length(vals)-1
                @test isapprox(vals[i], vals[i+1], rtol=1e-4)
            end
        end
    end

    @testset "Stiffness detection" begin
        prob_data = crowd_motion_2d(nex=100)
        mfg, Θ = prob_data.mfg, prob_data.Θ_init

        @test_skip begin
            # Auto-detect stiffness
            problem_type = auto_detect_stiffness(mfg, Θ)
            @test problem_type in [:stiff, :nonstiff]

            # Select appropriate solver
            solver, tols = select_ode_solver(problem_type)
            @test solver isa OrdinaryDiffEq.OrdinaryDiffEqAlgorithm

            # Should be able to solve
            Jc = mfg(Θ, use_diffeq=true, solver=solver,
                    abstol=tols.abstol, reltol=tols.reltol)
            @test isfinite(Jc)
        end
    end

end

@testset "Callback Integration" begin

    @testset "Progress monitoring callback" begin
        prob_data = gaussian_1d_transport(nex=50)
        mfg, Θ = prob_data.mfg, prob_data.Θ_init

        @test_skip begin
            # Create callback that records integration progress
            times = Float64[]
            norms = Float64[]

            function record_callback(u, t, integrator)
                push!(times, t)
                push!(norms, norm(u))
                return false  # Don't terminate
            end

            # Solve with callback
            Jc = mfg(Θ, use_diffeq=true, solver=Tsit5(),
                    callback=DiscreteCallback((u,t,i)->true, record_callback))

            @test length(times) > 0
            @test length(norms) > 0
            @test issorted(times)
        end
    end

    @testset "Divergence detection callback" begin
        prob_data = gaussian_1d_transport(nex=50)
        mfg, Θ = prob_data.mfg, prob_data.Θ_init

        @test_skip begin
            # Create callback that terminates if norm too large
            max_norm = 1e5
            condition = (u, t, integrator) -> norm(u) > max_norm
            affect! = (integrator) -> terminate!(integrator)

            cb = DiscreteCallback(condition, affect!)

            # This should complete normally with reasonable parameters
            Jc = mfg(Θ, use_diffeq=true, solver=Tsit5(), callback=cb)
            @test isfinite(Jc)

            # With bad parameters (very large), should terminate early
            Θ_bad = add_direction(Θ, random_direction(Θ), 1000.0)
            Jc_bad = mfg(Θ_bad, use_diffeq=true, solver=Tsit5(), callback=cb)
            # Might diverge or give large value
            @test true  # Test passes if it doesn't crash
        end
    end

end

@testset "State Trajectory Saving" begin

    @testset "Full trajectory storage" begin
        prob_data = gaussian_1d_transport(nex=20)
        mfg, Θ = prob_data.mfg, prob_data.Θ_init

        @test_skip begin
            # Solve with trajectory saving
            prob = create_ode_problem(mfg, Θ)
            sol = solve(prob.ode_prob, Tsit5();
                       save_everystep=true,
                       saveat=0.1)  # Save every 0.1 time units

            @test length(sol.t) > 2  # Should have saved intermediate states
            @test sol.t[1] ≈ 0.0
            @test sol.t[end] ≈ 1.0

            # Can interpolate at arbitrary times
            u_half = sol(0.5)
            @test length(u_half) == length(sol.u[1])
        end
    end

end

@testset "Edge Cases" begin

    @testset "Very short time span" begin
        prob_data = gaussian_1d_transport(nex=20)
        mfg, Θ = prob_data.mfg, prob_data.Θ_init

        # Modify to very short time
        mfg.tspan = [0.0, 0.01]

        @test_skip begin
            Jc = mfg(Θ, use_diffeq=true, solver=Tsit5())
            @test isfinite(Jc)
        end
    end

    @testset "Very long time span" begin
        prob_data = gaussian_1d_transport(nex=20)
        mfg, Θ = prob_data.mfg, prob_data.Θ_init

        # Modify to long time
        mfg.tspan = [0.0, 10.0]

        @test_skip begin
            Jc = mfg(Θ, use_diffeq=true, solver=Tsit5(),
                    abstol=1e-6, reltol=1e-3)
            @test isfinite(Jc)
        end
    end

    @testset "Single particle" begin
        # Edge case: only one particle
        prob_data = gaussian_1d_transport(nex=1)
        mfg, Θ = prob_data.mfg, prob_data.Θ_init

        # Legacy
        Jc_legacy = mfg(Θ)
        @test isfinite(Jc_legacy)

        @test_skip begin
            # DiffEq
            Jc_diffeq = mfg(Θ, use_diffeq=true, solver=Tsit5())
            @test isfinite(Jc_diffeq)
        end
    end

    @testset "Many particles" begin
        # Stress test: many particles
        prob_data = gaussian_2d_transport(nex=1000)
        mfg, Θ = prob_data.mfg, prob_data.Θ_init

        @test_skip begin
            Jc = mfg(Θ, use_diffeq=true, solver=Tsit5())
            @test isfinite(Jc)
        end
    end

end

@testset "Precision Types" begin

    @testset "Float64 (default)" begin
        prob_data = gaussian_1d_transport(nex=50, R=Float64)
        mfg, Θ = prob_data.mfg, prob_data.Θ_init

        Jc = mfg(Θ)
        @test eltype(Jc) == Float64
        @test eltype(mfg.UN) == Float64
    end

    @testset "Float32 compatibility" begin
        prob_data = gaussian_1d_transport(nex=50, R=Float32)
        mfg, Θ = prob_data.mfg, prob_data.Θ_init

        Jc = mfg(Θ)
        @test eltype(Jc) == Float32
        @test eltype(mfg.UN) == Float32

        @test_skip begin
            # DiffEq should also work with Float32
            Jc_diffeq = mfg(Θ, use_diffeq=true, solver=Tsit5())
            @test eltype(Jc_diffeq) == Float32
        end
    end

end
