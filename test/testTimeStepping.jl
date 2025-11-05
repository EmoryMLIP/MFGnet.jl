using Test
using MFGnet
using LinearAlgebra

@testset "Time Stepping Tests" begin

    @testset "RK1 (Forward Euler)" begin
        # Test on simple ODE: du/dt = -u, exact solution: u(t) = u0 * exp(-t)
        odefun_simple(J, u, Θ, t) = -u

        u0 = [1.0]
        t0, t1 = 0.0, 0.1
        stepper = RK1Step()

        @testset "Single step" begin
            u1 = MFGnet.step(stepper, odefun_simple, nothing, u0, nothing, t0, t1)

            # RK1: u1 = u0 + h*f(t0, u0) = 1 + 0.1*(-1) = 0.9
            @test u1[1] ≈ 0.9
        end

        @testset "Accuracy order" begin
            # Test convergence rate: should be first-order
            h_values = [0.1, 0.05, 0.025]
            errors = Float64[]

            for h in h_values
                u = copy(u0)
                nt = Int(1.0 / h)
                for i=1:nt
                    u = MFGnet.step(stepper, odefun_simple, nothing, u, nothing, (i-1)*h, i*h)
                end
                exact = u0 .* exp(-1.0)
                push!(errors, norm(u - exact))
            end

            # First-order method: error ∝ h, so error_ratio ≈ 2 when h is halved
            ratio1 = errors[1] / errors[2]
            ratio2 = errors[2] / errors[3]

            @test 1.5 < ratio1 < 2.5  # Allow some numerical error
            @test 1.5 < ratio2 < 2.5
        end
    end

    @testset "RK4 (Runge-Kutta 4)" begin
        # Test on du/dt = -u
        odefun_simple(J, u, Θ, t) = -u

        u0 = [1.0]
        t0, t1 = 0.0, 0.1
        stepper = RK4Step()

        @testset "Single step accuracy" begin
            u1 = MFGnet.step(stepper, odefun_simple, nothing, u0, nothing, t0, t1)

            exact = u0 .* exp(-0.1)
            @test u1[1] ≈ exact[1] atol=1e-6  # RK4 should be very accurate
        end

        @testset "Fourth-order convergence" begin
            # Test convergence rate: should be fourth-order
            h_values = [0.1, 0.05, 0.025]
            errors = Float64[]

            for h in h_values
                u = copy(u0)
                nt = Int(1.0 / h)
                for i=1:nt
                    u = MFGnet.step(stepper, odefun_simple, nothing, u, nothing, (i-1)*h, i*h)
                end
                exact = u0 .* exp(-1.0)
                push!(errors, norm(u - exact))
            end

            # Fourth-order method: error ∝ h^4, so error_ratio ≈ 16 when h is halved
            ratio1 = errors[1] / errors[2]
            ratio2 = errors[2] / errors[3]

            @test 8.0 < ratio1 < 32.0  # Expect around 16
            @test 8.0 < ratio2 < 32.0
        end

        @testset "Comparison with RK1" begin
            # RK4 should be much more accurate than RK1 for the same step size
            h = 0.1
            u_rk1 = copy(u0)
            u_rk4 = copy(u0)

            for i=1:10
                u_rk1 = MFGnet.step(RK1Step(), odefun_simple, nothing, u_rk1, nothing, (i-1)*h, i*h)
                u_rk4 = MFGnet.step(RK4Step(), odefun_simple, nothing, u_rk4, nothing, (i-1)*h, i*h)
            end

            exact = u0 .* exp(-1.0)
            error_rk1 = norm(u_rk1 - exact)
            error_rk4 = norm(u_rk4 - exact)

            @test error_rk4 < error_rk1 / 100  # RK4 should be much better
        end
    end

    @testset "integrate Function" begin
        odefun_linear(J, u, Θ, t) = -2*u

        u0 = [2.0, 3.0]
        tspan = [0.0, 1.0]
        stepper = RK4Step()

        @testset "Basic integration" begin
            U = integrate(stepper, odefun_linear, nothing, u0, nothing, tspan, 10)

            @test U isa AbstractVector
            @test length(U) == length(u0)
        end

        @testset "Final value correctness" begin
            U = integrate(stepper, odefun_linear, nothing, u0, nothing, tspan, 100)

            exact = u0 .* exp(-2.0)
            @test U ≈ exact atol=1e-4
        end
    end

    @testset "integrate2 Function" begin
        odefun_simple(J, u, Θ, t) = -u

        u0 = reshape([1.0], 1, 1)  # Matrix format: (state_dim, n_trajectories)
        tspan = [0.0, 1.0]
        N = 10
        stepper = RK4Step()

        @testset "State storage" begin
            UArray = integrate2(stepper, odefun_simple, nothing, u0, nothing, tspan, N)

            @test UArray isa AbstractArray
            @test ndims(UArray) == 3
            @test size(UArray, 1) == size(u0, 1)  # State dimension
            @test size(UArray, 2) == size(u0, 2)  # Number of trajectories
            @test size(UArray, 3) == N+1  # All intermediate states
        end

        @testset "Trajectory correctness" begin
            UArray = integrate2(stepper, odefun_simple, nothing, u0, nothing, tspan, N)

            # First state should be initial condition
            @test UArray[:, :, 1] == u0

            # Last state should approximate exact solution
            exact = u0 .* exp(-1.0)
            @test UArray[:, :, end] ≈ exact atol=1e-4
        end
    end

    @testset "Nonlinear ODE" begin
        # Test on nonlinear ODE: du/dt = u^2, u(0) = 0.5
        # Exact solution: u(t) = 1/(2 - t)
        odefun_nonlinear(J, u, Θ, t) = u.^2

        u0 = [0.5]
        tspan = [0.0, 0.5]
        stepper = RK4Step()

        @testset "Nonlinear integration" begin
            U = integrate(stepper, odefun_nonlinear, nothing, u0, nothing, tspan, 50)

            exact = 1.0 / (2.0 - 0.5)
            @test U[1, end] ≈ exact rtol=1e-3
        end
    end

    @testset "Vector ODE System" begin
        # Test on system: du1/dt = u2, du2/dt = -u1 (harmonic oscillator)
        # Exact: u1(t) = cos(t), u2(t) = -sin(t) for u0 = [1, 0]
        function odefun_oscillator(J, u, Θ, t)
            return [u[2], -u[1]]
        end

        u0 = [1.0, 0.0]
        tspan = [0.0, 2*π]
        stepper = RK4Step()

        @testset "Multi-dimensional system" begin
            U = integrate(stepper, odefun_oscillator, nothing, u0, nothing, tspan, 100)

            # After one period, should return to initial condition
            @test U[:, end] ≈ u0 atol=1e-3
        end
    end
end
