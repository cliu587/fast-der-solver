@testset "quicksylver-lib" begin
    @testset "solve_dense_sylvester_system returns affine frame points that solve the dense system" begin
        R, S, T, X, Y = sylvester_instance(4; seed=2)
        M = sylvester_system_matrix(R, S)
        rhs = rhs_vector(T)
        solution_frame = solve_dense_sylvester_system(R, S, T)

        @test length(solution_frame) >= 1
        @test check_sylvester_solution(R, S, T, [(X, Y)]; faster_randomized_check=false)

        for (X_frame, Y_frame) in solution_frame
            solution_vector = vcat(vec(X_frame), vec(Y_frame))
            @test isapprox(M * solution_vector, rhs; atol=1e-8, rtol=1e-8)
        end
    end

    @testset "sylvester_solver returns valid full solutions on a known solvable instance" begin
        R, S, T, X, Y = sylvester_instance(4; seed=3)
        @test check_sylvester_solution(R, S, T, [(X, Y)]; faster_randomized_check=false)

        solution_frame = sylvester_solver(R, S, T; faster_randomized_check=false)

        @test length(solution_frame) >= 1
        @test any(norm(solution[1]) + norm(solution[2]) > 1e-8 for solution in solution_frame)
        @test check_sylvester_solution(R, S, T, solution_frame; faster_randomized_check=false)
    end

    @testset "sylvester_solver handles a rectangular solvable instance" begin
        R, S, T, X, Y = rectangular_sylvester_instance(seed=4)
        @test check_sylvester_solution(R, S, T, [(X, Y)]; faster_randomized_check=false)

        solution_frame = sylvester_solver(R, S, T; faster_randomized_check=false)

        @test length(solution_frame) >= 1
        @test check_sylvester_solution(R, S, T, solution_frame; faster_randomized_check=false)
    end

    @testset "sylvester_solver errors when the unique-lift assumption fails" begin
        R, S, T = sylvester_unique_lift_failure_instance()
        @test_throws ErrorException sylvester_solver(
            R,
            S,
            T;
            double_restriction_size_override=(2, 2),
            faster_randomized_check=false
        )
    end
end
