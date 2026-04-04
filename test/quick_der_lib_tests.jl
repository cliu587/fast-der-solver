@testset "quick-der-lib" begin
    @testset "solve_dense_derivation_system returns DerTR basis vectors for the system matrix" begin
        R, S, T, _, _, _ = instance_with_identity_z(4; seed=1)
        M = derivation_system_matrix(R, S, T)
        DerTR_basis = solve_dense_derivation_system(R, S, T)

        @test size(DerTR_basis, 1) == size(M, 2)
        @test size(DerTR_basis, 2) >= 1
        @test isapprox(
            M * DerTR_basis,
            zeros(size(M, 1), size(DerTR_basis, 2));
            atol=1e-8,
            rtol=1e-8
        )
    end

    @testset "derivation_solver returns valid full solutions on a known solvable instance" begin
        R, S, T, X, Y, Z = instance_with_identity_z(4; seed=1)
        @test check_derivation_solution(R, S, T, [(X, Y, Z)]; faster_randomized_check=false)

        solution_basis = derivation_solver(
            R,
            S,
            T;
            faster_randomized_check=false
        )

        @test length(solution_basis) == 1
        @test any(norm(sol[1]) + norm(sol[2]) + norm(sol[3]) > 1e-8 for sol in solution_basis)
        @test check_derivation_solution(R, S, T, solution_basis; faster_randomized_check=false)
    end

    @testset "derivation_solver handles a rectangular solvable instance" begin
        R, S, T, X, Y, Z = rectangular_derivation_instance(seed=2)
        @test check_derivation_solution(R, S, T, [(X, Y, Z)]; faster_randomized_check=false)

        solution_basis = derivation_solver(R, S, T; faster_randomized_check=false)

        @test length(solution_basis) >= 1
        @test check_derivation_solution(R, S, T, solution_basis; faster_randomized_check=false)
    end

    @testset "derivation_solver handles a given square instance with invertible Z" begin
        R, S, T, X, Y, Z = square_derivation_instance_with_invertible_Z(6; seed=3)
        @test check_derivation_solution(R, S, T, [(X, Y, Z)]; faster_randomized_check=false)

        solution_basis = derivation_solver(R, S, T; faster_randomized_check=false)

        @test length(solution_basis) >= 1
        @test check_derivation_solution(R, S, T, solution_basis; faster_randomized_check=false)
    end

    @testset "undersized TripleRestrictedDer can fail even on a valid square instance" begin
        R, S, T, X, Y, Z = square_derivation_instance_with_invertible_Z(6; seed=1)
        @test check_derivation_solution(R, S, T, [(X, Y, Z)]; faster_randomized_check=false)

        @test_throws ErrorException derivation_solver(
            R,
            S,
            T;
            triple_restriction_size_override=(3, 3, 3),
            faster_randomized_check=false
        )

        solution_basis = derivation_solver(R, S, T; faster_randomized_check=false)

        @test length(solution_basis) >= 1
        @test check_derivation_solution(R, S, T, solution_basis; faster_randomized_check=false)
    end

    @testset "derivation_solver errors when the unique-lift assumption fails" begin
        R, S, T = full_rank_assumption_failure_instance()
        @test_throws ErrorException derivation_solver(
            R,
            S,
            T;
            triple_restriction_size_override=(2, 2, 2),
            faster_randomized_check=false
        )
    end
end
