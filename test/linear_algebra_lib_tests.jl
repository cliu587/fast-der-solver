@testset "linear-algebra-lib" begin
    @testset "lin_solve computes a nullspace basis" begin
        M = [
            1.0 2.0 3.0;
            2.0 4.0 6.0
        ]

        nullspace_basis = lin_solve(M)
        @test size(nullspace_basis, 1) == 3
        @test size(nullspace_basis, 2) == 2
        @test isapprox(M * nullspace_basis, zeros(2, 2); atol=1e-8, rtol=1e-8)
    end

    @testset "linear_equals_affine solves the unique feasible family" begin
        M = [
            1.0 0.0;
            0.0 1.0;
            1.0 1.0
        ]
        U_0 = [
            2.0 1.0;
            -1.0 3.0
        ]
        U_1 = [
            1.0 0.0;
            0.0 1.0
        ]
        U_2 = [
            -1.0 2.0;
            3.0 -2.0
        ]

        N_0 = M * U_0
        N_1 = M * U_1
        N_2 = M * U_2

        lifted_offset, lifted_directions = linear_equals_affine(
            M,
            N_0,
            [N_1, N_2]
        )

        @test isapprox(lifted_offset, U_0; atol=1e-8, rtol=1e-8)
        @test length(lifted_directions) == 2
        @test isapprox(lifted_directions[1], U_1; atol=1e-8, rtol=1e-8)
        @test isapprox(lifted_directions[2], U_2; atol=1e-8, rtol=1e-8)

        combined_solution = lifted_offset + 2.0 * lifted_directions[1] - 3.0 * lifted_directions[2]
        combined_rhs = N_0 + 2.0 * N_1 - 3.0 * N_2
        @test isapprox(M * combined_solution, combined_rhs; atol=1e-8, rtol=1e-8)
    end

    @testset "linear_equals_affine enforces the stated assumptions" begin
        M_rank_deficient = [
            1.0 0.0;
            2.0 0.0;
            3.0 0.0
        ]
        N_0 = zeros(3, 1)
        N_1 = zeros(3, 1)
        @test_throws ErrorException linear_equals_affine(
            M_rank_deficient,
            N_0,
            [N_1]
        )

        M = [
            1.0 0.0;
            0.0 1.0;
            0.0 0.0
        ]
        N_infeasible = reshape([
            0.0;
            0.0;
            1.0
        ], 3, 1)
        @test_throws ErrorException linear_equals_affine(
            M,
            zeros(3, 1),
            [N_infeasible]
        )
    end
end
