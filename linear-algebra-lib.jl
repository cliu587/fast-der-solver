using LinearAlgebra

function lin_solve(M; atol=1e-6)
    M_matrix = Matrix{Float64}(M)
    return nullspace(M_matrix; atol=atol, rtol=atol)
end

function linear_equals_affine(M, N_0, N_directions; atol=1e-6)
    M_matrix, N_0_matrix = Matrix{Float64}(M), Matrix{Float64}(N_0)

    if rank(M_matrix; atol=atol, rtol=atol) < size(M_matrix, 2)
        error("LinearEqualsAffine assumes the linear system matrix has full column rank.")
    end

    rhs_row_count, rhs_column_count = size(M_matrix, 1), size(N_0_matrix, 2)
    if size(N_0_matrix, 1) != rhs_row_count
        throw(DimensionMismatch("Right-hand-side offset has incompatible row dimension."))
    end

    N_direction_matrices = [Matrix{Float64}(N_i) for N_i in N_directions]
    for N_i in N_direction_matrices
        if size(N_i) != (rhs_row_count, rhs_column_count)
            throw(DimensionMismatch("All right-hand-side direction matrices must have the same size as the offset."))
        end
    end

    M_left_kernel_basis = lin_solve(M_matrix'; atol=atol)
    feasibility_blocks = Matrix{Float64}[M_left_kernel_basis' * N_0_matrix]
    append!(feasibility_blocks, [M_left_kernel_basis' * N_i for N_i in N_direction_matrices])
    if any(!isapprox(block, zeros(size(block)); atol=atol, rtol=atol) for block in feasibility_blocks)
        error("LinearEqualsAffine assumes every parameter is feasible, but Theta_feas != Theta.")
    end

    M_left_inverse = pinv(M_matrix)
    U_0 = M_left_inverse * N_0_matrix
    U_directions = [M_left_inverse * N_i for N_i in N_direction_matrices]
    return U_0, U_directions
end
