using Logging
using LinearAlgebra
using TimerOutputs

include("./linear-algebra-lib.jl")

sylver_to = TimerOutput()

function rhs_vector(T)
    return vcat((vec(T[:, :, k]) for k in axes(T, 3))...)
end

function sylvester_system_matrix(R, S)
    r_dim, b, c = size(R)
    a, s_dim, c_S = size(S)
    if c_S != c
        throw(DimensionMismatch("R and S must have the same number of slices."))
    end

    I_a = Matrix(I, a, a); I_b = Matrix(I, b, b)
    sylvester_blocks = [
        hcat(kron(Transpose(R[:, :, k]), I_a), kron(I_b, S[:, :, k]))
        for k in axes(R, 3)
    ]

    return vcat(sylvester_blocks...)
end

function solve_dense_sylvester_system(R, S, T; atol=1e-6)
    r_dim, b, c = size(R); a, s_dim, c_S = size(S); a_T, b_T, c_T = size(T)
    if (a_T, b_T, c_T) != (a, b, c) || c_S != c
        throw(DimensionMismatch("R, S, and T must have compatible Sylvester dimensions."))
    end

    M = @timeit sylver_to "dense sylvester matrix creation" sylvester_system_matrix(R, S)
    rhs = rhs_vector(T)
    rows, cols = size(M)

    @info "Dense Sylvester matrix size: $(size(M))"
    if cols > rows
        @warn "Dense Sylvester matrix has more columns than rows, so the restricted system is underconstrained."
    end

    dense_solution = @timeit sylver_to "dense sylvester linear solve" lin_solve(M, rhs; atol=atol)
    if isnothing(dense_solution)
        @info "Dense Sylvester system has no solution."
        return NTuple{2, Matrix{Float64}}[]
    end
    particular_solution, sylvester_nullspace_basis = dense_solution

    function unpack_DR_frame_point(solution_vector)
        x_stop_ind = a * r_dim
        X = reshape(solution_vector[1:x_stop_ind], (a, r_dim))
        Y = reshape(solution_vector[x_stop_ind + 1:end], (s_dim, b))
        return X, Y
    end

    return [
        unpack_DR_frame_point(particular_solution);
        [unpack_DR_frame_point(particular_solution + sylvester_nullspace_basis[:, i]) for i in axes(sylvester_nullspace_basis, 2)]
    ]
end

function check_sylvester_solution(R, S, T, solution_frame; faster_randomized_check=false)
    if isempty(solution_frame)
        return true
    end

    function frame_point_is_correct_on_slice(X, Y, slice_index)
        return isapprox(
            X * R[:, :, slice_index] + S[:, :, slice_index] * Y,
            T[:, :, slice_index];
            atol=1e-6,
            rtol=1e-6
        )
    end

    function frame_point_is_correct_on_all_slices(X, Y)
        return all(frame_point_is_correct_on_slice(X, Y, slice_index) for slice_index in axes(R, 3))
    end

    if faster_randomized_check
        slice_index = rand(axes(R, 3))
        frame_point_index = rand(eachindex(solution_frame))
        X, Y = solution_frame[frame_point_index]
        return frame_point_is_correct_on_slice(X, Y, slice_index)
    end

    return all(frame_point_is_correct_on_all_slices(X, Y) for (X, Y) in solution_frame)
end

function select_double_restriction_sizes(R, S, T)
    r_dim, b, c = size(R); a, s_dim, c_S = size(S); a_T, b_T, c_T = size(T)
    if (a_T, b_T, c_T) != (a, b, c) || c_S != c
        throw(DimensionMismatch("R, S, and T must have compatible Sylvester dimensions."))
    end

    rs_max = max(r_dim, s_dim)
    # Let a'r and  b's both be equal to some number n - this is the goal we try to solve for
    # where we set a' and b' in a way to equalize the number of slices.
    # We have a'b'c = (n^2 / rs_max^2)*c and a'r + b's = 2n, so n = 2 rs_max^2 / c.
    balanced_block_size = ceil(Int, 2.0 * rs_max^2 / c)

    a_prime = min(a, max(1, ceil(Int, balanced_block_size / r_dim) + 1))
    b_prime = min(b, max(1, ceil(Int, balanced_block_size / s_dim) + 1))

    num_equations = a_prime * b_prime * c
    num_unknowns = a_prime * r_dim + b_prime * s_dim
    if num_equations < num_unknowns
        error("The restriction sizes do not make DoubleRestrictedSylvester generically full column rank.")
    end

    return a_prime, b_prime
end

function solve_and_lift_sylvester_system(R, S, T; a_prime, b_prime)
    r_dim, b, c = size(R); a, s_dim, _ = size(S)

    I = 1:a_prime; I_hat = a_prime + 1:a 
    J = 1:b_prime; J_hat = b_prime + 1:b

    DR_frame = @timeit sylver_to "compute DoubleRestrictedSylvester" solve_dense_sylvester_system(
        R[:, J, :], S[I, :, :], T[I, J, :]
    )

    if isempty(DR_frame)
        return NTuple{2, Matrix{Float64}}[]
    end

    lift_setup = begin_timed_section!(sylver_to, "restricted lift setup")
    M_R = vcat([S[I, :, k] for k in 1:c]...)
    M_C = vcat([Transpose(R[:, J, k]) for k in 1:c]...)
    end_timed_section!(sylver_to, lift_setup)

    function N_R(X_I)
        return vcat([T[I, J_hat, k] - X_I * R[:, J_hat, k] for k in 1:c]...)
    end

    function N_C(Y_J)
        return vcat([Transpose(T[I_hat, J, k] - S[I_hat, :, k] * Y_J) for k in 1:c]...)
    end

    restricted_lifts = begin_timed_section!(sylver_to, "restricted lifts")

    DR_dimension = length(DR_frame) - 1
    @info "Lifting the $(DR_dimension + 1)-point affine frame of DoubleRestrictedSylvester."

    X_I_0, Y_J_0 = DR_frame[1]
    row_rhs_0, col_rhs_0 = N_R(X_I_0), N_C(Y_J_0)
    row_rhs_directions = [N_R(X_I) - row_rhs_0 for (X_I, _) in DR_frame[2:end]]
    col_rhs_directions = [N_C(Y_J) - col_rhs_0 for (_, Y_J) in DR_frame[2:end]]

    Y_hat_0, Y_hat_directions = linear_equals_affine(M_R, row_rhs_0, row_rhs_directions)
    X_hat_0, X_hat_directions = linear_equals_affine(M_C, col_rhs_0, col_rhs_directions)

    solution_frame = NTuple{2, Matrix{Float64}}[]
    push!(solution_frame, (vcat(X_I_0, Transpose(X_hat_0)), hcat(Y_J_0, Y_hat_0)))

    for i in 1:DR_dimension
        X_I_i, Y_J_i = DR_frame[i + 1]
        X_i = vcat(X_I_i, Transpose(X_hat_0 + X_hat_directions[i]))
        Y_i = hcat(Y_J_i, Y_hat_0 + Y_hat_directions[i])
        push!(solution_frame, (X_i, Y_i))
    end

    end_timed_section!(sylver_to, restricted_lifts)
    return solution_frame
end

function sylvester_solver(R, S, T; double_restriction_size_override=nothing, faster_randomized_check=false)
    if double_restriction_size_override !== nothing
        a_prime, b_prime = double_restriction_size_override
        a, b, _ = size(T)
        if !(1 <= a_prime <= a && 1 <= b_prime <= b)
            error("The DoubleRestrictedSylvester override must satisfy 1 <= a' <= a and 1 <= b' <= b.")
        end
    else
        a_prime, b_prime = select_double_restriction_sizes(R, S, T)
    end
    @info "Selected restriction sizes a'=$a_prime, b'=$b_prime."

    solution_frame = @timeit sylver_to "solve-and-lift system" solve_and_lift_sylvester_system(R, S, T; a_prime=a_prime, b_prime=b_prime)

    if isempty(solution_frame)
        return solution_frame
    end

    verified = @timeit sylver_to "verify solution" check_sylvester_solution(
        R, S, T, solution_frame; faster_randomized_check=faster_randomized_check
    )
    if !verified
        error("Sylvester solver did not find a correct affine frame. Retry with larger a',b'!")
    end

    return solution_frame
end
