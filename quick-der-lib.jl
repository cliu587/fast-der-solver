using Logging
using LinearAlgebra
using TimerOutputs
include("./linear-algebra-lib.jl")
to = TimerOutput()

function derivation_system_matrix(R, S, T)
    r_dim, b, c = size(R)
    a, s_dim, _ = size(S)
    _, _, t_dim = size(T)

    # rows first stride by c then b.
    R_flat = reshape(permutedims(R, (3, 2, 1)), (c * b, r_dim))

    I_a = Matrix{Float64}(I, a, a)
    # rows first stide by c then b then a, columns first stride by r_dim then a.
    R_mat = kron(I_a, R_flat)

    I_b = Matrix{Float64}(I, b, b)
    S_slices = [Transpose(S[i, :, :]) for i in 1:a] # Rows stride by c then a
    # Rows stride by c then b then a, columns stride by s_dim then b
    S_mat = vcat([kron(I_b, M) for M in S_slices]...)

    T_mat_reshape = reshape(permutedims(T, (2, 1, 3)), (b * a, t_dim))
    I_c = Matrix(I, c, c)

    # Rows stride by c then b then a, columns stride by c then t_dim
    T_mat = kron(T_mat_reshape, I_c)

    # 4) Stack the three pieces into one big M
    return hcat(R_mat, S_mat, -T_mat)
end

function solve_dense_derivation_system(R, S, T)
    M = @timeit to "dense derivation matrix" derivation_system_matrix(R, S, T)
    rows, cols = size(M)
    @info "Derivation-system matrix size: $(size(M))"
    if cols > rows
      @warn "Derivation-system matrix has more columns than rows, so the restricted system is underconstrained."
    end
    return @timeit to "dense derivation solve" lin_solve(M)
end

function outer_action(T, Z)
    a, b, c = size(T); z_cols = size(Z, 2)
    return reshape(reshape(T, (a * b, c)) * Z, (a, b, z_cols))
end

function check_derivation_solution(
    R, S, T, derivation_basis;
    # If true, check one random basis vector on one random slice.
    # Otherwise, check every basis vector on every slice.
    faster_randomized_check=true)

    if isempty(derivation_basis)
        return true
    end

    function basis_vector_is_correct_on_slice(X, Y, Z, slice_index)
        R_slice = R[:, :, slice_index]
        S_slice = S[:, :, slice_index]
        T_slice = outer_action(T, Z)[:, :, slice_index]
        return isapprox(X * R_slice + S_slice * Y - T_slice, zeros(size(T_slice)); rtol=1e-6, atol=1e-6)
    end

    function basis_vector_is_correct_on_all_slices(X, Y, Z)
        _, _, c = size(R)
        T_Z = outer_action(T, Z)
        return all(
            isapprox(X * R[:, :, i] + S[:, :, i] * Y - T_Z[:, :, i], zeros(size(R[:, :, i])); rtol=1e-6, atol=1e-6)
            for i in 1:c
        )
    end

    if faster_randomized_check
        _, _, c = size(R)
        slice_index = rand(1:c)
        basis_vector_index = rand(1:length(derivation_basis))
        X, Y, Z = derivation_basis[basis_vector_index]
        return basis_vector_is_correct_on_slice(X, Y, Z, slice_index)
    else
        return all(basis_vector_is_correct_on_all_slices(X, Y, Z) for (X, Y, Z) in derivation_basis)
    end
end

function select_prefix_restriction_sizes(R, S, T)
    r_dim, b, c = size(R); a, s_dim, _ = size(S); _, _, t_dim = size(T)
    cubic_extension_dim = max(r_dim, s_dim, t_dim)
    shared_unknown_block_size = ceil(Int, sqrt(3.0 * cubic_extension_dim^3))

    a_prime = min(a, ceil(Int, shared_unknown_block_size / r_dim))
    b_prime = min(b, ceil(Int, shared_unknown_block_size / s_dim))
    c_prime = min(c, ceil(Int, shared_unknown_block_size / t_dim))

    num_equations = a_prime * b_prime * c_prime
    num_unknowns = a_prime * r_dim + b_prime * s_dim + c_prime * t_dim
    if num_equations < num_unknowns
        error("The cubic-extension restriction sizes do not make TripleRestrictedDer generically full column rank.")
    end

    return a_prime, b_prime, c_prime
end

function solve_and_lift_derivation_system(R, S, T; a_prime, b_prime, c_prime)

    r_dim, b, c = size(R); a, s_dim, _ = size(S); _, _, t_dim = size(T)

    I = 1:a_prime; I_hat = a_prime + 1:a; I_hat_dim = a - a_prime
    J = 1:b_prime; J_hat = b_prime + 1:b; J_hat_dim = b - b_prime
    K = 1:c_prime; K_hat = c_prime + 1:c; K_hat_dim = c - c_prime

    R_JK = R[:, J, K]; S_IK = S[I, :, K]; T_IJ = T[I, J, :]

    DerTR_basis = @timeit to "compute TripleRestrictedDer" solve_dense_derivation_system(
        R_JK, S_IK, T_IJ
    )

    lift_setup = begin_timed_section!(to, "restricted lift setup")

    M_C = Transpose(reshape(R[:, J, K], (r_dim, b_prime * c_prime)))
    T_I_hat_J = T[I_hat, J, :]; S_I_hat_K = S[I_hat, :, K]

    M_R = reshape(permutedims(S[I, :, K], (1, 3, 2)), (a_prime * c_prime, s_dim))
    T_I_J_hat = T[I, J_hat, :]; R_J_hat_K = R[:, J_hat, K]

    M_D = reshape(T[I, J, :], (a_prime * b_prime, t_dim))
    R_J_K_hat = R[:, J, K_hat]; S_I_K_hat = S[I, :, K_hat]

    end_timed_section!(to, lift_setup)

    function unpack_DerTR_basis_vector(DerTR_basis_vector)
        x_stop = a_prime * r_dim; y_stop = x_stop + b_prime * s_dim

        X_I = permutedims(reshape(DerTR_basis_vector[1:x_stop], (r_dim, a_prime)), (2, 1))
        Y_J = reshape(DerTR_basis_vector[x_stop + 1:y_stop], (s_dim, b_prime))
        Z_K = permutedims(reshape(DerTR_basis_vector[y_stop + 1:end], (c_prime, t_dim)), (2, 1))

        return X_I, Y_J, Z_K
    end

    function solve_restricted_lift(M, rhs_row_count, hat_dim, unknown_dim, rhs_directions)
        if hat_dim == 0
            return [zeros(Float64, unknown_dim, 0) for _ in 1:length(rhs_directions)]
        end

        _, directions = linear_equals_affine(M, zeros(Float64, rhs_row_count, hat_dim), rhs_directions)
        return directions
    end

    function N_C(Y_J, Z_K)
        if I_hat_dim == 0
            return zeros(Float64, b_prime * c_prime, 0)
        end

        S_Y = Transpose(hcat([S_I_hat_K[:, :, k] * Y_J for k in 1:c_prime]...))
        T_Z_matrix = reshape(T_I_hat_J, (I_hat_dim * b_prime, t_dim)) * Z_K
        T_Z = reshape(
            permutedims(reshape(T_Z_matrix, (I_hat_dim, b_prime, c_prime)), (2, 3, 1)),
            (b_prime * c_prime, I_hat_dim)
        )
        return T_Z - S_Y
    end

    function N_R(X_I, Z_K)
        if J_hat_dim == 0
            return zeros(Float64, a_prime * c_prime, 0)
        end

        X_R = vcat([X_I * R_J_hat_K[:, :, k] for k in 1:c_prime]...)
        T_Z_matrix = reshape(T_I_J_hat, (a_prime * J_hat_dim, t_dim)) * Z_K
        T_Z = reshape(
            permutedims(reshape(T_Z_matrix, (a_prime, J_hat_dim, c_prime)), (1, 3, 2)),
            (a_prime * c_prime, J_hat_dim)
        )
        return T_Z - X_R
    end

    function N_D(X_I, Y_J)
        if K_hat_dim == 0
            return zeros(Float64, a_prime * b_prime, 0)
        end

        X_R = vcat([X_I * R_J_K_hat[:, j, :] for j in 1:b_prime]...)
        S_Y_slices = [Transpose(Y_J) * S_I_K_hat[i, :, :] for i in 1:a_prime]
        S_Y = reshape(
            permutedims(reshape(vcat(S_Y_slices...), b_prime, a_prime, K_hat_dim), (2, 1, 3)),
            (a_prime * b_prime, K_hat_dim)
        )
        return X_R + S_Y
    end

    restricted_lifts = begin_timed_section!(to, "restricted lifts")

    DerTR_basis_size = size(DerTR_basis, 2)
    @info "Lifting all $DerTR_basis_size basis vectors of DerTR (full basis size: $(size(DerTR_basis)))."

    if DerTR_basis_size == 0
        end_timed_section!(to, restricted_lifts)
        @info "dim DerTR = 0, so returning an empty derivation basis."
        return []
    end

    derivation_basis = Vector{NTuple{3, Matrix{Float64}}}()
    DerTR_basis_vectors = [unpack_DerTR_basis_vector(DerTR_basis[:, i]) for i in 1:DerTR_basis_size]

    col_rhs_directions = [N_C(Y_J, Z_K) for (_, Y_J, Z_K) in DerTR_basis_vectors]
    row_rhs_directions = [N_R(X_I, Z_K) for (X_I, _, Z_K) in DerTR_basis_vectors]
    depth_rhs_directions = [N_D(X_I, Y_J) for (X_I, Y_J, _) in DerTR_basis_vectors]

    X_hat_directions = solve_restricted_lift(M_C, b_prime * c_prime, I_hat_dim, r_dim, col_rhs_directions)
    Y_hat_directions = solve_restricted_lift(M_R, a_prime * c_prime, J_hat_dim, s_dim, row_rhs_directions)
    Z_hat_directions = solve_restricted_lift(M_D, a_prime * b_prime, K_hat_dim, t_dim, depth_rhs_directions)

    for (i, (X_I, Y_J, Z_K)) in enumerate(DerTR_basis_vectors)
        X = vcat(X_I, Transpose(X_hat_directions[i]))
        Y = hcat(Y_J, Y_hat_directions[i])
        Z = hcat(Z_K, Z_hat_directions[i])

        push!(derivation_basis, (X, Y, Z))
    end

    end_timed_section!(to, restricted_lifts)
    @info "lifted derivation basis size: $(size(derivation_basis))"
    return derivation_basis
end

function derivation_solver(R,S,T;
    triple_restriction_size_override=nothing,
    faster_randomized_check=true)

    if triple_restriction_size_override !== nothing
        a_prime, b_prime, c_prime = triple_restriction_size_override
    else
        a_prime, b_prime, c_prime = select_prefix_restriction_sizes(R, S, T)
    end
    @info "Selected prefix restriction sizes a'=$a_prime, b'=$b_prime, c'=$c_prime."

    derivation_basis = @timeit to "solve-and-lift derivation system" solve_and_lift_derivation_system(
        R, S, T; a_prime=a_prime, b_prime=b_prime, c_prime=c_prime
    )

    if isempty(derivation_basis)
        return derivation_basis
    end

    verified = @timeit to "verify derivation solution" check_derivation_solution(
        R, S, T, derivation_basis; faster_randomized_check=faster_randomized_check
    )
    if !verified
        error("Derivation solver did not find a correct solution triple. Retry with larger a',b',c'!")
    end

    return derivation_basis
end
