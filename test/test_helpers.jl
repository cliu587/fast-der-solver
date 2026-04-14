using LinearAlgebra
using Random

function instance_with_identity_z(n; seed=1)
    Random.seed!(seed)
    R = rand(-2.0:2, (n, n, n))
    S = rand(-2.0:2, (n, n, n))
    X = rand(-2.0:2, (n, n))
    Y = rand(-2.0:2, (n, n))
    Z = Matrix{Float64}(I, n, n)
    T = stack([X * R[:, :, i] + S[:, :, i] * Y for i in 1:n])
    return R, S, T, X, Y, Z
end

function full_rank_assumption_failure_instance()
    n = 4
    R = zeros(Float64, n, n, n)
    S = zeros(Float64, n, n, n)

    R[:, :, 1] = [1 0 0 0; 0 1 0 0; 0 0 0 0; 0 0 0 0]
    R[:, :, 2] = [0 0 0 0; 0 1 0 1; 1 0 0 0; 0 0 0 0]
    R[:, :, 3] = [0 1 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0]
    R[:, :, 4] = [0 0 0 0; 1 0 0 0; 0 0 0 0; 0 0 0 0]

    S[:, :, 1] = [1 0 0 0; 0 1 0 0; 0 0 0 0; 0 0 0 0]
    S[:, :, 2] = [0 0 1 0; 0 0 0 1; 0 0 0 0; 0 0 0 0]

    T = copy(R)
    return R, S, T
end

function sylvester_instance(n; seed=1)
    Random.seed!(seed)
    R = rand(-2.0:2, (n, n, n))
    S = rand(-2.0:2, (n, n, n))
    X = rand(-2.0:2, (n, n))
    Y = rand(-2.0:2, (n, n))
    T = cat([X * R[:, :, k] + S[:, :, k] * Y for k in 1:n]...; dims=3)
    return R, S, T, X, Y
end

function rectangular_derivation_instance(; a=3, b=5, c=4, r=2, s=7, t=4, seed=1)
    Random.seed!(seed)
    R = rand(-2.0:2, (r, b, c))
    S = rand(-2.0:2, (a, s, c))
    X = rand(-2.0:2, (a, r))
    Y = rand(-2.0:2, (s, b))
    Z = Matrix{Float64}(I, t, c)
    T = cat([X * R[:, :, k] + S[:, :, k] * Y for k in 1:c]...; dims=3)
    return R, S, T, X, Y, Z
end

function square_derivation_instance_with_invertible_Z(n; seed=1)
    Random.seed!(seed)
    R = rand(-2.0:2, (n, n, n))
    S = rand(-2.0:2, (n, n, n))
    X = rand(-2.0:2, (n, n))
    Y = rand(-2.0:2, (n, n))

    Z = rand(-2.0:2, (n, n))
    while rank(Z) < n
        Z = rand(-2.0:2, (n, n))
    end

    XR_plus_SY = stack([X * R[:, :, k] + S[:, :, k] * Y for k in 1:n])
    T = reshape(reshape(XR_plus_SY, (n * n, n)) / Z, (n, n, n))
    return R, S, T, X, Y, Z
end

function rectangular_sylvester_instance(; a=3, b=5, c=4, r=2, s=7, seed=1)
    Random.seed!(seed)
    R = rand(-2.0:2, (r, b, c))
    S = rand(-2.0:2, (a, s, c))
    X = rand(-2.0:2, (a, r))
    Y = rand(-2.0:2, (s, b))
    T = cat([X * R[:, :, k] + S[:, :, k] * Y for k in 1:c]...; dims=3)
    return R, S, T, X, Y
end

function sylvester_unique_lift_failure_instance()
    n = 4
    R = zeros(Float64, n, n, n)
    S = zeros(Float64, n, n, n)
    T = zeros(Float64, n, n, n)

    R[:, :, 1] = [1 0 0 0; 0 1 0 0; 0 0 0 0; 0 0 0 0]
    R[:, :, 2] = [0 1 0 0; 1 0 0 0; 0 0 0 0; 0 0 0 0]
    S[:, :, 1] = [1 0 0 0; 0 1 0 0; 0 0 0 0; 0 0 0 0]
    S[:, :, 2] = [0 0 1 0; 0 0 0 1; 0 0 0 0; 0 0 0 0]

    return R, S, T
end
