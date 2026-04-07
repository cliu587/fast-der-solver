using Logging
using Statistics
include("./quick-der-lib.jl")

function timed_derivation_run(solve_instance, R, S, T)
    TimerOutputs.reset_timer!(to)
    elapsed_seconds = @elapsed solve_instance(R, S, T)
    @info "time: $elapsed_seconds"
    show(to)
    println()
    return elapsed_seconds
end

function der_fixture(n; has_nontrivial_solution=true)
    R = randn(n, n, n)
    S = randn(n, n, n)
    if !has_nontrivial_solution
        return R, S, randn(n, n, n)
    end

    X = randn(n, n); Y = randn(n, n); Z = randn(n, n)
    XR_plus_SY = stack([X * R[:, :, i] + S[:, :, i] * Y for i in 1:n])
    T = reshape(reshape(XR_plus_SY, (n * n, n)) / Z, (n, n, n))
    return R, S, T
end

function warm_up_derivation_benchmark()
    with_logger(NullLogger()) do
        R, S, T = der_fixture(10; has_nontrivial_solution=true)
        solve_dense_derivation_system(R, S, T)

        R, S, T = der_fixture(10; has_nontrivial_solution=true)
        derivation_solver(R, S, T)
    end
end

function bench_derivation(;slow_solver_sizes, fast_solver_sizes, n_trials)
    warm_up_derivation_benchmark()

    slow_results = Dict{Int, Vector{Float64}}()
    fast_results = Dict{Int, Vector{Float64}}()
    num_with_nontrivial_solution = round(Int, n_trials / 2, RoundUp)

    for size in slow_solver_sizes
        @info "slow solver for $size"
        elapsed_seconds = Float64[]
        for trial in 1:n_trials
            has_nontrivial_solution = trial <= num_with_nontrivial_solution
            R, S, T = der_fixture(size; has_nontrivial_solution=has_nontrivial_solution)
            push!(elapsed_seconds, timed_derivation_run(solve_dense_derivation_system, R, S, T))
        end
        slow_results[size] = elapsed_seconds
    end

    for size in fast_solver_sizes
        @info "quick derivation solver for $size"
        elapsed_seconds = Float64[]
        for trial in 1:n_trials
            has_nontrivial_solution = trial <= num_with_nontrivial_solution
            R, S, T = der_fixture(size; has_nontrivial_solution=has_nontrivial_solution)
            push!(elapsed_seconds, timed_derivation_run(derivation_solver, R, S, T))
        end
        fast_results[size] = elapsed_seconds
    end

    return slow_results, fast_results
end

function print_csv_summary(results)
    println("n,time")
    for n in sort(collect(keys(results)))
        println("$(n),$(mean(results[n]))")
    end
end

# Only run benchmark if using directly.
if abspath(PROGRAM_FILE) == @__FILE__
    slow_results, fast_results = bench_derivation(
        slow_solver_sizes=[5, 10, 15, 20, 23, 25],
        fast_solver_sizes=[10, 20, 30, 50, 75, 100],
        # slow_solver_sizes=[5],
        # fast_solver_sizes=[10],
        n_trials=2
    )
    println("\nslow solver performance")
    print_csv_summary(slow_results)
    println("\nfast solver performance")
    print_csv_summary(fast_results)
end
