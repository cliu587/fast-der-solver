using Logging
include("./quicksylver-lib.jl")
using Statistics

function timed_sylvester_run(solve_instance, R, S, T)
    TimerOutputs.reset_timer!(sylver_to)
    elapsed_seconds = @elapsed solve_instance(R, S, T)
    @info "time: $elapsed_seconds"
    show(sylver_to)
    println()
    return elapsed_seconds
end

function sylvester_fixture(n; has_solution=true)
    R = randn(n, n, n); S = randn(n, n, n)

    if !has_solution
        return R, S, randn(n, n, n)
    end

    X = randn(n, n); Y = randn(n, n)
    T = cat([X * R[:, :, k] + S[:, :, k] * Y for k in 1:n]...; dims=3)
    return R, S, T
end

# Warms up the JIT so the first trial isn't unnecessarily slow.
function warm_up_sylvester_benchmark()
    with_logger(NullLogger()) do
        R, S, T = sylvester_fixture(10; has_solution=true)
        solve_dense_sylvester_system(R, S, T)

        R, S, T = sylvester_fixture(10; has_solution=true)
        sylvester_solver(R, S, T)
    end
end

function bench_sylvester(;slow_sizes, fast_sizes, n_trials)
    warm_up_sylvester_benchmark()

    slow_results = Dict{Int, Vector{Float64}}()
    fast_results = Dict{Int, Vector{Float64}}()
    num_with_solution = round(Int, n_trials / 2, RoundUp)

    for size in slow_sizes
        @info "slow solver for n=$size"
        elapsed_seconds = Float64[]
        for trial in 1:n_trials
            has_solution = trial <= num_with_solution
            R, S, T = sylvester_fixture(size; has_solution=has_solution)
            push!(elapsed_seconds, timed_sylvester_run(solve_dense_sylvester_system, R, S, T))
        end
        slow_results[size] = elapsed_seconds
    end

    for size in fast_sizes
        @info "quick Sylvester solver for n=$size"
        elapsed_seconds = Float64[]
        for trial in 1:n_trials
            has_solution = trial <= num_with_solution
            R, S, T = sylvester_fixture(size; has_solution=has_solution)
            push!(elapsed_seconds, timed_sylvester_run(sylvester_solver, R, S, T))
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
    slow_results, fast_results = bench_sylvester(
        # slow_sizes=[5,10,15,20,23,25],
        # fast_sizes=[10, 20, 40, 60, 80, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700],
        slow_sizes=[],
        fast_sizes=[500],
        n_trials=2
    )
    println("\nslow solver performance")
    print_csv_summary(slow_results)
    println("\nfast solver performance")
    print_csv_summary(fast_results)
end
