using Logging
using Statistics
include("./quick-der-lib.jl")
include("./plotting-lib.jl")

function timed_derivation_run(solve_instance, R, S, T)
    TimerOutputs.reset_timer!(to)
    GC.gc()
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
        R, S, T = der_fixture(10)
        solve_dense_derivation_system(R, S, T)

        R, S, T = der_fixture(10)
        derivation_solver(R, S, T)
    end
end

function bench_derivation(;slow_sizes, fast_sizes, n_trials)
    warm_up_derivation_benchmark()

    slow_results = Dict{Int, Vector{Float64}}()
    fast_results = Dict{Int, Vector{Float64}}()
    num_with_nontrivial_solution = round(Int, n_trials / 2, RoundUp)

    for size in slow_sizes
        @info "slow solver for $size"
        elapsed_seconds = Float64[]
        for trial in 1:n_trials
            R, S, T = der_fixture(size)
            push!(elapsed_seconds, timed_derivation_run(solve_dense_derivation_system, R, S, T))
        end
        slow_results[size] = elapsed_seconds
    end

    for size in fast_sizes
        @info "quick derivation solver for $size"
        elapsed_seconds = Float64[]
        for trial in 1:n_trials
            R, S, T = der_fixture(size)
            push!(elapsed_seconds, timed_derivation_run(derivation_solver, R, S, T))
        end
        fast_results[size] = elapsed_seconds
    end

    return slow_results, fast_results
end

function print_single_csv_summary(results; io=stdout)
    println(io, "n,time")
    for n in sort(collect(keys(results)))
        println(io, "$(n),$(median(results[n]))")
    end
end

function print_csv_summary(slow_results, fast_results; io=stdout)
    println(io, "slow solver performance")
    print_single_csv_summary(slow_results; io=io)
    println(io)
    println(io, "fast solver performance")
    print_single_csv_summary(fast_results; io=io)
end

function write_csv_summary(path, slow_results, fast_results)
    open(path, "w") do io
        print_csv_summary(slow_results, fast_results; io=io)
    end
end

# Only run benchmark if using directly.
if abspath(PROGRAM_FILE) == @__FILE__
    slow_sizes = [5, 8, 12, 18, 25]
    fast_sizes = vcat(slow_sizes, [35, 50, 70, 100])

    slow_sizes_BIG = [5, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 33]
    fast_sizes_BIG = vcat(slow_sizes_BIG, [30, 40, 55, 75, 100, 130, 145, 164])


    slow_results, fast_results = bench_derivation(
        slow_sizes=slow_sizes,
        fast_sizes=fast_sizes,
        # slow_sizes=[5],
        # fast_sizes=[10],
        n_trials=5
    )
    println()
    print_csv_summary(slow_results, fast_results)
    write_csv_summary("der-results.csv", slow_results, fast_results)
    plot_benchmark_results(
        slow_results,
        fast_results,
        "der-results.png";
        title="Solving a Derivation System with Regularity Conditions"
    )
end
