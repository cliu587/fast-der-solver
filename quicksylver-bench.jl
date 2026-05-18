using Logging
include("./quicksylver-lib.jl")
include("./plotting-lib.jl")

function timed_sylvester_run(solve_instance, R, S, T)
    TimerOutputs.reset_timer!(sylver_to)
    GC.gc()
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
        R, S, T = sylvester_fixture(10)
        solve_dense_sylvester_system(R, S, T)

        R, S, T = sylvester_fixture(10)
        sylvester_solver(R, S, T)
    end
end

function bench_sylvester(;slow_sizes, fast_sizes, n_trials)
    warm_up_sylvester_benchmark()

    slow_results = Dict{Int, Vector{Float64}}()
    fast_results = Dict{Int, Vector{Float64}}()

    for size in slow_sizes
        @info "slow solver for n=$size"
        elapsed_seconds = Float64[]
        for trial in 1:n_trials
            R, S, T = sylvester_fixture(size)
            push!(elapsed_seconds, timed_sylvester_run(solve_dense_sylvester_system, R, S, T))
        end
        slow_results[size] = elapsed_seconds
    end

    for size in fast_sizes
        @info "quick Sylvester solver for n=$size"
        elapsed_seconds = Float64[]
        for trial in 1:n_trials
            R, S, T = sylvester_fixture(size)
            push!(elapsed_seconds, timed_sylvester_run(sylvester_solver, R, S, T))
        end
        fast_results[size] = elapsed_seconds
    end

    return slow_results, fast_results
end

function print_single_csv_summary(results; io=stdout)
    println(io, "n,trial,time")
    for n in sort(collect(keys(results)))
        for (trial, elapsed_seconds) in enumerate(results[n])
            println(io, "$(n),$(trial),$(elapsed_seconds)")
        end
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

function sylvester_benchmark_sizes(mode)
    if mode == :short
        slow_sizes = [5, 8, 12, 18]
        fast_sizes = vcat(slow_sizes, [25, 35, 50, 70])
        return (; slow_sizes, fast_sizes)
    end

    if mode == :long
        slow_sizes = [5, 8, 12, 18, 25, 29, 31]
        fast_sizes = vcat(slow_sizes, [35, 50, 70, 100, 140, 200, 280, 400, 560, 700, 800])
        return (; slow_sizes, fast_sizes)
    end

    error("Unknown benchmark mode $mode. Use :short or :long.")
end

# Only run benchmark if using directly.
if abspath(PROGRAM_FILE) == @__FILE__
    # Short: does everything that long does but quickly so I can test stuff.
    mode = isempty(ARGS) ? :short : Symbol(ARGS[1])
    sizes = sylvester_benchmark_sizes(mode)

    slow_results, fast_results = bench_sylvester(
        slow_sizes=sizes.slow_sizes,
        fast_sizes=sizes.fast_sizes,
        n_trials=5
    )
    println()
    print_csv_summary(slow_results, fast_results)
    write_csv_summary("quicksylver-results.csv", slow_results, fast_results)
    plot_benchmark_results(
        slow_results,
        fast_results,
        "quicksylver-results.png";
        title="Solving a Simultaneous Sylvester System with Regularity Conditions"
    )
end
