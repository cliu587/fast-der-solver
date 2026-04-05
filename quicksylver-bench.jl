include("./quicksylver-lib.jl")

function show_sylvester_run_timer()
    show(sylver_to)
    println()
end

function timed_sylvester_run(solve_instance, R, S, T)
    TimerOutputs.reset_timer!(sylver_to)
    elapsed_seconds = @elapsed solve_instance(R, S, T)
    @info "time: $elapsed_seconds"
    show_sylvester_run_timer()
    return elapsed_seconds
end

function sylvester_with_solution(a_dim, b_dim, c_dim, r_dim, s_dim)
    R = rand(-10.0:10, (r_dim, b_dim, c_dim))
    S = rand(-10.0:10, (a_dim, s_dim, c_dim))
    X = rand(-10.0:10, (a_dim, r_dim))
    Y = rand(-10.0:10, (s_dim, b_dim))
    T = cat([X * R[:, :, k] + S[:, :, k] * Y for k in 1:c_dim]...; dims=3)
    return R, S, T
end

function bench_sylvester(;
    slow_sizes,
    fast_sizes,
    n_trials)
    slow_results = Dict{Int, Vector{Float64}}()
    fast_results = Dict{Int, Vector{Float64}}()

    for size in slow_sizes
        @info "slow solver for n=$size"
        R, S, T = sylvester_with_solution(size, size, size, size, size)
        slow_results[size] = [timed_sylvester_run(solve_dense_sylvester_system, R, S, T) for _ in 1:n_trials]
    end

    for size in fast_sizes
        @info "quick Sylvester solver for n=$size"
        elapsed_seconds = Float64[]
        for _ in 1:n_trials
            R, S, T = sylvester_with_solution(size, size, size, size, size)
            push!(elapsed_seconds, timed_sylvester_run(sylvester_solver, R, S, T))
        end
        fast_results[size] = elapsed_seconds
    end

    return slow_results, fast_results
end

# Only run benchmark if using directly.
if abspath(PROGRAM_FILE) == @__FILE__
    bench_sylvester(
        slow_sizes=[10, 20, 30, 40, 50, 60, 80, 100],
        fast_sizes=[10, 20, 40, 60, 80, 100, 150, 200, 250, 300, 350, 400],
        n_trials=2
    )
end
