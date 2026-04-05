using Logging

include("./quick-der-bench.jl")
include("./quicksylver-bench.jl")

const DERIVATION_SIZES = [10, 16, 25, 40, 63, 100]
const SYLVESTER_SIZES = [10, 18, 32, 56, 100, 177, 316, 500]
const N_TRIALS = 2

function warm_up()
    R_der, S_der, T_der = der_with_solution(10, 10, 10, 10, 10, 10)
    R_syl, S_syl, T_syl = sylvester_with_solution(10, 10, 10, 10, 10)

    with_logger(NullLogger()) do
        derivation_solver(R_der, S_der, T_der)
        sylvester_solver(R_syl, S_syl, T_syl)
    end
end

function collect_timings(label, sizes, build_instance, solve_instance)
    timings = Vector{Tuple{Int, Vector{Float64}}}()

    for n in sizes
        trial_times = Float64[]
        for trial in 1:N_TRIALS
            instance = build_instance(n)
            elapsed_seconds = @elapsed with_logger(NullLogger()) do
                solve_instance(instance...)
            end
            push!(trial_times, elapsed_seconds)
            println("$label,$n,$trial,$elapsed_seconds")
        end
        push!(timings, (n, trial_times))
    end

    return timings
end

function run_log_log_bench()
    warm_up()
    println("solver,n,trial,seconds")

    derivation_timings = collect_timings(
        "derivation",
        DERIVATION_SIZES,
        n -> der_with_solution(n, n, n, n, n, n),
        derivation_solver
    )
    sylvester_timings = collect_timings(
        "sylvester",
        SYLVESTER_SIZES,
        n -> sylvester_with_solution(n, n, n, n, n),
        sylvester_solver
    )

    return derivation_timings, sylvester_timings
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_log_log_bench()
end
