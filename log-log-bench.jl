using Logging
using LinearAlgebra

include("./quick-der-bench.jl")
include("./quicksylver-bench.jl")

const DERIVATION_SIZES = [25, 30, 36, 43, 51, 61, 73, 87, 105, 125]
const SYLVESTER_SIZES = [200, 230, 264, 304, 349, 401, 461, 530, 609, 700]
const N_TRIALS = 1
# Magic constant to make the O(n^3) and O(n^{4.5}) problems take enough time
const REFERENCE_SCALE = 1.0 

function warm_up()
    R_der, S_der, T_der = der_with_solution(10, 10, 10, 10, 10, 10)
    R_syl, S_syl, T_syl = sylvester_with_solution(10, 10, 10, 10, 10)

    with_logger(NullLogger()) do
        derivation_solver(R_der, S_der, T_der)
        sylvester_solver(R_syl, S_syl, T_syl)
        build_reference_run(10; exponent=1.0)()
        build_reference_run(10; exponent=1.5)()
    end
end

function build_reference_run(n; exponent)
    matrix_dim = ceil(Int, REFERENCE_SCALE * n^exponent)
    return () -> inv(rand(matrix_dim, matrix_dim))
end

function build_derivation_run(n)
    R, S, T = der_with_solution(n, n, n, n, n, n)
    return () -> derivation_solver(R, S, T)
end

function build_sylvester_run(n)
    R, S, T = sylvester_with_solution(n, n, n, n, n)
    return () -> sylvester_solver(R, S, T)
end

function collect_timings(label, sizes, build_run)
    timings = Vector{Tuple{Int, Vector{Float64}}}()

    for n in sizes
        trial_times = Float64[]
        for trial in 1:N_TRIALS
            elapsed_seconds = @elapsed with_logger(NullLogger()) do
                build_run(n)()
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
        build_derivation_run
    )
    derivation_reference_timings = collect_timings(
        "derivation O(n^{4.5}) reference",
        DERIVATION_SIZES,
        n -> build_reference_run(n; exponent=1.5)
    )

    sylvester_timings = collect_timings(
        "sylvester",
        SYLVESTER_SIZES,
        build_sylvester_run
    )
    sylvester_reference_timings = collect_timings(
        "sylvester O(n^{3}) reference",
        SYLVESTER_SIZES,
        n -> build_reference_run(n; exponent=1.0)
    )

    return derivation_timings, derivation_reference_timings, sylvester_timings, sylvester_reference_timings
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_log_log_bench()
end
