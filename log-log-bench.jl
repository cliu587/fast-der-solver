using Logging
using LinearAlgebra

include("./quick-der-bench.jl")
include("./quicksylver-bench.jl")

const DERIVATION_SIZES = [60, 66, 72, 79, 87, 95, 104, 114, 121, 130]
const SYLVESTER_SIZES = [300, 325, 350, 375, 405, 440, 475, 515, 555, 600]
const N_TRIALS = 5
# Magic constants to make the O(n^3) and O(n^{4.5}) problems take enough time to see the asymptotics, but not too long that it doesn't finish.
const DER_SCALE = 9
const SYLV_SCALE = 16
# const DER_SCALE = 1.5 
# const SYLV_SCALE = 1.5

function warm_up()
    with_logger(NullLogger()) do
        build_derivation_run(10)()
        build_sylvester_run(10)()
        build_der_reference(10)()
        build_sylv_reference(10)()
    end
end

function build_der_reference(n)
    matrix_dim = ceil(Int, DER_SCALE * n^1.5)
    return () -> inv(rand(matrix_dim, matrix_dim))
end

function build_sylv_reference(n)
    matrix_dim = ceil(Int, SYLV_SCALE * n)
    return () -> inv(rand(matrix_dim, matrix_dim))
end

function build_derivation_run(n)
    R, S, T = der_fixture(n)
    return () -> derivation_solver(R, S, T)
end

function build_sylvester_run(n)
    R, S, T = sylvester_fixture(n)
    return () -> sylvester_solver(R, S, T)
end

function collect_timings(label, sizes, build_run)
    timings = Vector{Tuple{Int, Vector{Float64}}}()

    for n in sizes
        trial_times = Float64[]
        for trial in 1:N_TRIALS
            GC.gc()
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

    derivation_reference_timings = collect_timings(
        "derivation O(n^{4.5}) reference",
        DERIVATION_SIZES,
        build_der_reference
    )
    derivation_timings = collect_timings(
        "derivation",
        DERIVATION_SIZES,
        build_derivation_run
    )

    sylvester_timings = collect_timings(
        "sylvester",
        SYLVESTER_SIZES,
        build_sylvester_run
    )
    sylvester_reference_timings = collect_timings(
        "sylvester O(n^{3}) reference",
        SYLVESTER_SIZES,
        build_sylv_reference
    )

    return derivation_timings, derivation_reference_timings, sylvester_timings, sylvester_reference_timings
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_log_log_bench()
end
