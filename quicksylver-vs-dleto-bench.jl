using Logging

include("./quicksylver-lib.jl")
using LinearAlgebra
using Dleto
using ITensors
include("./plotting-lib.jl")

function timed_single_tensor_run(solve_instance, T)
    TimerOutputs.reset_timer!(sylver_to)
    GC.gc()
    solution_frame = nothing
    elapsed_seconds = @elapsed solution_frame = solve_instance(T)
    @info "time: $elapsed_seconds"
    show(sylver_to)
    println()
    return elapsed_seconds, solution_frame
end

function dleto_single_tensor_adjoint(T; tol=1e-6)
    frames = [Index(size(T, i), "a_$i") for i in 1:ndims(T)]
    T_tensor = ITensor(T, frames...)
    ops = Dleto.IndTransverseOps(frames, Dleto.UniversalOp())
    chisel = Dleto.AdjointChisel(ndims(T), 1, 2)
    return Dleto.derTrOpsReduced(ops, chisel, T_tensor; tol=tol, nd=size(T, 1))
end

function K_M_field_tensor(n)
    M = zeros(Float64, n, n)
    M[1, n] = 1
    for i in 1:n - 1
        M[i + 1, i] = 1
    end

    coefficients = rand(vcat(-10:-1, 1:10), n)
    return cat([coefficients[k] * M^(k - 1) for k in 1:n]...; dims=3)
end

function warm_up_quicksylver_vs_dleto_benchmark()
    with_logger(NullLogger()) do
        T = K_M_field_tensor(8)
        solve_dense_sylvester_system(T, T, zeros(size(T)))
        dleto_single_tensor_adjoint(T)
        sylvester_solver(T, T, zeros(size(T)))
    end
end

function benchmark_implementation(label, sizes, n_trials, solve_instance)
    results = Dict{Int, Vector{Float64}}()

    for n in sizes
        @info "$label benchmark for $n"
        elapsed_times = Float64[]

        for trial in 1:n_trials
            T = K_M_field_tensor(n)
            elapsed_seconds, solution = timed_single_tensor_run(solve_instance, T)
            solution_dimension = label == "OpenDleto" ? size(solution[3], 2) : length(solution) - 1

            if solution_dimension != n
                error("Expected $label to return adjoint dimension $n, got $solution_dimension.")
            end

            push!(elapsed_times, elapsed_seconds)
        end

        results[n] = elapsed_times
    end

    return results
end

function bench_quicksylver_vs_dleto(; baseline_sizes, dleto_sizes, quick_sizes, n_trials)
    warm_up_quicksylver_vs_dleto_benchmark()

    baseline_results = benchmark_implementation("baseline", baseline_sizes, n_trials, T -> solve_dense_sylvester_system(T, T, zeros(size(T))))
    dleto_results = benchmark_implementation("OpenDleto", dleto_sizes, n_trials, dleto_single_tensor_adjoint)
    quick_results = benchmark_implementation("solve-and-lift", quick_sizes, n_trials, T -> sylvester_solver(T, T, zeros(size(T))))

    return baseline_results, dleto_results, quick_results
end

function print_single_csv_summary(results; io=stdout)
    println(io, "n,trial,time")
    for n in sort(collect(keys(results)))
        for (trial, elapsed_seconds) in enumerate(results[n])
            println(io, "$(n),$(trial),$(elapsed_seconds)")
        end
    end
end

function print_csv_summary(summary_pairs; io=stdout)
    for (i, (label, results)) in enumerate(summary_pairs)
        i > 1 && println(io)
        println(io, label)
        print_single_csv_summary(results; io=io)
    end
end

function write_csv_summary(path, summary_pairs)
    open(path, "w") do io
        print_csv_summary(summary_pairs; io=io)
    end
end

function quicksylver_vs_dleto_benchmark_sizes(mode)
    if mode == :short
        sizes = [8, 10, 12, 15]
        return (; baseline_sizes=sizes, dleto_sizes=sizes, quick_sizes=sizes)
    end

    if mode == :long
        baseline_sizes = [8, 10, 12, 15, 20]
        dleto_sizes = [8, 10, 12, 15, 20, 25, 30, 35]
        quick_sizes = dleto_sizes
        return (; baseline_sizes, dleto_sizes, quick_sizes)
    end

    error("Unknown benchmark mode $mode. Use :short or :long.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    mode = isempty(ARGS) ? :short : Symbol(ARGS[1])
    sizes = quicksylver_vs_dleto_benchmark_sizes(mode)

    baseline_results, dleto_results, quick_results = bench_quicksylver_vs_dleto(
        baseline_sizes=sizes.baseline_sizes,
        dleto_sizes=sizes.dleto_sizes,
        quick_sizes=sizes.quick_sizes,
        n_trials=5
    )
    summary_pairs = [
        ("baseline", baseline_results),
        ("OpenDleto", dleto_results),
        ("solve-and-lift", quick_results)
    ]
    println()
    print_csv_summary(summary_pairs)
    write_csv_summary("quicksylver-vs-dleto-results.csv", summary_pairs)
    plot_benchmark_results(
        baseline_results,
        quick_results,
        "quicksylver-vs-dleto-results.png";
        title="Comparing OpenDleto and solve-and-lift on a non-regular Sylvester family",
        slow_label="baseline",
        medium_results=dleto_results,
        medium_label="OpenDleto",
        fast_label="solve-and-lift"
    )
end
