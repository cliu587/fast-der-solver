using LinearAlgebra
using Statistics
include("./quick-der-lib.jl") 

function show_derivation_run_timer()
  show(to)
  println()
end

function timed_derivation_run(solve_instance, R, S, T)
  TimerOutputs.reset_timer!(to)
  elapsed_seconds = @elapsed solve_instance(R, S, T)
  @info "time: $elapsed_seconds"
  show_derivation_run_timer()
  return elapsed_seconds
end

function der_with_solution(a_dim,b_dim,c_dim,r_dim,s_dim,t_dim)
  R = rand(-10.0:10, (r_dim,b_dim,c_dim))
  S = rand(-10.0:10, (a_dim,s_dim,c_dim))
  X = rand(-10.0:10, (a_dim,r_dim))
  Y = rand(-10.0:10, (s_dim,b_dim))

  Z = rand(-10.0:10, (t_dim,c_dim))
  while rank(Z) < min(t_dim, c_dim)
    Z = rand(-10.0:10, (t_dim,c_dim))
  end

  XR_plus_SY = stack([X * R[:, :, i] + S[:, :, i] * Y for i in 1:c_dim])
  T = reshape(reshape(XR_plus_SY, (a_dim * b_dim, c_dim)) / Z, (a_dim, b_dim, t_dim))
  return R,S,T
end

function bench_derivation(;
  slow_solver_sizes,
  fast_solver_sizes,
  n_trials)

  slow_results = Dict{Int, Vector{Float64}}()
  fast_results = Dict{Int, Vector{Float64}}()

  for size in slow_solver_sizes
    @info "slow solver for $size"
    R, S, T = der_with_solution(size, size, size, size, size, size)
    slow_results[size] = [timed_derivation_run(solve_dense_derivation_system, R, S, T) for _ in 1:n_trials]
  end

  for size in fast_solver_sizes
    @info "quick derivation solver for $size"
    elapsed_seconds = Float64[]
    for _ in 1:n_trials
      R, S, T = der_with_solution(size, size, size, size, size, size)
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
    slow_solver_sizes=[5,10,15,20,23,25],
    fast_solver_sizes=[10,20,30,50,75,100],
    # slow_solver_sizes=[5],
    # fast_solver_sizes=[10],
    n_trials=1
  )
  println("\nslow solver performance")
  print_csv_summary(slow_results)
  println("\nfast solver performance")
  print_csv_summary(fast_results)
end
