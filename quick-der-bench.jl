include("./quick-der-lib.jl") 

function show_run_timer()
  show(to)
  println()
end

function der_with_solution(a_dim,b_dim,c_dim,r_dim,s_dim,t_dim)
  R = rand(-10.0:10, (r_dim,b_dim,c_dim))
  S = rand(-10.0:10, (a_dim,s_dim,c_dim))
  X = rand(-10.0:10, (a_dim,r_dim))
  Y = rand(-10.0:10, (s_dim,b_dim))

  Z = rand(-10.0:10, (t_dim,c_dim))
  XR_plus_SY = stack([-(X * R[:, :, i] + S[:, :, i] * Y) for i in 1:c_dim])
  T = outer_action(XR_plus_SY, Z)
  return R,S,T
end

function bench()
  slow_solver_sizes = [10,15,20]
  fast_solver_sizes = [10,20,30,50,75,100]
  n_trials = 3
  slow_results = Dict{Int, Vector{Float64}}()
  fast_results = Dict{Int, Vector{Float64}}()
  for size in slow_solver_sizes
    @info "slow solver for $size"
    results = Float64[]
    R,S,T = der_with_solution(size, size, size, size, size, size)
    for _ in 1:n_trials
      TimerOutputs.reset_timer!(to)
      result = @elapsed solve_dense_derivation_system(R,S,T)
      @info "time: $result"
      show_run_timer()
      push!(results, result)
    end
    slow_results[size] = results
  end
  for size in fast_solver_sizes
    @info "quick derivation solver for $size"
    results = Float64[]
    R,S,T = der_with_solution(size, size, size, size, size, size)
    for _ in 1:n_trials
      R,S,T = der_with_solution(size, size, size, size, size, size)
      TimerOutputs.reset_timer!(to)
      result = @elapsed derivation_solver(R,S,T; faster_randomized_check=true)
      @info "time: $result"
      show_run_timer()
      push!(results, result)
    end
    fast_results[size] = results
  end
  return slow_results, fast_results
end
bench()
