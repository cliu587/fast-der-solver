include("./quick-der-lib.jl") 

function der_with_solution(a_dim,b_dim,c_dim,r_dim,s_dim,t_dim)
  r = rand(-10.0:10, (r_dim,b_dim,c_dim))
  s = rand(-10.0:10, (a_dim,s_dim,c_dim))
  x = rand(-10.0:10, (a_dim,r_dim))
  y = rand(-10.0:10, (s_dim,b_dim))

  # t = rand(-10.0:10, (a,b,t_dim))

  z = rand(-10.0:10, (t_dim,c_dim))
  xr_sy = stack([ -(x*r[:,:,i] + s[:,:,i]* y) for i in 1:c_dim])
  t = outer_action(xr_sy, z)
  return r,s,t
end

# a_dim=b_dim=c_dim=r_dim=s_dim=t_dim=50
# r,s,t = der_with_solution(a_dim,b_dim,c_dim,r_dim,s_dim,t_dim)
# der_solution = solve_dense_der(r,s,t)
# der_solution = __quickder_NO_GUARDRAILS(r,s,t,a_prime_dim=4,b_prime_dim=4,c_prime_dim=4)
# der_solution = derivation_solver(r,s,t)

# x_soln,y_soln,z_soln = der_solution[1]
# ans_correct = __answer_check(r,s,t,x_soln,y_soln,z_soln, faster_randomized_check=false)
# i=5
# tz = outer_action(t, z_soln)
# check_on_slice = x_soln*r[:,:,i] + s[:,:,i] * y_soln - tz[:,:,i]
# show(to)

function bench_suite()
  slow_solver_sizes = [10,15,20]
  fast_solver_sizes = [10,20,30,50,75,100]
  n_trials = 3
  slow_results = Dict()
  fast_results = Dict()
  for size in slow_solver_sizes
    @info "slow solver for $size"
    results = []
    r,s,t = der_with_solution(size, size, size, size, size, size)
    for trial in 1:n_trials
      result = @elapsed solve_dense_der(r,s,t)
      @info "time: $result"
      push!(results, result)
    end
    slow_results[size] = results
  end
  for size in fast_solver_sizes
    @info "quick derivation solver for $size"
    results = []
    r,s,t = der_with_solution(size, size, size, size, size, size)
    for trials in 1:n_trials
      r,s,t = der_with_solution(size, size, size, size, size, size)
      result = @elapsed derivation_solver(r,s,t)
      @info "time: $result"
      push!(results, result)
    end
    fast_results[size] = results
  end
end
bench_suite()