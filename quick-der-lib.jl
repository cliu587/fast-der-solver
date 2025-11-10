using Logging
using LinearAlgebra
using TimerOutputs
to = TimerOutput()

# Solves the dense system.
function solve_dense_der(r,s,t; use_svd_for_nullspace=false)
    # unpack sizes and sanity-check
    r_dim, b, c    = size(r)
    a,     s_dim, _ = size(s)
    _,    _,    t_dim = size(t)

    # rows first stride by c then b.
    r_flat = reshape(permutedims(r, (3, 2, 1)), (c*b, r_dim))

    I_a  = Matrix{Float64}(I, a, a)
    # rows first stide by c then b then a. 
    # columns first stride by r_dim then a.
    r_mat = kron(I_a, r_flat)

    I_b      = Matrix{Float64}(I, b, b)
    s_slices = [Transpose(s[i, :, :]) for i in 1:a] # Rows stride by c then a
    # Rows stride by c then b then a
    # Columns stride by s_dim then b
    s_mat = vcat([kron(I_b, m) for m in s_slices]...)

    t_mat_reshape = reshape(permutedims(t,(2,1,3)), (b*a, t_dim))
    I_c = Matrix(I, c,c)

    # Rows stride by c then b then a
    # Columns stride by c then t_dim
    t_mat = kron(t_mat_reshape, I_c)

    # A slower method.
    # t_mat_v2 = vcat([kron(Transpose(t[i,j,:]), I_c) for i in 1:a for j in 1:b]...)

    # 4) Stack the three pieces into one big M
    M = hcat(r_mat, s_mat, -t_mat)
    rows, cols = size(M)
    @info "Dense solver size: $(size(M))"
    if cols > rows
      @warn "Dense derivation solver system has more cols than rows, most likely underconstrained!"
    end
    ns = 1
    if use_svd_for_nullspace # This method is better when system is overdetermined
      @info "Using svd to approximate nullspace!"
      svd_out = LinearAlgebra.svd(M)
      ns_vecs = [(sigma,v) for (sigma,v) in zip(svd_out.S, eachcol(svd_out.V)) if sigma < 1e-6]
      ns = hcat([v for (sigma, v) in ns_vecs]...)
    else
      ns = nullspace(M)
    end
    return ns
end


function outer_action(t, z)
  a,b,c = size(t)
  z_cols = size(z)[2]
  reshape(reshape(t, (a*b, c)) * z, (a,b,z_cols))
end

function __answer_check(r,s,t,x,y,z;
  # If true, check only on 1 slice that xr_i + s_iy = t_i.
  # Otherwise, check for all i in 1..c.
  faster_randomized_check=true)

  r_dim,b,c = size(r)
  a,s_dim,_ = size(s)
  _,_,t_dim = size(s)

  if faster_randomized_check
    random_c = rand(1:c);
    r_slice  = r[:,:,random_c]
    s_slice  = s[:,:,random_c]
    t_slice  = outer_action(t, z)[:,:,random_c]
    # @info "anwer check out: $(x*r_slice + s_slice * y - t_slice)"
    return isapprox(x*r_slice + s_slice * y - t_slice, zeros(size(t_slice)), rtol=1e-6, atol=1e-6)
  else
    tz = outer_action(t,z)
    return all([
      isapprox(x*r[:,:,i] + s[:,:,i] * y - tz[:,:,i], zeros(size(r[:,:,i])), rtol=1e-6, atol=1e-6) for i in 1:c])
  end
end

function __quickder_NO_GUARDRAILS(r,s,t, 
  ;a_prime_dim, 
  b_prime_dim,
  c_prime_dim,
  num_nullspace_dims_to_sample = 10)

  r_dim,b,c = size(r)
  a,s_dim,_ = size(s)
  _,_,t_dim = size(t)
  
  if a_prime_dim == a || b_prime_dim == b || c_prime_dim == c
    @warn ("Running derivation solver with a_prime = a, b_prime = b, or c_prime = c! Probably not getting a performance improvement as a result. (a_prime: $a_prime_dim a: $a) (b_prime: $b_prime_dim, b: $b) (c_prime: $c_prime_dim, c: $c)")
  end

  A_prime = 1:a_prime_dim; U = a_prime_dim+1:a; U_dim = a-a_prime_dim;
  B_prime = 1:b_prime_dim; V = b_prime_dim+1:b; V_dim = b-b_prime_dim;
  C_prime = 1:c_prime_dim; W = c_prime_dim+1:c; W_dim = c-c_prime_dim;

  # DATA MANAGEMENT: Creating a smaller dense system
  r_prime = r[1:r_dim, B_prime, C_prime]
  s_prime = s[A_prime, 1:s_dim, C_prime]
  t_prime = t[A_prime, B_prime, 1:t_dim]

  ns = 1
  try
    # COST: Solve linear system of a'r + b's + c't variables.
    # Number of equations is a'b'c'
    @timeit to "dense system" begin
      ns = solve_dense_der(r_prime, s_prime, t_prime)
      # @info "ns size: $(size(ns)), r_prime: $(size(r_prime)), s_prime: $(size(s_prime)), t_prime: $(size(t_prime))"
    end
  catch e
    throw("No solution exists!");
  end

  # Step 3: Find inverses and lift dense system solutions.
  backsub_setup = begin_timed_section!(to, "backsub_setup")

  # Needed for x_U backsub. Stack to be b'c' x r
  r_B_prime_C_prime = Transpose(
    reshape(
      r[1:r_dim, B_prime, C_prime], (r_dim, b_prime_dim * c_prime_dim)
    )
  )

  # COST: Left inverse of the (b'c' by r) matrix. Output: (r by b'c')
  r_B_prime_C_prime_inv = pinv(r_B_prime_C_prime);

  t_U_B_prime = t[U, B_prime, 1:t_dim]
  s_U_C_prime = s[U, 1:s_dim, C_prime]

  # Needed for y_V backsub. Stack to be a'c' by s.
  s_A_prime_C_prime = reshape(
    permutedims(s[A_prime, 1:s_dim, C_prime], (1,3,2)), (a_prime_dim*c_prime_dim, s_dim))

  # COST: Left inverse of the (a'c' by s) matrix. Output: (s by a'c')
  s_A_prime_C_prime_inv = pinv(s_A_prime_C_prime);

  t_A_prime_V = t[A_prime, V, 1:t_dim]
  r_V_C_prime = r[1:r_dim, V, C_prime];

  # Needed for z_W backsub. Stack a matrix to be a'b' by t.
  t_A_prime_B_prime = reshape(t[A_prime, B_prime, :], (a_prime_dim * b_prime_dim, t_dim))
  t_A_prime_B_prime_inv = pinv(t_A_prime_B_prime)

  r_B_prime_W = r[1:r_dim, B_prime, W]
  s_A_prime_W = s[A_prime, 1:s_dim, W]

  end_timed_section!(to, backsub_setup)

  # START: Backsub for a given x_prime, y_prime, z_prime triple.
  function backsub(x_prime, y_prime, z_prime)
    # For x_U backsub. Each output is (b'c') by U matrix
    sy_U_B_prime_stacked = Transpose(
      # After hcat, U by b_prime * c_prime
      hcat(
        # Each is U by b_prime
        [s_U_C_prime[:,:,i] * y_prime for i in 1:c_prime_dim]...)
    ); 

    # This matrix is (U * b') by c'
    tz_U_B_prime_mat = reshape(t_U_B_prime, (U_dim*b_prime_dim, t_dim)) * z_prime
    tz_U_B_prime_stacked = reshape(
      permutedims(
        reshape(tz_U_B_prime_mat, (U_dim, b_prime_dim, c_prime_dim)), (2,3,1)), (b_prime_dim * c_prime_dim, U_dim)
      )
    
    # For y_V backsub. Each matrix should be a'c' by V
    rx_A_prime_V_stacked = 
      vcat([
        # Each matrix is a' by V
        x_prime * r_V_C_prime[:,:,i] for i in 1:c_prime_dim]...)

    # Matrix is a' * V by c'
    tz_A_prime_V_mat = reshape(t_A_prime_V, (a_prime_dim*V_dim, t_dim)) * z_prime
    # Reshape it to a'c' by V
    tz_A_prime_V_stacked = reshape(
      permutedims(reshape(
        tz_A_prime_V_mat, (a_prime_dim, V_dim, c_prime_dim)
      ), (1,3,2)), 
      (a_prime_dim * c_prime_dim, V_dim)
    )

    # For z_W backsub. Each matrix should be a'b' by W
    rx_A_prime_B_prime_W_stacked = 
      vcat([x_prime * r_B_prime_W[:,i,:] for i in 1:b_prime_dim]...)

    
    s_A_prime_W_slices = [
      # Each matrix is b' by W, just need to stack and permute to get a'b' by W
      Transpose(y_prime) * s_A_prime_W[i,:,:] for i in 1:a_prime_dim]

    
    sy_A_prime_B_prime_W_stacked = reshape(
        permutedims(
        # Output is b'a' by W, permute to a'b' by W
        reshape(vcat(s_A_prime_W_slices...), b_prime_dim, a_prime_dim, W_dim),
        (2,1,3)),
      (a_prime_dim * b_prime_dim, W_dim)
    )

    x_U = r_B_prime_C_prime_inv * (tz_U_B_prime_stacked - sy_U_B_prime_stacked);
    x = vcat(x_prime, Transpose(x_U));

    y_V = s_A_prime_C_prime_inv * (tz_A_prime_V_stacked - rx_A_prime_V_stacked);
    y = hcat(y_prime, y_V);

    z_W = t_A_prime_B_prime_inv * (rx_A_prime_B_prime_W_stacked + sy_A_prime_B_prime_W_stacked)
    z = hcat(z_prime, z_W)

    return x,y,z
  end

  backsub_section = begin_timed_section!(to, "backsub")

  # We pay a cost for every nullspace solution we compute.
  nullspace_basis = [];
  if num_nullspace_dims_to_sample > 0
    dims = min(size(ns)[2], num_nullspace_dims_to_sample);
    @info "Solving $dims ns! (size(ns): $(size(ns))"
    for i in 1:dims
      xyz_prime_basis = ns[:,i]
      x_prime = Transpose(reshape(xyz_prime_basis[1:a_prime_dim * r_dim], (r_dim, a_prime_dim)));

      y_prime = reshape(xyz_prime_basis[a_prime_dim * r_dim + 1:a_prime_dim*r_dim+b_prime_dim*s_dim], (s_dim, b_prime_dim));

      z_prime = Transpose(reshape(xyz_prime_basis[a_prime_dim*r_dim+b_prime_dim*s_dim + 1:end], (c_prime_dim, t_dim)));

      # @info "backsubbing with x_prime: $x_prime, y_prime: $y_prime, z_prime: $z_prime"
      x_bs,y_bs,z_bs = backsub(x_prime, y_prime, z_prime);
      @info "sizes: x_bs: $(size(x_bs)), y_bs: $(size(y_bs)), z_bs: $(size(z_bs))"
      push!(nullspace_basis, [x_bs,y_bs,z_bs]);
    end
  end
  end_timed_section!(to, backsub_section)
  @info "nullspace basis size: $(size(nullspace_basis))"
  return nullspace_basis;
end

function derivation_solver(r,s,t;
  # Controls how many solutions from the nullspace we produce.
  num_nullspace_dims_to_sample=5, 
  # Customizations that the user can do.
  a_prime_b_prime_c_prime_overrides=-1, 
  eqn_to_variables_ratio=1.1,
  faster_randomized_check=true)

  preprocess_section = begin_timed_section!(to, "preprocess section")
  r_dim,b,c = size(r)
  a,s_dim,c = size(s)
  _,_,t_dim = size(t)

  # Start with numbers that make sense in the cubic case.
  a_prime_dim = Int(ceil(min(a, a^(1/3))));
  b_prime_dim = Int(ceil(min(b, b^(1/3))));
  c_prime_dim = Int(ceil(min(c, c^(1/3))));
  @info ("a_prime dim start: $a_prime_dim b_prime dim start: $b_prime_dim c_prime dim start: $c_prime_dim");

  # Modify the starting point guess by ensuring a_prime, b_prime, c_prime large enough to satisfy the equation to variable ratio.
  while ceil((a_prime_dim * b_prime_dim * c_prime_dim)/eqn_to_variables_ratio) < a_prime_dim * r_dim + b_prime_dim * s_dim + c_prime_dim * t_dim
    if a_prime_dim < a
      a_prime_dim = a_prime_dim + 1
    end
    if b_prime_dim < b
      b_prime_dim = b_prime_dim + 1
    end
    if c_prime_dim < c
      c_prime_dim = c_prime_dim + 1
    end
  end

  # Unless user hard codes dimension overrides, we find
  # the values needed to invert slices from 1 to a_prime, and if not, making each a multiplicative factor of the number we previously tried, so we don't try too much.
  if a_prime_b_prime_c_prime_overrides == -1
    @info("Finding a_prime, b_prime, and c_prime to guarantee invertibility of partial slices...");
    # This is the current 1..b_prime_dim slices, see if it is invertible, and if not, keep on raising b_prime_dim.
    cur_r = reshape(
      permutedims(r[1:r_dim,1:b_prime_dim,1:c_prime_dim], (2,3,1)),
      (b_prime_dim*c_prime_dim, r_dim))
    cur_s = reshape(
      permutedims(s[1:a_prime_dim, 1:s_dim, 1:c_prime_dim], (1,3,2)),
      (a_prime_dim*c_prime_dim, s_dim))
    cur_t = reshape(
      t[1:a_prime_dim, 1:b_prime_dim, 1:t_dim],
      (a_prime_dim*b_prime_dim, t_dim))

    while (rank(cur_r) < size(cur_r, 2) || 
      rank(cur_s) < size(cur_s, 2) || rank(cur_t) < size(cur_t, 2)) &&
      !(c_prime_dim == c && b_prime_dim == b && a_prime_dim == a)
      @info "rank(cur_r): $(rank(cur_r)), cols: $(size(cur_r, 2))"
      @info "rank(cur_s): $(rank(cur_s)), cols: $(size(cur_s, 2))"
      @info "rank(cur_t): $(rank(cur_t)), cols: $(size(cur_t, 2))"
      a_prime_dim_old = a_prime_dim;
      a_prime_dim = Int(min(ceil(a_prime_dim * 1.25), a));
      b_prime_dim_old = b_prime_dim
      b_prime_dim = Int(min(ceil(b_prime_dim * 1.25), b));
      c_prime_dim_old = c_prime_dim
      c_prime_dim = Int(min(ceil(c_prime_dim * 1.25), c));
      @info "a_prime_dim old: $a_prime_dim_old, b_prime_dim_old: $b_prime_dim_old, b_prime_dim: $b_prime_dim, c_prime_dim_old: $c_prime_dim_old, c_prime_dim: $c_prime_dim"
      cur_r = reshape(
        permutedims(r[1:r_dim,1:b_prime_dim,1:c_prime_dim], (2,3,1)),
        (b_prime_dim*c_prime_dim, r_dim))
      cur_s = reshape(
        permutedims(s[1:a_prime_dim, 1:s_dim, 1:c_prime_dim], (1,3,2)),
        (a_prime_dim*c_prime_dim, s_dim))
      cur_t = reshape(
        t[1:a_prime_dim, 1:b_prime_dim, 1:t_dim],
        (a_prime_dim*b_prime_dim, t_dim))
    end;
  else
    a_prime_dim, b_prime_dim, c_prime_dim  = a_prime_b_prime_c_prime_overrides
  end;

  @info "a_prime_dim end: $a_prime_dim, b_prime_dim end: $b_prime_dim, c_prime_dim end: $c_prime_dim"
  end_timed_section!(to, preprocess_section)

  @timeit to "main derivation solver routine" begin
    solutions = __quickder_NO_GUARDRAILS(r, s, t, a_prime_dim=a_prime_dim, b_prime_dim=b_prime_dim,c_prime_dim=c_prime_dim,num_nullspace_dims_to_sample=num_nullspace_dims_to_sample);
  end

  x,y,z = solutions[end] # Check last solution

  if !__answer_check(r,s,t,x,y,z,faster_randomized_check=faster_randomized_check)
    throw("Derivation solver did not find a correct answer! Try setting `eqn_to_variables_ratio` to be a bigger number, or supply a fully non-degenerate tensor.")
  end

  return solutions
end;