# fast-der-solver

Code demonstrating the quick derivation solving algorithm.

Accompanying poster [here](https://slides.com/chrisliu/gradshow-2025/) 

## Running instructions

Install Julia through `juliaup`, then instantiate this project from the repository root:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The benchmark comparison with OpenDleto uses the GitHub version of that package:

```bash
julia --project=. -e 'using Pkg; Pkg.add(url="https://github.com/thetensor-space/OpenDleto")'
```

To run the benchmark suite used for the dissertation figures:

```bash
./run_bench.sh
```

For quicker runs, each individual benchmark file takes in "short" as the first positional parameter, which will give an instance that finishes quickly.
For instance,

```bash
julia --project=. quick-der-bench.jl short
julia --project=. quicksylver-bench.jl short
julia --project=. quicksylver-vs-dleto-bench.jl short
```
