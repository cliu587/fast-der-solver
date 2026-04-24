#! /bin/zsh
julia --project=. quick-der-bench.jl long
julia --project=. quicksylver-bench.jl long
julia --project=. quicksylver-vs-dleto-bench.jl long
