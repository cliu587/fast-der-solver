using Plots
using Statistics

gr()

function averaged_results(results)
    sizes = sort(collect(keys(results)))
    times = [mean(results[n]) for n in sizes]
    return sizes, times
end

function log_x_ticks(x_lower, x_upper)
    exponent_min = floor(Int, log10(max(x_lower, 1.0)))
    exponent_max = ceil(Int, log10(x_upper))
    tick_values = Float64[]

    for exponent in exponent_min:exponent_max
        scale = 10.0^exponent
        append!(tick_values, scale .* [1.0, 2.0, 5.0])
    end

    return [value for value in tick_values if x_lower <= value <= x_upper]
end

function log_y_ticks(y_lower, y_upper)
    exponent_min = floor(Int, log10(y_lower))
    exponent_max = ceil(Int, log10(y_upper))
    tick_values = Float64[]

    for exponent in exponent_min:exponent_max
        scale = 10.0^exponent
        append!(tick_values, scale .* [1.0, 2.0, 5.0])
    end

    return [value for value in tick_values if y_lower <= value <= y_upper]
end

# Needed as otherwise labels are in scientific notation.
function plain_tick_labels(tick_values)
    return map(tick_values) do value
        rounded_value = round(value; sigdigits=6)

        if isapprox(rounded_value, round(rounded_value); atol=1e-12, rtol=1e-12)
            return string(round(Int, rounded_value))
        end

        return string(rounded_value)
    end
end

function plot_benchmark_results(slow_results, fast_results, output_path; title)
    isempty(slow_results) && isempty(fast_results) && error("plot_benchmark_results requires at least one non-empty results dictionary.")

    slow_sizes, slow_times = averaged_results(slow_results)
    fast_sizes, fast_times = averaged_results(fast_results)
    all_sizes = vcat(slow_sizes, fast_sizes)
    all_times = vcat(slow_times, fast_times)
    x_lower = 0.95 * minimum(all_sizes)
    x_upper = 1.08 * maximum(all_sizes)
    x_ticks = log_x_ticks(x_lower, x_upper)
    y_lower = 0.9 * minimum(all_times)
    y_upper = 1.12 * maximum(all_times)
    y_ticks = log_y_ticks(y_lower, y_upper)

    plot_figure = plot(
        title=title,
        xlabel="cube side length (n, log scale)",
        ylabel="time (seconds, log scale)",
        legend=:bottomright,
        xscale=:log10,
        yscale=:log10,
        xticks=(x_ticks, plain_tick_labels(x_ticks)),
        yticks=(y_ticks, plain_tick_labels(y_ticks)),
        xlims=(x_lower, x_upper),
        ylims=(y_lower, y_upper),
        xformatter=:plain,
        yformatter=:plain,
        size=(1440, 880),
        titlefontsize=24,
        guidefontsize=24,
        tickfontsize=16,
        legendfontsize=18,
        framestyle=:box,
        grid=true,
        minorgrid=true,
        minorticks=5,
        tick_direction=:out,
        left_margin=18Plots.mm,
        bottom_margin=16Plots.mm,
        right_margin=12Plots.mm,
        top_margin=10Plots.mm
    )

    plot!(plot_figure, slow_sizes, slow_times; label="full linear system", lw=2.5, marker=:circle, markersize=8)
    plot!(plot_figure, fast_sizes, fast_times; label="fast solve-and-lift", lw=2.5, marker=:circle, markersize=8)

    savefig(plot_figure, output_path)
    return output_path
end
