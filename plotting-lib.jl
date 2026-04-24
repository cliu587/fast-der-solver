using Plots
gr()

const NUM_X_TICKS = 5
const NUM_Y_TICKS = 5
const X_TICK_BLOCK_SIZE = 40
const Y_TICK_BLOCK_SIZE = 4
const MIN_X_TICK_UPPER = 40
const MIN_Y_TICK_UPPER = 12
const X_AXIS_PADDING = 1.08
const Y_AXIS_PADDING = 1.12
const MARKER_ALPHA = 0.65

function raw_results(results)
    sizes = Float64[]
    times = Float64[]
    for n in sort(collect(keys(results)))
        append!(sizes, fill(Float64(n), length(results[n])))
        append!(times, results[n])
    end
    return sizes, times
end

function multiple_of_ten_ticks(upper; count=NUM_X_TICKS)
    tick_upper = max(MIN_X_TICK_UPPER, X_TICK_BLOCK_SIZE * ceil(Int, upper / X_TICK_BLOCK_SIZE))
    tick_step = tick_upper ÷ (count - 1)
    return collect(Float64, 0:tick_step:tick_upper)
end

function whole_number_ticks(upper; count=NUM_Y_TICKS)
    tick_upper = max(MIN_Y_TICK_UPPER, Y_TICK_BLOCK_SIZE * ceil(Int, upper / Y_TICK_BLOCK_SIZE))
    tick_step = tick_upper ÷ (count - 1)
    return collect(Float64, 0:tick_step:tick_upper)
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

function plot_benchmark_results(
    slow_results,
    fast_results,
    output_path;
    title,
    slow_label="baseline",
    medium_results=nothing,
    medium_label="OpenDleto",
    fast_label="solve-and-lift",
    x_label="cube side length (n)",
    y_label="time (seconds)"
)
    if isempty(slow_results) && isnothing(medium_results) && isempty(fast_results)
        error("plot_benchmark_results requires at least one non-empty results dictionary.")
    end

    slow_sizes, slow_times = raw_results(slow_results)
    medium_sizes, medium_times = isnothing(medium_results) ? (Float64[], Float64[]) : raw_results(medium_results)
    fast_sizes, fast_times = raw_results(fast_results)
    all_sizes = vcat(slow_sizes, medium_sizes, fast_sizes)
    all_times = vcat(slow_times, medium_times, fast_times)
    x_lower = 0.0
    x_upper = X_TICK_BLOCK_SIZE * ceil(Int, (X_AXIS_PADDING * maximum(all_sizes)) / X_TICK_BLOCK_SIZE)
    x_ticks = multiple_of_ten_ticks(x_upper)
    y_upper = max(Float64(MIN_Y_TICK_UPPER), Y_TICK_BLOCK_SIZE * ceil(Int, (Y_AXIS_PADDING * maximum(all_times)) / Y_TICK_BLOCK_SIZE))
    y_lower = -0.03 * y_upper
    y_ticks = whole_number_ticks(y_upper)

    plot_figure = plot(
        title=title,
        xlabel=x_label,
        ylabel=y_label,
        legend=:bottomright,
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
        tick_direction=:out,
        left_margin=18Plots.mm,
        bottom_margin=16Plots.mm,
        right_margin=12Plots.mm,
        top_margin=10Plots.mm
    )

    scatter!(plot_figure, slow_sizes, slow_times; label=slow_label, marker=:circle, markersize=8, color=:red, alpha=MARKER_ALPHA)
    if !isnothing(medium_results)
        scatter!(plot_figure, medium_sizes, medium_times; label=medium_label, marker=:diamond, markersize=8, color=:orange, alpha=MARKER_ALPHA)
    end
    scatter!(plot_figure, fast_sizes, fast_times; label=fast_label, marker=:square, markersize=8, color=:green, alpha=MARKER_ALPHA)

    savefig(plot_figure, output_path)
    return output_path
end
