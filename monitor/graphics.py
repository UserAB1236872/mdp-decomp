"""
Graphical Representation of Data
"""
from bokeh import plotting
from bokeh.colors import RGB
from bokeh.palettes import Category10
from bokeh.models import Legend, LegendItem, HoverTool
from bokeh.layouts import row


def plot(perf_data, run_mean_data, file_path):
    plotting.output_file(file_path)

    perf_plot = plotting.figure(
        tools="pan,box_zoom,reset,save",
        title="Performance over time",
        y_axis_label='Score',
        x_axis_label='Training Intervals',
        toolbar_location="above",
        sizing_mode="stretch_both",
    )
    perf_plot.add_tools(HoverTool(
        tooltips=[
            ('algorithm', '$name'),
            ('score', '@y'),
            ('episode interval', '$x')
        ],
        mode='vline'
    ))
    perf_plot.below[0].formatter.use_scientific = False
    # draw
    legend_items = []
    for i, name in enumerate(sorted(perf_data.keys())):
        color = Category10[10][i]
        p = perf_plot.line(range(len(perf_data[name])), perf_data[name], line_color=color, name=name)
        legend_items.append(LegendItem(label=name, renderers=[p]))

    legend = Legend(items=legend_items)
    legend.click_policy = 'hide'
    perf_plot.add_layout(legend, 'right')

    run_mean_plot = plotting.figure(
        tools="pan,box_zoom,reset,save",
        title="Running Score Mean over time",
        y_axis_label='Running Score Mean',
        x_axis_label='Training Intervals',
        toolbar_location="above",
        sizing_mode="stretch_both",
    )
    run_mean_plot.add_tools(HoverTool(
        tooltips=[
            ('algorithm', '$name'),
            ('Running Score Mean', '@y'),
            ('Training interval', '$x')
        ],
        mode='vline'
    ))
    run_mean_plot.below[0].formatter.use_scientific = False
    # draw
    rm_legend_items = []
    for i, name in enumerate(sorted(perf_data.keys())):
        color = Category10[10][i]
        p = run_mean_plot.line(range(len(run_mean_data[name])), run_mean_data[name], line_color=color, name=name)
        rm_legend_items.append(LegendItem(label=name, renderers=[p]))

    rm_legend = Legend(items=rm_legend_items)
    rm_legend.click_policy = 'hide'
    run_mean_plot.add_layout(rm_legend, 'right')
    p = row(perf_plot, run_mean_plot)
    plotting.save(p)
