"""
Graphical Representation of Data
"""
from bokeh import plotting
from bokeh.colors import RGB
from bokeh.palettes import Category10
from bokeh.models import Legend, LegendItem, HoverTool
from bokeh.layouts import row


def _plot(data, file_path):
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
            ('metric_and_method', '$name'),
            ('score', '@y'),
            ('episode interval', '$x')
        ],
        mode='vline'
    ))
    perf_plot.below[0].formatter.use_scientific = False
    # draw
    for i, name in enumerate(sorted(data.keys())):
        color = Category10[10][i]
        perf_plot.line(data[name]['test'], line_color=color, name=name)

    plotting.save(row(perf_plot))
