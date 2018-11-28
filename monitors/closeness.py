from bokeh import plotting
from bokeh.colors import RGB
from bokeh.palettes import Category10
from bokeh.models import Legend, LegendItem, HoverTool
from gridworld.q_world import QWorld
import numpy as np

import logging


class SolutionMonitor(object):
    def __init__(self, world, exact_constructor, solver_constructors, sample_step=100, max_steps=500000):
        self.exact = exact_constructor(world)
        self.solvers = {type(solver).__name__: solver for solver in map(
            lambda s: s(QWorld(world), verbose=False), solver_constructors)}
        self.world = world
        self.sample_step = sample_step
        self.max_steps = max_steps

        self.samples = np.arange(
            sample_step, max_steps+sample_step, step=sample_step, dtype=int)

        self.curr_index = 0

        self.max_deviations = {}
        self.avg_deviations = {}
        self.total_max_deviations = {}
        self.avg_max_deviation = {}
        self.typed_max_dev = {}
        for name in self.solvers.keys():
            self.max_deviations[name] = []
            self.avg_deviations[name] = []
            self.total_max_deviations[name] = []
            self.avg_max_deviation[name] = []
            self.typed_max_dev[name] = {}
            for r_type in self.world.rewards.keys():
                self.typed_max_dev[name][r_type] = []

        self.graph = None

        self.colors = {}
        for i, name in enumerate(self.solvers.keys()):
            self.colors[name] = Category10[10][i]

    def compute_deviations(self, solver, solver_name):
        eval_max = -np.inf
        eval_avg = 0.0
        eval_avg_denom = 0.0
        eval_avg_max_norm = 0.0
        typed_maxes = {}

        total_max_deviations = -np.inf

        for action in self.world.actions:
            for r_type in self.world.rewards.keys():
                abs_distance = np.abs(self.exact.values[action][r_type] -
                                      solver.q_vals[action][r_type])
                eval_max = max(eval_max, np.max(abs_distance))
                eval_avg = eval_avg + np.sum(abs_distance)
                eval_avg_denom += self.world.shape[0] * self.world.shape[1]
                eval_avg_max_norm += eval_max

                typed_maxes[r_type] = -np.inf if \
                    r_type not in typed_maxes else typed_maxes[r_type]
                typed_maxes[r_type] = max(
                    typed_maxes[r_type], np.max(abs_distance))

            total_max_deviations = max(total_max_deviations, np.max(
                np.abs(solver.total[action] - self.exact.total[action])))

        for r_type, val in typed_maxes.items():
            self.typed_max_dev[solver_name][r_type].append(val)

        self.max_deviations[solver_name].append(eval_max)
        self.avg_deviations[solver_name].append(eval_avg / eval_avg_denom)
        self.total_max_deviations[solver_name].append(total_max_deviations)
        self.avg_max_deviation[solver_name].append(
            eval_avg_max_norm / (len(self.world.actions) * len(self.world.rewards)))

    def run_step(self):
        if self.curr_index == len(self.samples):
            logging.info("Expanding number of samples beyond initial request")
            self.samples.append(self.samples[-1] + self.sample_step)

        if not self.exact.solved:
            logging.warn("Exact solution not found yet, solving")
            self.exact.solve()

        logging.info("Running batch %d (episodes %d-%d/%d)" % (self.curr_index+1, self.curr_index *
                                                               self.sample_step, self.curr_index*self.sample_step+self.sample_step, self.max_steps))

        for name, solver in self.solvers.items():
            solver.run_fixed_eps(num_eps=self.sample_step)
            self.compute_deviations(solver, name)

        self.curr_index += 1

    def compute(self, graph_file="graph.html", text_file="solved.txt", append=True):
        for _ in self.samples:
            self.run_step()

        self.__plot(outfile=graph_file)
        self.__text(outfile="solved.txt", append=append)

    def __plot(self, outfile="graph.html"):
        plotting.output_file(outfile)

        plot = plotting.figure(
            tools="pan,box_zoom,reset,save",
            title="Deviation from ground truth over time",
            y_axis_label='Absolute Deviation',
            x_axis_label='Training Episodes',
            toolbar_location="above",
            sizing_mode="stretch_both",
        )

        plot.add_tools(HoverTool(
            tooltips=[
                ('metric_and_method', '$name'),
                ('deviation', '@y'),
                ('episode', '$x')
            ],
            # formatters={
            #     'deviation':}
            mode='vline'
        ))
        #pylint: disable=E1136
        plot.below[0].formatter.use_scientific = False

        plots = []
        for name in self.solvers.keys():
            color = self.colors[name]
            p0 = plot.line(
                self.samples, self.max_deviations[name], line_color=color, name="%s Max Deviation" % name)
            p1 = plot.line(
                self.samples, self.avg_max_deviation[name], line_color=color, line_dash="4 4", name="%s Avg Max Devation" % name)
            p2 = plot.circle(
                self.samples, self.avg_deviations[name], fill_color="white", line_color=color, name="%s Avg Deviation" % name)
            p2b = plot.line(
                self.samples, self.avg_deviations[name], line_dash="4 4", line_color=color, name="%s Avg Deviation" % name)
            p3 = plot.circle(
                self.samples, self.total_max_deviations[name], fill_color=color, line_color=color, name="%s Total Max Deviation (total Q-values)" % name)
            p4 = plot.line(
                self.samples, self.total_max_deviations[name], line_color=color, line_dash="2 2", name="%s Total Max Deviation (total Q-values)" % name)

            plots.append(LegendItem(label="Max Deviation for % s" %
                                    name, renderers=[p0]))
            plots.append(LegendItem(label="Avg Deviation for %s" %
                                    name, renderers=[p2, p2b]))
            plots.append(LegendItem(label="Total Max Deviation for %s" %
                                    name, renderers=[p3, p4]))
            plots.append(LegendItem(label="Avg Max Deviation for %s (Avg of Max Devs across Reward Types)" %
                                    name, renderers=[p1]))

            color = RGB(int(color[1:3], base=16), int(
                color[3:5], base=16), int(color[5:], base=16))
            color = color.lighten(0.2)
            for (i, (r_type, max_dev)) in enumerate(self.typed_max_dev[name].items()):
                pr = plot.line(self.samples, max_dev, line_color=color, line_dash="%d %d" % (
                    i, i), name="%s Deviation for %s" % (r_type, name))
                plots.append(LegendItem(label="%s Deviation for %s" %
                                        (r_type, name), renderers=[pr]))

        legend = Legend(items=plots)
        legend.click_policy = "hide"

        plot.add_layout(legend, 'right')
        plotting.save(plot)

    def __text(self, outfile="solved.txt", append=True):
        out_string = "World %s\n\n" % type(self.world).__name__
        printable = self.world.printable()
        for r_type, vals in printable.items():
            if r_type == 'total':
                continue
            out_string += "%s:\n %s\n\n" % (r_type, vals)
        out_string += "total:\n %s\n\n" % printable['total']

        out_string += "Exact solution w/ %s\n" % type(self.exact).__name__
        out_string += "%s\n\n" % self.exact.format_solution()

        for name, solver in self.solvers.items():
            out_string += "Exploratory solution w/ %s\n" % name
            out_string += "%s\n\n" % solver.format_solution()

        rw_bit = 'a+' if append else 'w+'
        with open(outfile, rw_bit) as f:
            f.write(out_string)
