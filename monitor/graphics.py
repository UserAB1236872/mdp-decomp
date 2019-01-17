"""
Graphical Representation of Data
"""
from bokeh import plotting
from bokeh.colors import RGB
from bokeh.palettes import Category10
from bokeh.models import Legend, LegendItem, HoverTool
from bokeh.layouts import row


# Todo: refactor
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


def msx_plot(data, solvers, reward_types, actions, optimal_solver):
    import dash
    import dash_core_components as dcc
    import dash_html_components as html
    from dash.dependencies import Input, Output, State



    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css','#game_area>div : {margin:10px 0px}']
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
    graph_style = {'width': '20%', 'display': 'inline-block'}
    game_area_style= {'width':'20%','float':'left','margin':'25% auto'}
    graph_area_style= {'width':'80%','float':'right'}

    action_options = []
    for i, a in enumerate(actions):
        action_options += [{'label': a + ',' + a_dash, 'value': str(i) + '_' + str(j)} for j, a_dash in
                           enumerate(actions) if
                           a != a_dash]
    action_pair_selector = dcc.Dropdown(
        id='action_pair_selector',
        options=action_options,
        value=action_options[0]['value']
    )

    state_slider = dcc.Slider(
        id='state_selector',
        min=0,
        max=len(data)-1,
        step=1,
        value=0,
        marks={i: str(i).format(i) for i in range(len(data))},
    )

    q_values_graphs, msx_graphs = [], []
    for solver in solvers:
        q_graph = dcc.Graph(
            id='q-value-' + solver,
            style=graph_style,
            className=solver
        )
        m_graph = dcc.Graph(
            id='msx-' + solver,
            style=graph_style,
            className=solver
        )
        q_values_graphs.append(q_graph)
        msx_graphs.append(m_graph)

    # create the layout
    action_wrap = html.Div(className='action_area',children=["Action Pair:",action_pair_selector])
    state_slider_wrap = html.Div(className='trajectory',children=["Trajectory:",state_slider])
    game_area = html.Div(id='game_area',style=game_area_style,
                         children=[html.Div(id='state', children=''),
                                   action_wrap,
                                   state_slider_wrap])
    graph_area = html.Div(id='graph_area',style=graph_area_style,
                          children=q_values_graphs + msx_graphs)
    children = [game_area, graph_area]
    app.layout = html.Div(children=children)

    # create callbacks
    @app.callback(
        Output(component_id='state', component_property='children'),
        [Input(component_id='state_selector', component_property='value')])
    def update_state(i):
        return data[i]['state']

    for solver in solvers:

        @app.callback(
            Output(component_id='q-value-' + solver, component_property='figure'),
            [Input(component_id='state_selector', component_property='value')],
            [State(component_id='q-value-' + solver, component_property='className')])
        def update_q_graphs(i, solver):
            rt_data = []
            for rt_i, rt in enumerate(reward_types):
                rt_data.append({'x': actions,
                                'y': [data[i]['solvers'][solver]['q_values'][rt_i][a] for a in
                                      range(len(actions))],
                                'type': 'bar',
                                'name': rt})
            figure = {
                'data': rt_data,
                'layout': {
                    'title': solver
                }
            }
            return figure

        @app.callback(
            Output(component_id='msx-' + solver, component_property='figure'),
            [Input(component_id='state_selector', component_property='value'),
             Input(component_id='action_pair_selector', component_property='value')],
            [State(component_id='msx-' + solver, component_property='className')])
        def update_msx_graphs(state, action_pair, solver):
            first_action, sec_action = [int(a) for a in action_pair.split('_')]
            rt_data = []
            for rt_i, rt in enumerate(reward_types):
                rt_data.append({'x': [''],
                                'y': [round(data[state]['solvers'][solver]['msx'][first_action][sec_action][rt], 2)],
                                'type': 'bar',
                                'name': rt})
            figure = {
                'data': rt_data,
                'layout': {
                    'title': solver
                }
            }
            return figure
    app.run_server(debug=True)
