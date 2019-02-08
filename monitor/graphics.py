"""
Graphical Representation of Data
"""
import os
from bokeh import plotting
from bokeh.colors import RGB
from bokeh.palettes import Category10
from bokeh.models import Legend, LegendItem, HoverTool
from bokeh.layouts import row
import colorlover as cl
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from plotly import tools
import pickle


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


def x_layout(app, data, reward_types, actions, prefix):
    # graph_style = {'width': '20%', 'display': 'inline-block'}
    graph_style = {}
    solvers = sorted(data.keys())
    game_area_style = {'width': '300px', 'float': 'left', 'position': 'fixed'}
    graph_area_style = {'width': str(500 * (len(solvers) + 1)) + 'px', 'float': 'right', 'position': 'absolute'}

    reward_colors = {rt: cl.scales['7']['qual']['Dark2'][i] for i, rt in enumerate(reward_types)}

    solver_dropdown_options = []
    for solver in solvers:
        for i, ep in enumerate(data[solver]):
            option = {'label': str(i) + '- ' + solver + '- Score: ' + str(ep['score']),
                      'value': solver + '-' + str(i)}
            solver_dropdown_options.append(option)
    solver_dropdown = dcc.Dropdown(
        id=prefix + 'solver_dropdown',
        options=solver_dropdown_options,
        value=solver_dropdown_options[0]['value']
    )
    manager = {'n_clicks': {}, 'curr_solver_episode': solver_dropdown_options[0]['value']}
    action_options = []
    for i, a in enumerate(actions):
        action_options += [{'label': a + ',' + a_dash, 'value': str(i) + '_' + str(j)} for j, a_dash in
                           enumerate(actions) if a != a_dash]

    state_selector = html.Div(className='state_selector',
                              id=prefix + 'state_selector',
                              children=[
                                  html.Button('Prev State', id=prefix + 'prev_state'),
                                  html.Span('0', id=prefix + 'curr_state'),
                                  html.Span('/'),
                                  html.Span(str(len(data) - 1), id=prefix + 'max_state'),
                                  html.Button('Next State', id=prefix + 'next_state')
                              ])
    manager['n_clicks'][prefix + 'prev_state'] = 0
    manager['n_clicks'][prefix + 'next_state'] = 0

    q_values_graphs, decomp_q_values_graphs, rdx_graphs, msx_graphs, graph_action_pairs, greedy_actions = [], [], [], [], [], []
    for solver in solvers:
        q_graph = dcc.Graph(
            id=prefix + 'q-value' + solver,
            className=solver,
            style=graph_style
        )
        decomp_q_graph = dcc.Graph(
            id=prefix + 'decomp-q-value' + solver,
            className=solver,
            style=graph_style
        )
        rdx_graph = dcc.Graph(
            id=prefix + 'rdx-' + solver,
            className=solver,
            style=graph_style
        )
        msx_graph = dcc.Graph(
            id=prefix + 'msx-' + solver,
            className=solver,
            style=graph_style
        )
        action_pair_selector = dcc.Dropdown(
            id=prefix + 'action_pair_selector-' + solver,
            options=action_options,
            value=action_options[0]['value'],
            className='action_pair'
        )
        q_values_graphs.append(html.Div(className='graph_box', children=[q_graph]))
        decomp_q_values_graphs.append(html.Div(className='graph_box', children=[decomp_q_graph]))
        graph_action_pairs.append(html.Div(className='graph_box', children=[action_pair_selector]))
        msx_graphs.append(html.Div(className='graph_box', children=[msx_graph]))
        rdx_graphs.append(html.Div(className='graph_box', children=[rdx_graph]))
        greedy_actions.append(html.Div(className='graph_box', children=[html.Div(children='', className='center',
                                                                                 id=prefix + 'greedy_action-' + solver)]))

    q_values_graphs.append(html.Div(className='clear'))
    decomp_q_values_graphs.append(html.Div(className='clear'))
    msx_graphs.append(html.Div(className='clear'))
    rdx_graphs.append(html.Div(className='clear'))
    graph_action_pairs.append(html.Div(className='clear'))
    greedy_actions.append(html.Div(className='clear'))

    q_value_wrapper = html.Section(children=[html.Div(children='Q-values', className='title'),
                                             html.Div(className='graph_group', children=q_values_graphs),
                                             html.Div(className='clear')])
    decomp_q_value_wrapper = html.Section(children=[html.Div(children='Decomposed Q-values', className='title'),
                                                    html.Div(className='graph_group', children=decomp_q_values_graphs),
                                                    html.Div(className='clear')])
    msx_wrapper = html.Section(children=[html.Div(children='MSX', className='title'),
                                         html.Div(className='graph_group', children=msx_graphs),
                                         html.Div(className='clear')])
    rdx_wrapper = html.Section(children=[html.Div(children='RDX', className='title'),
                                         html.Div(className='graph_group', children=rdx_graphs),
                                         html.Div(className='clear')])
    action_pair_wrapper = html.Section(
        children=[html.Div(children='Action Pair Selector for RDX/MSX', className='title'),
                  html.Div(className='graph_group', children=graph_action_pairs),
                  html.Div(className='clear')])
    greedy_action_wrapper = html.Section(children=[html.Div(children='Greedy Action:', className='title'),
                                                   html.Div(className='graph_group', children=greedy_actions),
                                                   html.Div(className='clear')])

    solver_dropdown_wrapper = html.Div(children=[html.Div(children='Trajectory Selector:'), solver_dropdown],
                                       className='solver_selector')
    # create the layout
    # action_wrap = html.Div(className='action_area', children=["Action Pair:", action_pair_selector])
    state_slider_wrap = html.Div(className='trajectory', children=["States:", state_selector])

    state_info_wrap = html.Div(children=[html.Div('State Info:'),
                                         html.Div(id=prefix + 'state', children='', className='state'),
                                         html.Div('Reward Info:'),
                                         html.Div(id=prefix + 'reward-info', children='', className='reward_info'),
                                         html.Span('Terminal:'),
                                         html.Span(id=prefix + 'terminal', children='', className='terminal')],
                               className='state_info_box'
                               )

    game_area = html.Div(className='game_area', style=game_area_style,
                         children=[html.Div(children=[solver_dropdown_wrapper,
                                                      state_info_wrap,
                                                      state_slider_wrap
                                                      ])])
    graph_area = html.Div(className='graph_area', style=graph_area_style,
                          children=[q_value_wrapper, greedy_action_wrapper, decomp_q_value_wrapper, action_pair_wrapper,
                                    rdx_wrapper, msx_wrapper])
    # game_control = html.Div(children=[state_slider_wrap, msx_checkbox], className='game_control')
    children = [game_area, graph_area]
    layout = html.Div(children=children)

    # create callbacks
    @app.callback(
        Output(component_id=prefix + 'max_state', component_property='children'),
        [Input(component_id=prefix + 'solver_dropdown', component_property='value')])
    def update_max_state_count(selected_base_solver_episode):
        selected_base_solver, episode = selected_base_solver_episode.split('-')
        episode = int(episode)
        return str(len(data[selected_base_solver][episode]['data']) - 1)

    @app.callback(
        Output(component_id=prefix + 'curr_state', component_property='children'),
        [Input(component_id=prefix + 'prev_state', component_property='n_clicks'),
         Input(component_id=prefix + 'next_state', component_property='n_clicks'),
         Input(component_id=prefix + 'solver_dropdown', component_property='value')],
        [State(component_id=prefix + 'curr_state', component_property='children'),
         State(component_id=prefix + 'max_state', component_property='children')])
    def update_state(prev_state_n_clicks, next_state_n_clicks, selected_solver_episode, curr_state, max_state):
        if selected_solver_episode == manager['curr_solver_episode']:
            curr_state, max_state = int(curr_state), int(max_state)
            prev_state_n_clicks = 0 if prev_state_n_clicks is None else prev_state_n_clicks
            next_state_n_clicks = 0 if next_state_n_clicks is None else next_state_n_clicks

            if manager['n_clicks'][prefix + 'prev_state'] < prev_state_n_clicks:
                curr_state = (curr_state - 1) if curr_state > 0 else 0
            elif manager['n_clicks'][prefix + 'next_state'] < next_state_n_clicks:
                curr_state = min(curr_state + 1, max_state)
            manager['n_clicks'][prefix + 'prev_state'] = prev_state_n_clicks
            manager['n_clicks'][prefix + 'next_state'] = next_state_n_clicks
        else:
            manager['curr_solver_episode'] = selected_solver_episode
            manager['n_clicks'][prefix + 'prev_state'] = 0
            manager['n_clicks'][prefix + 'next_state'] = 0
            curr_state = 0
        return str(curr_state)

    # create callbacks
    @app.callback(
        Output(component_id=prefix + 'state', component_property='children'),
        [Input(component_id=prefix + 'curr_state', component_property='children')],
        [State(component_id=prefix + 'solver_dropdown', component_property='value')])
    def update_state_info(curr_state, selected_solver_episode):
        selected_base_solver, episode = selected_solver_episode.split('-')
        episode = int(episode)
        curr_state = int(curr_state)
        html_state = [html.Span(children=row) for row in
                      data[selected_base_solver][episode]['data'][curr_state]['state'].split('\n')]
        return html_state

    # create callbacks
    @app.callback(
        Output(component_id=prefix + 'reward-info', component_property='children'),
        [Input(component_id=prefix + 'curr_state', component_property='children')],
        [State(component_id=prefix + 'solver_dropdown', component_property='value')])
    def update_reward_info(curr_state, selected_solver_episode):
        selected_base_solver, episode = selected_solver_episode.split('-')
        episode = int(episode)
        curr_state = int(curr_state)
        html_reward = [html.Div(children=row[0] + ':' + str(row[1])) for row in
                       data[selected_base_solver][episode]['data'][curr_state]['reward'].items()]
        return html_reward

    # create callbacks
    @app.callback(
        Output(component_id=prefix + 'terminal', component_property='children'),
        [Input(component_id=prefix + 'curr_state', component_property='children')],
        [State(component_id=prefix + 'solver_dropdown', component_property='value')])
    def update_terminal_info(curr_state, selected_solver_episode):
        selected_base_solver, episode = selected_solver_episode.split('-')
        episode = int(episode)
        curr_state = int(curr_state)
        return html.Span(str(data[selected_base_solver][episode]['data'][curr_state]['terminal']))

    for solver in solvers:

        @app.callback(
            Output(component_id=prefix + 'q-value' + solver, component_property='figure'),
            [Input(component_id=prefix + 'curr_state', component_property='children')],
            [State(component_id=prefix + 'decomp-q-value' + solver, component_property='className'),
             State(component_id=prefix + 'solver_dropdown', component_property='value')])
        def update_q_graphs(i, solver, selected_base_solver_episode):
            selected_base_solver, episode = selected_base_solver_episode.split('-')
            episode = int(episode)
            i = int(i)
            y_data = []
            for a in range(len(actions)):
                y_data.append(sum(
                    data[selected_base_solver][episode]['data'][i]['solvers'][solver]['q_values'][rt_i][a] for rt_i in
                    range(len(reward_types))))
            figure = {
                'data': [{'x': actions,
                          'y': y_data,
                          'type': 'bar',
                          }],
                'layout': {
                    'title': solver,
                    'showgrid': True
                }
            }
            return figure

        @app.callback(
            Output(component_id=prefix + 'decomp-q-value' + solver, component_property='figure'),
            [Input(component_id=prefix + 'curr_state', component_property='children')],
            [State(component_id=prefix + 'decomp-q-value' + solver, component_property='className'),
             State(component_id=prefix + 'solver_dropdown', component_property='value')])
        def update_decomp_q_graphs(i, solver, selected_base_solver_episode):
            selected_base_solver, episode = selected_base_solver_episode.split('-')
            episode = int(episode)
            i = int(i)
            rt_data = []
            for rt_i, rt in enumerate(reward_types):
                rt_data.append({'x': actions,
                                'y': [
                                    data[selected_base_solver][episode]['data'][i]['solvers'][solver]['q_values'][rt_i][
                                        a]
                                    for a in range(len(actions))],
                                'type': 'bar',
                                'name': rt,
                                'marker': {'color': reward_colors[rt]}
                                })
            figure = {
                'data': rt_data,
                'layout': {
                    'title': solver,
                    'showgrid': True
                }
            }
            return figure

        @app.callback(
            Output(component_id=prefix + 'action_pair_selector-' + solver, component_property='value'),
            [Input(component_id=prefix + 'curr_state', component_property='children')],
            [State(component_id=prefix + 'decomp-q-value' + solver, component_property='className'),
             State(component_id=prefix + 'solver_dropdown', component_property='value')])
        def update_action_pair(state, solver, selected_base_solver_episode):
            selected_base_solver, episode = selected_base_solver_episode.split('-')
            episode = int(episode)
            state = int(state)
            greedy_act = data[selected_base_solver][episode]['data'][state]['solvers'][solver]['action']
            return str(greedy_act) + '_' + str([x for x in range(len(actions)) if x != greedy_act][0])

        @app.callback(
            Output(component_id=prefix + 'greedy_action-' + solver, component_property='children'),
            [Input(component_id=prefix + 'curr_state', component_property='children')],
            [State(component_id=prefix + 'decomp-q-value' + solver, component_property='className'),
             State(component_id=prefix + 'solver_dropdown', component_property='value')])
        def update_greedy_action(state, solver, selected_base_solver_episode):
            selected_base_solver, episode = selected_base_solver_episode.split('-')
            episode = int(episode)
            state = int(state)
            greedy_act = data[selected_base_solver][episode]['data'][state]['solvers'][solver]['action']
            return 'Action: ' + actions[greedy_act]

        @app.callback(
            Output(component_id=prefix + 'rdx-' + solver, component_property='figure'),
            [Input(component_id=prefix + 'action_pair_selector-' + solver, component_property='value')],
            [State(component_id=prefix + 'curr_state', component_property='children'),
             State(component_id=prefix + 'rdx-' + solver, component_property='className'),
             State(component_id=prefix + 'solver_dropdown', component_property='value')])
        def update_rdx_graphs(action_pair, state, solver, selected_base_solver_episode):
            selected_base_solver, episode = selected_base_solver_episode.split('-')
            episode = int(episode)
            state = int(state)
            first_action, sec_action = [int(a) for a in action_pair.split('_')]
            rt_data = []
            for rt_i, rt in enumerate(reward_types):
                rt_data.append({'x': [''],
                                'y': [data[selected_base_solver][episode]['data'][state]['solvers'][solver]['rdx'][
                                        first_action][sec_action][rt]],
                                'type': 'bar',
                                'name': rt,
                                'marker': {'color': reward_colors[rt]}})
            rt_data.sort(key=lambda x: x['y'][0], reverse=True)

            figure = {
                'data': rt_data,
                'layout': {
                    'title': solver,
                    'showlegend': True,
                    'showgrid': True
                }
            }
            return figure

        @app.callback(
            Output(component_id=prefix + 'msx-' + solver, component_property='figure'),
            [Input(component_id=prefix + 'action_pair_selector-' + solver, component_property='value')],
            [State(component_id=prefix + 'curr_state', component_property='children'),
             State(component_id=prefix + 'msx-' + solver, component_property='className'),
             State(component_id=prefix + 'solver_dropdown', component_property='value')])
        def update_msx_graphs(action_pair, state, solver, selected_base_solver_episode):
            selected_base_solver, episode = selected_base_solver_episode.split('-')
            episode = int(episode)
            state = int(state)
            first_action, sec_action = [int(a) for a in action_pair.split('_')]

            msx_data = data[selected_base_solver][episode]['data'][state]['solvers'][solver]['msx'][first_action][
                sec_action]
            pos_msx_data, neg_msx_data = msx_data

            pos_rt_data, neg_rt_data = [], []
            for rt in pos_msx_data:
                pos_rt_data.append(go.Bar(
                    x=[''],
                    y=[data[selected_base_solver][episode]['data'][state]['solvers'][solver]['rdx'][
                                 first_action][sec_action][rt]],
                    name=rt,
                    marker={'color': reward_colors[rt]}
                ))
            for rt in neg_msx_data:
                neg_rt_data.append(go.Bar(
                    x=[''],
                    y=[data[selected_base_solver][episode]['data'][state]['solvers'][solver]['rdx'][
                                 first_action][sec_action][rt]],
                    name=rt,
                    marker={'color': reward_colors[rt]}
                ))
            pos_rt_data.sort(key=lambda x: x['y'][0], reverse=True)
            neg_rt_data.sort(key=lambda x: x['y'][0], reverse=True)

            fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('+ve', '-ve'), shared_yaxes=True)
            for _rt_data in pos_rt_data:
                fig.append_trace(_rt_data, 1, 1)
            for _rt_data in neg_rt_data:
                fig.append_trace(_rt_data, 1, 2)
            fig['layout'].update(
                title=solver,
                showlegend=True
            )
            return fig

    return layout


def train_page_layout(app, train_data, run_mean_data, q_val_dev_data, solvers, runs=1, prefix=''):
    solver_colors = {s: cl.scales['7']['qual']['Dark2'][i] for i, s in enumerate(solvers)}
    train_traces, run_mean_traces, q_val_dev_traces = [], [], []
    for solver in solvers:
        train_trace = {
            'x': [_ + 1 for _ in range(len(train_data[solver]))],
            'y': train_data[solver],
            'type': 'scatter',
            'mode': 'lines',
            'name': solver,
            'line': {'color': solver_colors[solver]}
        }
        run_mean_trace = {
            'x': [_ + 1 for _ in range(len(run_mean_data[solver]))],
            'y': run_mean_data[solver],
            'type': 'scatter',
            'mode': 'lines',
            'name': solver,
            'line': {'color': solver_colors[solver]}
        }
        if solver in q_val_dev_data:
            q_val_dev_trace = {
                'x': [_ + 1 for _ in range(len(q_val_dev_data[solver]))],
                'y': q_val_dev_data[solver],
                'type': 'scatter',
                'mode': 'lines',
                'name': solver,
                'line': {'color': solver_colors[solver]}
            }
            q_val_dev_traces.append(q_val_dev_trace)
        train_traces.append(train_trace)
        run_mean_traces.append(run_mean_trace)

    train_graph = html.Div(className='graph_box', children=[
        dcc.Graph(
            id=prefix + '-train',
            figure={
                'data': train_traces,
                'layout': {
                    'title': 'Running Score',
                    'showlegend': True,
                    'showgrid': True
                }
            }
        )])
    run_mean_graph = html.Div(className='graph_box', children=[
        dcc.Graph(
            id=prefix + '-run_mean',
            figure={
                'data': run_mean_traces,
                'layout': {
                    'title': 'Running Score Mean',
                    'showlegend': True,
                    'showgrid': True
                }
            }
        )])
    q_val_dev_graph = html.Div(className='graph_box', children=[
        dcc.Graph(
            id=prefix + '-q_val_dev',
            figure={
                'data': q_val_dev_traces,
                'layout': {
                    'title': 'Q value Deviation',
                    'showlegend': True,
                    'showgrid': True
                }
            }
        )])

    info_box = html.Div(children='Runs:' + str(runs))
    layout = html.Div(children=[train_graph, run_mean_graph, q_val_dev_graph, info_box])
    return layout


def visualize_results(result_path, host, port):
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css',
                            'https://codepen.io/koulanurag/pen/maYYKN.css']
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
    app.config.suppress_callback_exceptions = True

    envs = [(o, os.path.join(result_path, o)) for o in os.listdir(result_path)
            if os.path.isdir(os.path.join(result_path, o))]
    index_children = []
    layouts = {}
    for env, env_path in envs:
        layouts[env] = {}
        x_path = os.path.join(env_path, 'x_data.p')
        train_path = os.path.join(env_path, 'train_data.p')

        if os.path.exists(train_path):
            prefix = env.lower() + '_training'
            _path = '/' + prefix
            train_page = dcc.Link(env + ': Training ', href=_path)
            train_data = pickle.load(open(train_path, 'rb'))
            layouts[_path] = train_page_layout(app, train_data['data'], train_data['run_mean'],
                                               train_data['q_val_dev'], train_data['data'].keys(),
                                               runs=train_data['runs'], prefix=prefix)
            index_children.append(train_page)
            index_children.append(html.Br())

        if os.path.exists(x_path):
            prefix = env.lower() + '_explanations'
            _path = '/' + prefix
            x_page = dcc.Link(env + ': Explanations', href=_path)
            x_data = pickle.load(open(x_path, 'rb'))
            layouts[_path] = x_layout(app, x_data['data'], x_data['reward_types'],
                                      x_data['actions'], prefix)
            index_children.append(x_page)
            index_children.append(html.Br())

    app.layout = html.Div([
        dcc.Location(id='url', refresh=False),
        html.Div(id='page-content')
    ])

    index_page = html.Div(children=index_children)

    @app.callback(Output('page-content', 'children'),
                  [Input('url', 'pathname')])
    def display_page(pathname):
        if pathname in layouts:
            return layouts[pathname]
        return index_page

    app.run_server(debug=False, port=port, host=host)
