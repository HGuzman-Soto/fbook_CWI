import plotly.express as px
import dash_core_components as dcc
import dash_html_components as html
import plotly.figure_factory as ff
import scipy

from dash.dependencies import Input, Output
import pandas as pd
import dash

df = pd.read_csv('results/all_feats_ada/news_all_feats_ada_train_results.csv')
# fig = px.scatter(df, x='syllables', y='output', size='length', color='complex_binary')
# fig.show()
print(len([df['length']]))
fig = ff.create_distplot([df['length']], ['length'])
# fig.show()




app = dash.Dash(__name__)
server = app.server
# word = df['word']
category = df.columns
category.sort_values()
print(category)
app.layout = html.Div([
    html.Div([dcc.Dropdown(id='group-select', options=[{'label': i, 'value': i} for i in category],
                           value='TOR', style={'width': '140px'})]),
    dcc.Graph('shot-dist-graph', config={'displayModeBar': False})])

@app.callback(
    Output('shot-dist-graph', 'figure'),
    [Input('group-select', 'value')]
)
def update_graph(category):
    return ff.create_distplot([df[category]], [category], bin_size=5)

    # return px.scatter(df[df['word'] == word], x='dep num', y='google frequency', size='length', color='complex_binary')

if __name__ == '__main__':
    app.run_server(host = '127.0.0.1')
