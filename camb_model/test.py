from dash import Dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

name = "wikipedia_test_all_data_ada_results"
df = pd.read_csv("results/" + name + ".csv")

app = Dash()

app.layout = html.Div([
    dcc.Graph(id="scatter-plot"),
    html.P("Google Frequency:"),
    dcc.RangeSlider(
        id='range-slider',
        min=0, max=900, step=10,
        marks={0: '0', 900: '900'},
        value=[0, 10]
    ),
])


@app.callback(
    Output("scatter-plot", "figure"),
    [Input("range-slider", "value")])
def update_bar_chart(slider_range):
    low, high = slider_range
    mask = (df['google frequency'] > low) & (df['google frequency'] < high)
    fig = px.scatter(
        df[mask], x="word", y="google frequency",
        color="output",
        hover_data=['google frequency'])
    return fig


app.run_server(host='127.0.0.1')
