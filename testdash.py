import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__)

df = pd.DataFrame({
    "Chaînes": ["TF1", "France 2"],
    "Temps de pub en moyenne par heure (en minutes)": [14.89, 11.6] #à compléter
})

fig = px.bar(df, x="Chaînes", y="Temps de pub en moyenne par heure (en minutes)", barmode="group")

app.layout = html.Div(children=[
    html.H1(children='Projet Publicité E4'),

    html.Div(children='''
        Temps de pub pour chaque chaîne analysée :
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)