
"""All imports necessary to carry out this process"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output 
import plotly.express as px
import pandas as pd
import os

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

def main():
    app = dash.Dash(__name__)

    os_path = os.getcwd()
    # df = pd.DataFrame({
    #     "Chaînes": ["TF1", "France 2", "France 3", "France 4", "France 5", "C8", "TMC", "TFX", "NRJ12", "LCP", "BFM TV", "CNews", "CStar", "Gulli" ],
    #     "Temps de pub en moyenne par heure (en minutes)": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # })
    # df.to_csv('.\\df_channel', index=False)
    df = pd.read_csv(os_path+"\\df_channel")
    df.set_index('Chaînes', inplace = True)

    fig = px.bar(df, x=list(df.index), y="Temps de pub en moyenne par heure (en minutes)", barmode="group")

    app.layout = html.Div(children=[
        html.H1(children='Projet Publicité E4'),

        dcc.Dropdown(
            id="chanelpicker",
            style={"width" : "50%"},
            options=[
                {'label': 'TF1', 'value':'TF1'},
                {'label': 'France 2', 'value':'France 2'},
                {'label': 'France 3', 'value':'France 3'},
                {'label': 'France 4', 'value':'France 4'},
                {'label': 'France 5', 'value':'France 5'},
                {'label': 'C8', 'value':'C8'},
                {'label': 'W9', 'value':'W9'},
                {'label': 'TMC', 'value':'TMC'},
                {'label': 'TFX', 'value':'TFX'},
                {'label': 'NRJ12', 'value':'NRJ12'},
                {'label': 'LCP', 'value':'LCP'},
                {'label': 'BFM TV', 'value':'BFM TV'},
                {'label': 'CNews', 'value':'CNews'},
                {'label': 'CStar', 'value':'Cstar'},
                {'label': 'Gulli', 'value':'Gulli'}],
            multi=True,
            value=['TF1']
        ),

        html.Div(id="output_container1", children=[]),

        html.Br(),

        html.Div(children='''
             Poucentage de pub observé sur la durée totale
        '''),

        dcc.Graph(
            id='example-graph',
            figure=fig
        )
    ])

    @app.callback(
        [Output(component_id='example-graph', component_property='figure'),
         Output(component_id='output_container1', component_property='children')
        ],
        [Input(component_id='chanelpicker', component_property='value')
        ]
    )

    def update_output(value):
        container = "Vous avez sélectionné : {}".format(value)


        if(len(container) != 0):
            fig = px.bar(x=list(df.T[value].T.index), y=df.T[value].T["Temps de pub en moyenne par heure (en minutes)"], barmode="group")

        return fig, container
    
    app.run_server(debug=True)

    return None

if __name__ == '__main__':
    main()
