SQuiich
#9194

Coralie Secondo — 28/01/2021
nickel
je suis en réunion junior du coup mais si tu valides j'envoie à 19h
Lilian — 28/01/2021
C'est envoyé du coup?
Coralie Secondo — 28/01/2021
j'étais en train de faire le mail
Lilian — 28/01/2021
ok
attends il faut recréer le pdf
Coralie Secondo — 28/01/2021
c'est fait tkt
Lilian — 28/01/2021
ok nice
C'est un plaisir de te voir, 
Eziou
.
 — 19/02/2021
Lilian — 09/03/2021
Commandes intéressantes pour le traitement d'image sur ImageMagick (à tester sur run.py) : https://legacy.imagemagick.org/Usage/blur/
Lilian — 20/03/2021
Bon on s'est pas parlé depuis longtemps mais j'ai essayé de faire des trucs de mon côté.
En premier lieu j'ai essayé de refaire notre database qui avait des photos en double donc pour l'instant j'ai trié les photos par chaîne et on pourra faire un nouveau modèle (j'ai entraîné un modèle temporaire seulement sur tf1 et il a un meilleur taux de précision même s'il reste quelques erreurs de détection)
Ensuite j'ai essayé certaines commandes ImageMagick et il y en a qui sont plus efficaces que d'autres (à voir si je peux les utiliser pour créer un autre modèle)
Je peux passer de ça

à ça

Par contre je sais pas encore si ça peux améliorer la précision de l'IA
Vu que ça vire les couleurs
Le problème c'est que sur d'autres images c'est beaucoup plus dur à détecter comme ici :


Coralie Secondo — 11/04/2021
@everyone Hello les gars, Lilian, j'ai vu que t'avais un peu taffé sur le projet, tu ene s où ? est ce que la dernière version est sur Git ?
Lilian — 11/04/2021
j'ai tout push
Coralie Secondo — 11/04/2021
@everyone bon je viens d'apprendre un truc qui va vous plaire (c'est faux). Avant jeudi midi il faut faire une vidéo de présentation du projet !

Manuel — 11/04/2021
Relou on est au courant de rien
Coralie Secondo — 12/04/2021
@everyone ok les gars on a une réu avec le prof à 13h, on est avec Manu en train d'essayer de comprendre le prgramme donc si vous êtes dispo ce serait cool de venir
on est sur le serveur
Manuel — 12/04/2021
J'ai avancé sur le dashboard

J'ai repris ce qui avait déjà été mis en place et j'ai fait en sorte d'avoir le graph de manière interactive
Il faut juste mettre un fichier csv à jour à la fin de prédiction avec le nom de la chaine et la moyenne et ça se mettra automatiquement à jour sur le dash
Pour ca on aura besoin de quelqu'un qui connait le programme de prédiction, car il y a un soucis quand on essaye de lancer le programme
Donc dites moi si quelqu'un est dispo prochainement pour ça
Manuel — Aujourd’hui à 13:06
@Théo Nicolas
@Fangming ZOU
Théo Nicolas — Aujourd’hui à 13:41
https://github.com/GuerroisTheo/Detect_PUB/blob/faa53074fe0d545e46987caac2b1a11591be7d74/Detection_Pub.py
GitHub
GuerroisTheo/Detect_PUB
Contribute to GuerroisTheo/Detect_PUB development by creating an account on GitHub.

Manuel — Aujourd’hui à 13:46
["TF1", "France 2", "France 3", "France 4", "France 5", "M6", "C8", "W9", "TMC", "TFX", "NRJ12", "LCP", "BFM TV", "CNews", "CStar", "Gulli", "France Ô", "6ter" ]
Coralie Secondo — Aujourd’hui à 13:49
France 4 France 5 C8 TFX NRJ12 LCP BFM Cnews, Cstar, France ô
Théo : France 4
Manu : TFX
Adrien — Aujourd’hui à 13:58
Adrien : c8
Adrien — Aujourd’hui à 14:10

Coralie Secondo — Aujourd’hui à 14:24
Logo gauche : BFM, C News, LCP
Coralie Secondo — Aujourd’hui à 14:35
Coralie = W9
Fangming ZOU — Aujourd’hui à 14:36

Théo Nicolas — Aujourd’hui à 15:03
theo : gulli
Fangming ZOU — Aujourd’hui à 15:36

TF1 ne fonctionne pas?
Adrien — Aujourd’hui à 16:49
["tf1.h5", "france2.h5", "france3.h5", "france4.h5", "france5.h5", "c8.h5", "tmc.h5", "tfx.h5", "nrj12.h5", "lcp.h5", "bfm.h5", "cnews.h5", "cstar.h5", "gulli.h5"]
Théo Nicolas — Aujourd’hui à 17:06
Je reviens apres je donne un cours de maths, si y a besoin de moi dites moi, je suis en distanciel donc je peux faire des trucs a coté
Coralie Secondo — Aujourd’hui à 17:35
        #img = ImageGrab.grab(bbox=(1594, 41, 1902 , 137))#à gauche
        img = ImageGrab.grab(bbox=(0, 41, 308 , 137))#à droite
Adrien — Aujourd’hui à 17:52

Manuel — Aujourd’hui à 19:56
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output 
import plotly.express as px
import pandas as pd
Afficher plus
testdash.py
3 Ko
"""All imports necessary to carry out this process"""
import cv2
import keylog
import numpy as np
import pandas as pd
import os
Afficher plus
run.py
5 Ko
Type de fichier joint : unknown
df_channel
213 bytes
﻿
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