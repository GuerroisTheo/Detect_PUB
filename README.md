# Detect_PUB

Detect_PUB est un projet de détection de publicité sur les chaines de la TNT.

## Installation
### Librairies
Voici la liste des librairies utilisés pour ce projet :

* cv2 (opencv-python)
* numpy
* keylog
* PIL (Pillow)
* time
* keras
* tensorflow
* sklearn
* matplotlib
* plotly

On peut les installer en utilisant la commande suivante :
```
pip install "nom de la librairie"
```
## Lancement
### Modèles
Nous avons décidé de faire un modèle par chaîne de télévision pour être plus précis.

Voici la liste des modèles utilisés pour ce projet :
```
["tf1.h5", "france2.h5", "france3.h5", "france4.h5", "france5.h5", "c8.h5", "tmc.h5", "tfx.h5", "nrj12.h5", "okoo.h5", "bfm.h5", "cnews.h5", "cstar.h5", "gulli.h5"]
```

Pour choisir le modèle que l’on souhaite utiliser, il suffit d’écrire le nom du modèle correspondant à la chaîne que l’on regarde et de lancer le programme.

A la ligne 19 dans le fichier run.py, on peut choisir quel modèle on souhaite utiliser.
```
model = models.load_model(./Modeles/“nom du modèle”)
```

