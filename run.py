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
"""All imports necessary to carry out this process"""
import cv2
import keylog
import numpy as np
import pandas as pd
import os
from PIL import ImageGrab,ImageFilter, Image, ImageOps
import time
import repeatedTime
import collections
from keras import utils, layers, optimizers, models
from keras.preprocessing import image


"""Cv2 and image grab don't take the whole screen"""
from ctypes import windll
user32 = windll.user32
user32.SetProcessDPIAware()

modelchaine = 'france2.h5'
model = models.load_model('.\\Modeles\\'+modelchaine) #TF1model

g_queue = collections.deque([0.,0.,0.,0.])
g_tempsatt = 4

datagen = image.ImageDataGenerator(rescale=1./255)

CATEGORIES = ["LOGO","PUB"]
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
g_repscreen = None
tempsPub = []
g_klog = None
g_bloqueur = 1
t1 = 0
t2 = 0
ttotal = 0
nombrePub = 0

def init():
    """This function instantiates all our global variables."""
    global g_repscreen, g_bloqueur, g_klog, ttotal
    g_repscreen = repeatedTime.RepeatedTimer(1,screen)
    g_klog = keylog.KeyLogger()
    ttotal = time.time()
    #print(ttotal)
    startAll()

def startAll():
    """Starts listening to the keyboard and recording the screen."""
    global g_repscreen, g_klog
    g_repscreen.start()
    g_klog.start()


def stopAll():
    global g_repscreen, g_klog, t2

    g_repscreen.stop()
    g_klog.stop()


def screen():
    global g_tempsatt, tempsPub, data

    """Give the AI an image of the top right corner and return if its advertising or logo"""
    if (g_klog.a_stopMain):
        screen = ImageGrab.grab(bbox=(1594, 41, 1902 , 137)) #Take the top right corner image
        os_path = os.getcwd()+"\puber\images"
        FILES_DIR = os_path
        SAVE_PATH = os_path
        LOGFILE_NAME = "puber.png"

        LOGFILE_PATH = os.path.join(SAVE_PATH, FILES_DIR, LOGFILE_NAME)
        screen.save(LOGFILE_PATH)

        #########Option 1

        test = datagen.flow_from_directory("./puber", class_mode=None, target_size=(100,100), batch_size=1)
        prediction = model.predict(test[0])

        #########Option 2

        image = Image.open('./puber/images/puber.png')

        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image_array = np.asarray(image)
        #image.show()
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array
        prediction = model.predict(data)

        #########Final

        labels = np.argmax(prediction, axis=1) #0 if logo / 1 if pub

        print(CATEGORIES[labels[0]])
        
        g_queue.append(labels) #Add it to the queue

        if len(g_queue) > g_tempsatt : 
                timer(g_queue)
                taillemaxqueue(g_tempsatt,g_queue)
    else:
        stopAll()
        print("Temps pub : " + str(sum(tempsPub)))
        print("Nombre de pubs : " + str(nombrePub))
        print("Temps total visionnage : " + str(time.time() - ttotal))
        print('You have just watch {0} sec of Pub equals to {1} min during {2} min'.format(sum(tempsPub), sum(tempsPub)/60, (time.time() - ttotal)/60))
        updatecsv()


def taillemaxqueue(max,queue):
    """Restriction of the maximum lenght of the queue"""
    if len(queue)>max:
        queue.popleft()
        taillemaxqueue(max,queue)

def timer(g_queue):
    """Append pub duration in a list"""
    global g_repscreen, g_bloqueur, tempsPub, t1, t2, nombrePub

    labels = list(g_queue)
    #print(labels)
    #print(labels.count(1))
    #print(labels.count(0))

    if labels.count(1) == 5 or labels.count(0) == 5:

        if CATEGORIES[int(labels[-1])] == "PUB" and g_bloqueur == 1:
            t1 = time.time()
            #print(t1)
            nombrePub = nombrePub + 1
            g_bloqueur = 0
            #print(g_bloqueur)
        if CATEGORIES[int(labels[-1])] != "PUB" and g_bloqueur == 0:
            t2 = time.time()
            tempsPub.append(t2-t1)
            print(t2-t1)
            g_bloqueur = 1 
    else:
        pass

def updatecsv():
    os_path = os.getcwd()
    df = pd.read_csv(os_path+"\\df_channel")
    modellist = ["tf1.h5", "france2.h5", "france3.h5", "france4.h5", "france5.h5", "c8.h5", "tmc.h5", "tfx.h5", "nrj12.h5", "lcp.h5", "bfm.h5", "cnews.h5", "cstar.h5", "gulli.h5"]

    if (modelchaine in modellist):
        i = modellist.index(modelchaine)
        df.at[i,'Temps de pub en moyenne par heure (en minutes)'] = (sum(tempsPub)*100)/(time.time()-ttotal)
        df.to_csv('.\\df_channel', index=False)
    
    

if __name__ == "__main__":
    init()