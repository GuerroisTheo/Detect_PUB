import cv2
import numpy as np
import os
from PIL import ImageGrab,ImageFilter
import time
import repeatedTime
from keras import models

# c'est pour avoir tout l'écran
from ctypes import windll
user32 = windll.user32
user32.SetProcessDPIAware()

# model = models.load_model('bestmodel.h5')

CATEGORIES = ["LOGO","PUB"]
g_repscreen = None
tempsPub = []
g_bloqueur = 1
t1 = 0
t2 = 0

def init():
    global g_repscreen, g_bloqueur
    g_repscreen = repeatedTime.RepeatedTimer(1,timer)
    
    startAll()

def startAll():
    global g_repscreen, g_bloqueur
    g_repscreen.start()


def screen():

    # if(y != 0): #dont mind me im a test (y & n were global)
    #     print('yes')
    #     retourne = "yes"
    #     y = y - 1
    # if(n != 0 and y == 0):
    #     print('no')
    #     retourne = "no"
    #     n = n - 1
    screen = np.array(ImageGrab.grab(bbox=(1594, 41, 1902 , 137)))
    image = cv2.cvtColor((screen), cv2.COLOR_BGR2RGB)
    return image
    
    
    # prediction = model.predict(image)
    # return CATEGORIES[int(prediction[0][0])]

def timer():
    global g_repscreen, g_bloqueur, tempsPub, t1, t2
    cat = screen()

    if cat == "PUB" and g_bloqueur == 1:
        t1 = time.time()
        g_bloqueur = 0
    if cat != "PUB" and g_bloqueur == 0:
        t2 = time.time()
        tempsPub.append(t2-t1)
        print(t2-t1)
        g_bloqueur = 1



if __name__ == "__main__":
    init()
    while(True):
        if 0xFF == ord('q'):
            g_repscreen.stop()





