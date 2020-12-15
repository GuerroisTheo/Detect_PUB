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
tempsPub = []
g_bloqueur = 1

def init():
    g_repscreen.start()

def screen():
    screen = np.array(ImageGrab.grab(bbox=(1594, 41, 1902 , 137)))
    image = cv2.cvtColor((screen), cv2.COLOR_BGR2RGB)
    return image
    
    
    # prediction = model.predict(image)
    # return CATEGORIES[int(prediction[0][0])]

def timer():
    global g_repscreen, g_bloqueur
    cat = screen()


    # if cat == "PUB" and g_bloqueur == 1:
    #     t1 = time.time()
    #     g_bloqueur = 0
    # if cat != "PUB" and g_bloqueur == 0:
    #     t2 = time.time()
    #     tempsPub.append(t2-t1)
    #     g_bloqueur = 1


g_repscreen = repeatedTime.RepeatedTimer(1,timer)

if __name__ == "__main__":
    init()
    while(True):
        if 0xFF == ord('q'):
            g_repscreen.stop()





