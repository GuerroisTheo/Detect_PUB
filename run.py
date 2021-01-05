import cv2
import keylog
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
g_klog = None
g_bloqueur = 1
t1 = 0
t2 = 0

def init():
    global g_repscreen, g_bloqueur, g_klog
    g_repscreen = repeatedTime.RepeatedTimer(1,timer)
    g_klog = keylog.KeyLogger()
    startAll()

def startAll():
    global g_repscreen, g_klog
    g_repscreen.start()
    g_klog.start()


def stopAll():
    global g_repscreen, g_klog
    g_repscreen.stop()
    g_klog.stop()


def screen():
    screen = np.array(ImageGrab.grab(bbox=(1594, 41, 1902 , 137)))
    image = cv2.cvtColor((screen), cv2.COLOR_BGR2RGB)
    return image
    
    
    # prediction = model.predict(image)
    # return CATEGORIES[int(prediction[0][0])]

def timer():
    global g_repscreen, g_bloqueur, tempsPub, t1, t2, g_klog
    
    if (g_klog.a_stopMain):
        cat = screen()

        if cat == "PUB" and g_bloqueur == 1:
            t1 = time.time()
            g_bloqueur = 0
        if cat != "PUB" and g_bloqueur == 0:
            t2 = time.time()
            tempsPub.append(t2-t1)
            print(t2-t1)
            g_bloqueur = 1
    else:
        stopAll()


if __name__ == "__main__":
    init()





