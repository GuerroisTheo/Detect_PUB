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

model = models.load_model('bestmodel.h5')

CATEGORIES = ["LOGO","PUB"]
tempsPub = []
blocker = 1

g_repscreen = repeatedTime.RepeatedTimer(1,timer)

def init():
    g_repscreen.start()

def timer():
    cat = screen()
    if cat == "PUB" and blocker == 1:
        t1 = time.time()
        blocker = 0
    if cat != "PUB" and blocker == 0:
        t2 = time.time()
        tempsPub.append(t2-t1)
        blocker = 1

def screen():
    image = ImageGrab.grab(bbox=(1594, 41, 1902 , 137))
    prediction = model.predict(image)
    return CATEGORIES[int(prediction[0][0])]

if __name__ == "__main__":
    init()
    while(True):
        if 0xFF == ord('q'):
            g_repscreen.stop()
            break




