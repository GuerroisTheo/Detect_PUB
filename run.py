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

CATEGORIES = ["LOGO", "PUB"]

g_repscreen = repeatedTime.RepeatedTimer(1,screen)

def init():
    g_repscreen.start()

def screen():
    image = ImageGrab.grab(bbox=(1594, 41, 1902 , 137))
    prediction = model.predict(image)
    print(CATEGORIES[int(prediction[0][0])])
    return prediction


if __name__ == "__main__":
    init()
    while(True):
        if cv2.waitKey(25)& 0xFF == ord('q'):
            g_repscreen.stop()
            cv2.destroyAllWindows()
            break




