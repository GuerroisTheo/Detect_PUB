import cv2
import numpy as np
import os
from PIL import ImageGrab,ImageFilter
import time
import repeatedTime

# c'est pour avoir tout l'écran
from ctypes import windll
user32 = windll.user32
user32.SetProcessDPIAware()

g_repscreen = repeatedTime.RepeatedTimer(1,screen)

def init():
    g_repscreen.start()

def screen():
    return ImageGrab.grab(bbox=(1594, 41, 1902 , 137))


if __name__ == "__main__":
    init()
    while(True):
        if cv2.waitKey(25)& 0xFF == ord('q'):
            g_repscreen.stop()
            cv2.destroyAllWindows()
            break




