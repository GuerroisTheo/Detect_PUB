import cv2
import keylog
import numpy as np
import os
from PIL import ImageGrab,ImageFilter
import time
import repeatedTime
from keras import utils, layers, optimizers, models
from keras.preprocessing import image


# c'est pour avoir tout l'écran
from ctypes import windll
user32 = windll.user32
user32.SetProcessDPIAware()

model = models.load_model('param.h5')

datagen = image.ImageDataGenerator(rescale=1./255, validation_split=0.2)

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
    screen = ImageGrab.grab(bbox=(1594, 41, 1902 , 137))
    #image = cv2.cvtColor((screen), cv2.COLOR_BGR2RGB)

    FILES_DIR = 'C:/Users/TLG/Desktop/IAImage/ProjetPubE4/Detect_PUB/puber/images'
    SAVE_PATH = 'C:/Users/TLG/Desktop/IAImage/ProjetPubE4/Detect_PUB/puber/images'
    #SAVE_PATH = os.path.expanduser("~")    #It is cross-platform
    LOGFILE_NAME = "puber.png"

    LOGFILE_PATH = os.path.join(SAVE_PATH, FILES_DIR, LOGFILE_NAME)
    screen.save(LOGFILE_PATH)

    #image = cv2.resize(image, (40,40))
    #return image

    #test = cv2.imread('C:/Users/TLG/Desktop/IAImage/ProjetPubE4/Detect_PUB//puber/puber.png', cv2.IMREAD_UNCHANGED)
    #print(test.shape)
    #test.shape = (1,96,308,3)
    #print(test.shape)

    test = datagen.flow_from_directory("./puber", class_mode=None, target_size=(40,40), batch_size=1)
    
    prediction = model.predict(test[0])
    labels = np.argmax(prediction, axis=1)
    print(labels)
    #print(CATEGORIES[int(prediction[0][0])])
    return CATEGORIES[int(prediction[0][0])]

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





