"""All imports necessary to carry out this process"""
import cv2
import keylog
import numpy as np
import os
from PIL import ImageGrab,ImageFilter
import time
import repeatedTime
import collections
from keras import utils, layers, optimizers, models
from keras.preprocessing import image


"""Cv2 and image grab don't take the whole screen"""
from ctypes import windll
user32 = windll.user32
user32.SetProcessDPIAware()


"""Global variable initialization"""
model = models.load_model('param.h5')
g_queue = collections.deque([0.,0.,0.,0.,0.])
g_tempsatt = 5
datagen = image.ImageDataGenerator(rescale=1./255, validation_split=0.2)
CATEGORIES = ["LOGO","PUB"]
g_repscreen = None
tempsPub = []
g_klog = None
g_bloqueur = 1
g_temps_global = 0
g_temps_global_stop = 0
t1 = 0
t2 = 0

def init():
    """This function instantiates all our global variables."""
    global g_repscreen, g_bloqueur, g_klog, g_temps_global
    g_temps_global = time.time()
    g_repscreen = repeatedTime.RepeatedTimer(1,screen)
    g_klog = keylog.KeyLogger()
    startAll()

def startAll():
    """Starts listening to the keyboard and recording the screen."""
    global g_repscreen, g_klog
    g_repscreen.start()
    g_klog.start()


def stopAll():
    """Stops all processes."""
    global g_repscreen, g_klog
    g_repscreen.stop()
    g_klog.stop()


def screen():
    """Give the AI an image of the top right corner and return if its advertising or logo"""
    global g_tempsatt, tempsPub
    if (g_klog.a_stopMain):
        screen = ImageGrab.grab(bbox=(1594, 41, 1902 , 137)) #Take the top right corner image
        os_path = os.getcwd()+"\puber\images"
        FILES_DIR = os_path
        SAVE_PATH = os_path
        LOGFILE_NAME = "puber.png"

        LOGFILE_PATH = os.path.join(SAVE_PATH, FILES_DIR, LOGFILE_NAME)
        screen.save(LOGFILE_PATH)
        test = datagen.flow_from_directory("./puber", class_mode=None, target_size=(40,40), batch_size=1)
        prediction = model.predict(test[0]) #The AI compare the image to her database 

        labels = np.argmax(prediction, axis=1) #0 if logo / 1 if pub

        print(CATEGORIES[labels[0]])
        
        g_queue.append(labels) #Add it to the queue

        if len(g_queue) > g_tempsatt : 
                timer(g_queue)
                taillemaxqueue(g_tempsatt,g_queue)
    else:
        stopAll()
        g_temps_global_stop = time.time()
        print('You have just watch {0} sec of Pub equals to {1} min during {2} min'.format(sum(tempsPub), sum(tempsPub)/60, (g_temps_global_stop-g_temps_global)/60))


def taillemaxqueue(max,queue):
    """Restriction of the maximum lenght of the queue"""
    if len(queue)>max:
        queue.popleft()
        taillemaxqueue(max,queue)

def timer(g_queue):
    """Append pub duration in a list"""
    global g_repscreen, g_bloqueur, tempsPub, t1, t2, g_klog

    labels = list(collections.deque(g_queue))

    if labels.count(1) == 6 or labels.count(0) == 6:

        if CATEGORIES[int(labels[-1])] == "PUB" and g_bloqueur == 1:
            t1 = time.time()
            g_bloqueur = 0
        if CATEGORIES[int(labels[-1])] != "PUB" and g_bloqueur == 0:
            t2 = time.time()
            tempsPub.append(t2-t1+5)
            g_bloqueur = 1
    
    else:
        pass


if __name__ == "__main__":
    init()





