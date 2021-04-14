"""All imports necessary to carry out this process"""
import cv2
import numpy as np
import os
from PIL import ImageGrab,ImageFilter
import time 

"""Cv2 and image grab don't take the whole screen"""
from ctypes import windll
user32 = windll.user32
user32.SetProcessDPIAware()

import repeatedTime
import keylog

#here we have our global variables
g_klog = None
g_repscreen = None
cmpt = 102 
sec = 0

def init():
    """This function instantiates all our global variables."""
    global g_klog, g_repscreen
    g_repscreen = repeatedTime.RepeatedTimer(1,process_image)
    g_klog = keylog.KeyLogger()
    startAll()

def startAll():
    """Starts listening to the keyboard and recording the screen."""
    global g_klog, g_repscreen
    g_repscreen.start()
    g_klog.start()


def stopAll():
    """Stops all processes."""
    global g_repscreen, g_klog
    g_repscreen.stop()
    g_klog.stop()


def process_image():
    """This function takes a screenshot of the top right corner (because french tv put the logo here)"""
    global cmpt, sec

    if sec == 2 :

    	if (g_klog.a_stopMain):
        	cmpt = cmpt+1
        	img = ImageGrab.grab(bbox=(0, 41, 308, 137))

        	os_path = os.getcwd()+"\Photos\logo"
        	FILES_DIR = 'C:/Users/TLG/Desktop/Captures'
        	SAVE_PATH = 'C:/Users/TLG/Desktop/Captures'
        	#SAVE_PATH = os.path.expanduser("~")    #It is cross-platform
        	LOGFILE_NAME = "BFM_"+str(cmpt)+".png"

        	LOGFILE_PATH = os.path.join(SAVE_PATH, FILES_DIR, LOGFILE_NAME)
        	img.save(LOGFILE_PATH)
        	sec = 0
    
    	else:
        	stopAll()

    else :
    	sec += 1


if __name__ == "__main__":
    init()
