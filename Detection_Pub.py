import cv2
import numpy as np
import os
from PIL import ImageGrab,ImageFilter
import time 

from ctypes import windll
user32 = windll.user32
user32.SetProcessDPIAware()

def localisation(img):

    image_rect = cv2.pyrDown(img)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(image_rect, cv2.MORPH_GRADIENT, kernel)

    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    mask = np.zeros(bw.shape, dtype=np.uint8)

    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        mask[y:y+h, x:x+w] = 0
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

        if r > 0.45 and w > 8 and h > 8:
            cv2.rectangle(image_rect, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)

    return cv2.imshow('rects', image_rect)

def ROI(img, sommet):
    masque = np.zeros_like(img)
    cv2.fillPoly(masque,sommet, 255)
    img_masquee = cv2.bitwise_and(img,masque)
    return img_masquee

def process_image(image_ori, cmpt):

    processed_img = image_ori
    L = processed_img.shape[1]
    H = processed_img.shape[0]
    HDL = int((L*5)/6)
    HDH = int(H/8)

    sommet = np.array([[L,H/25], [L,HDH], [HDL,HDH], [HDL,H/25]], np.int32)
    new_img = ROI(processed_img,[sommet])

    img = ImageGrab.grab(bbox=(1594, 41, 1902 , 137))
    FILES_DIR = 'C:/Users/Theo/Documents/GITHUB/Detect_PUB/Photos/logo'
    SAVE_PATH = "C:/Users/Theo/Documents/GITHUB/Detect_PUB/Photos/logo"
    #SAVE_PATH = os.path.expanduser("~")    #It is cross-platform
    LOGFILE_NAME = "france"+str(cmpt)+".png"
    LOGFILE_PATH = os.path.join(SAVE_PATH, FILES_DIR, LOGFILE_NAME)
    img.save(LOGFILE_PATH)

    localisation(new_img)

    return new_img

# last_time = time.time()
cmpt = 778
while(True):
    ecran = np.array(ImageGrab.grab().filter(ImageFilter.SHARPEN)) #bbox=(0,40, 900, 800)

    ecran_gris = cv2.cvtColor((ecran), cv2.COLOR_BGR2RGB)
    ecran_gris = cv2.cvtColor((ecran), cv2.COLOR_BGR2GRAY)

    new_ecran = process_image(ecran_gris,cmpt)
    cmpt +=1
    # cv2.imshow('window1',cv2.cvtColor((new_ecran), cv2.COLOR_BGR2RGB))
    # cv2.imshow('window2',cv2.cvtColor((ecran), cv2.COLOR_BGR2RGB))


    if cv2.waitKey(25)& 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
