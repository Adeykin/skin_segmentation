import numpy as np
import cv2


def peer2(img):
    b,g,r = cv2.split(img)
    
    ret,m1 = cv2.threshold(r, 220, 255, cv2.THRESH_BINARY)
    ret,m2 = cv2.threshold(g, 210, 255, cv2.THRESH_BINARY)
    ret,m3 = cv2.threshold(b, 170, 255, cv2.THRESH_BINARY)    
    ret,m4 = cv2.threshold(cv2.absdiff(r,g), 15, 255, cv2.THRESH_BINARY)
    m5 = cv2.compare(r,g, cv2.CMP_GT)
    m6 = cv2.compare(r,b, cv2.CMP_GE)
    
    mask = m1 & m2 & m3 & m4 & m5 & m6 
    
    return mask    

def peer(img):
    b,g,r = cv2.split(img)
    
    ret,m1 = cv2.threshold(r, 95, 255, cv2.THRESH_BINARY)
    ret,m2 = cv2.threshold(g, 40, 255, cv2.THRESH_BINARY)
    ret,m3 = cv2.threshold(b, 20, 255, cv2.THRESH_BINARY)
    mmax = cv2.max(r, cv2.max(g,b))
    mmin = cv2.min(r, cv2.min(g,b))
    ret,m4 = cv2.threshold(mmax-mmin, 15, 255, cv2.THRESH_BINARY)
    ret,m5 = cv2.threshold(cv2.absdiff(r,g), 15, 255, cv2.THRESH_BINARY)
    m6 = cv2.compare(r,g, cv2.CMP_GT)
    m7 = cv2.compare(r,b, cv2.CMP_GE)
    
    mask = m1 & m2 & m3 & m4 & m5 & m6 & m7
    
    return mask

def ndrplz(img):    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(img)
    ret,h1m = cv2.threshold(h, 5, 255, cv2.THRESH_BINARY)
    ret,h2m = cv2.threshold(h, 17, 255, cv2.THRESH_BINARY_INV)
    ret,s1m = cv2.threshold(s, 38, 255, cv2.THRESH_BINARY)
    ret,s2m = cv2.threshold(s, 250, 255, cv2.THRESH_BINARY_INV)
    ret,v1m = cv2.threshold(v, 51, 255, cv2.THRESH_BINARY)
    ret,v2m  = cv2.threshold(v, 242, 255, cv2.THRESH_BINARY_INV)
    
    mask = h1m & h2m & s1m & s2m & v1m & v2m
    
    return mask

def ahlbert(img):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    #img.convertTo(img, cv2.CV_32FC3)
    #img = np.float32(img)
    y,cb,cr = cv2.split(img)
    ret,cr1m = cv2.threshold(cr, 138, 255, cv2.THRESH_BINARY)
    ret,cr2m = cv2.threshold(cr, 178, 255, cv2.THRESH_BINARY_INV)
    cr = np.float32(cr)
    cb = np.float32(cb)
    sum = cb + 0.6*cr
    ret,sum1m = cv2.threshold(sum, 200, 255, cv2.THRESH_BINARY)
    ret,sum2m = cv2.threshold(sum, 215, 255, cv2.THRESH_BINARY_INV)
    sum1m = np.uint8(sum1m)
    sum2m = np.uint8(sum2m)
    
    
def sobottka(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(img)
    ret, h1m = cv2.threshold(h, 0,255, cv2.THRESH_BINARY)
    ret, h2m = cv2.threshold(h, 35,255, cv2.THRESH_BINARY_INV)
    ret, s1m = cv2.threshold(s, 58,255, cv2.THRESH_BINARY)
    ret, s2m = cv2.threshold(s, 173,255, cv2.THRESH_BINARY_INV)
    mask = h1m & h2m & s1m & s2m
    
    return mask
