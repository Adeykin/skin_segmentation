import numpy as np
import cv2
import seg

cap = cv2.VideoCapture("/home/adeykin/projects/gestures/M2U00002.MPG")

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #frame = cv2.GaussianBlur(frame, (5,5), 10)
    
    mask = seg.ahlbert(frame)
    frame = cv2.bitwise_and(frame,frame, mask=mask)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()