import cv2
import sys
import seg
import csv

if __name__ == '__main__':
    
    reader = csv.reader( open(sys.argv[1], 'rt'), delimiter = ';' )
    for row in reader:
        img = cv2.imread(row[1])
        
        mask = seg.ndrplz(img)
        img = cv2.bitwise_and(img,img, mask=mask)
    
        """
        cv2.namedWindow("windows")
        cv2.imshow("windows", img)
        if cv2.waitKey(0) == 'q':
            break
        """
        
        cv2.imwrite( sys.argv[2] + "/" + row[1], img)