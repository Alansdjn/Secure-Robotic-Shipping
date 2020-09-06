# python 3.7
import cv2
from imutils.object_detection import non_max_suppression
import numpy as np

# Camera capture
commands='raspistill -v -o '+'capture'+'.jpg'
os.system(commands)
img=cv2.imread("./capture.jpg")
orig = img.copy() 

# Detect persons and resize their images
defaultHog=cv2.HOGDescriptor()# Define HOG target
defaultHog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())# Define SVN classifier
(rects, weights) = defaultHog.detectMultiScale(img, winStride=(4, 4),padding=(8, 8), scale=1.05)
rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
for (xA, yA, xB, yB) in pick:
    cv2.rectangle(img, (xA, yA), (xB, yB), (0, 255, 0), 2)
cropImg = img[yA:yB, xA:xB]
cv2.imwrite("./data/reid_robot/000.jpg", cropImg)
print('Image Captured')




