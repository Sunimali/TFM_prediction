import cv2
import numpy as np
# from matplotlib import pyplot as plt

cell = 'Cell_6'
data_dir = f"/home/local2/FTTC/Code/Traction_LSTM/Trial_4/{cell}/Cell1/cropCell200001.bmp.tif"

# Load image, grayscale, Otsu's threshold
image = cv2.imread(data_dir)
h, w, c = image.shape
thresh2 = np.zeros((h, w, 3), dtype=np.uint8)
thresh2.fill(255)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray,(1,1),0)
# new = cv2.addWeighted(gray,1.5,blur,-1,0)
kernel = np.array([[-1,-1,-1], [-1,7,-1], [-1,-1,-1]])
new = cv2.filter2D(gray, -1, kernel)
thresh = cv2.threshold(new, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# # Filter out large non-connecting objects
# cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# for c in cnts:
#     area = cv2.contourArea(c)
#     if area < 500:
#         cv2.drawContours(thresh,[c],0,0,-1)

# Morph open using elliptical shaped kernel
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)

# Find circles 
# cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# print(len(cnts))
# for c in cnts:
#     print("Area")
#     area = cv2.contourArea(c)
#     if area > 0.05 :
#         ((x, y), r) = cv2.minEnclosingCircle(c)
#         print(x, y, r)
#         cv2.circle(thresh, (int(x), int(y)), int(r), (36, 255, 12), 2)

detected_circles = cv2.HoughCircles(thresh,  
                   cv2.HOUGH_GRADIENT, 1, 2, param1 = 5, 
               param2 = 3, minRadius = 0, maxRadius = 6)

#Draw circles that are detected. 
if detected_circles is not None: 
  
    # Convert the circle parameters a, b and r to integers. 
    detected_circles = np.uint16(np.around(detected_circles)) 
  
    for pt in detected_circles[0, :]: 
        a, b, r = pt[0], pt[1], pt[2] 
  
        # Draw the circumference of the circle. 
        cv2.circle(thresh2, (a, b), 0, (0, 0, 0), -1) 
  
        # Draw a small circle (of radius 1) to show the center. 
        # cv2.circle(thresh, (a, b), 1, (0, 0, 255), 3) 
        # cv2.imshow("Detected Circle", img) 
        # cv2.waitKey(0) 
cv2.imshow('thresh_crop', thresh)
cv2.imshow('thresh_dots', thresh2)
cv2.imwrite(f"/home/local2/FTTC/Code/Traction_LSTM/Trial_4/{cell}/Cell1/thresh_initial.png", thresh)
cv2.imwrite(f"/home/local2/FTTC/Code/Traction_LSTM/Trial_4/{cell}/Cell1/thresh_dots_initial.png", thresh2)
# cv2.imshow('opening', opening)
cv2.imshow('image', image)
cv2.waitKey(0)