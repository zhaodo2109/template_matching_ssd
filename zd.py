from zdlib import *
from tkinter import Tk
import cv2
import numpy as np
from tkinter import filedialog, simpledialog

from tkinter.filedialog import askopenfilename

import time

print("OpenCV Version:", cv2.__version__)
print("Numpy Version: ", np.__version__)

Tk().withdraw()
filename = filedialog.askopenfilename(title="Select an Image") # open img file

image = cv2.imread(filename, 0)
color_image = cv2.imread(filename, 1)
cv2.imshow("Original Image", color_image) #show original image

height, width = image.shape #get img dimension
height1, width1,channels = color_image.shape #get img dimension



sigma = simpledialog.askfloat("Gaussian Sigma", "Enter the sigma value:")
start_time = time.time()

g,half_w = Gaussian(sigma)
g_der = Gaussian_Deriv(sigma)

# Horizontal Gradient by smoothing with Gaussian vertically first then convolve the result with
# Gaussian Derivative kernel
gx=ConvolveSeparable(image,g_der,g,half_w)
cv2.imshow("Horizontal Gradient Template", gx)

# Vertical Gradient by smoothing with Gaussian horizontally first then convolve the result with
# Gaussian Derivative kernel
gy=ConvolveSeparable(image,g,g_der,half_w) #Vertical Gradient
cv2.imshow("Vertical Gradient Template", gy)

G,angle = MagnitudeGradient(gx,gy)
cv2.imshow("Magnitude Gradient", G.astype(np.uint8))
cv2.imshow("Gradient Image",angle)

#Suppression
suppression_orig_image = NonMaxSuppression(G.astype(np.uint8),angle)
cv2.imshow("Suppressed Image", suppression_orig_image)

#edgelinking
edges_img = Hysteresis(suppression_orig_image.astype(np.uint8))
cv2.imshow("Edge-Linking Image", edges_img)

#chamfer distance
chamfer_orig = chamfer_distance(edges_img)
cv2.imshow("Chamfer Original Image", chamfer_orig)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.4f} seconds")

cv2.waitKey(0)
cv2.destroyAllWindows() # close all image windows

''' Template '''
file_path = askopenfilename(title="Select a template")

image2 = cv2.imread(file_path, 0)
color_image2 = cv2.imread(file_path, 1)
cv2.imshow("Original Image", color_image)
cv2.imshow("Template", color_image2)

out_img = color_image.copy()
height_T, width_T = image2.shape #get template dimension

gx_T=ConvolveSeparable(image2,g_der,g,half_w) #Template Horizontal Gradient
gy_T=ConvolveSeparable(image2,g,g_der,half_w) #Template Vertical Gradient

G_T,angle_T = MagnitudeGradient(gx_T,gy_T)

#Suppression
suppression_T = NonMaxSuppression(G_T.astype(np.uint8),angle_T)

#edgelinking
edges_T = Hysteresis(suppression_T)

#chamfer
chamfer_temp = chamfer_distance(edges_T)
cv2.imshow("Chamfer Template Image", chamfer_temp)
cv2.imshow("Original Chamfer Image", chamfer_orig)

'''template matching'''

start_time = time.time()

position = template_matching(chamfer_temp, chamfer_orig)
border = get_border(chamfer_temp,position)
# Border is an array of (y, x) tuples
# Traverse through the border and change the color to green when found match.
for (ty, tx) in border:
    if 0 <= ty < out_img.shape[0] and 0 <= tx < out_img.shape[1]:  # Check bounds
        out_img[ty, tx] = (0, 255, 0)  # Change color
cv2.imshow(" Out Image", out_img)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.4f} seconds")

cv2.waitKey(0)
cv2.destroyAllWindows() # close all image windows