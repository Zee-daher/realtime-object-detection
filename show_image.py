# show_image.py — see the image as BOTH numbers and a picture

import cv2
import os

image_files = [f for f in os.listdir(".") if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
filename = image_files[0]
image = cv2.imread(filename)

print("Found image:", filename)
print("Shape (height, width, channels):", image.shape)

# --- the image IS a matrix of numbers — let's prove it ---
print("\nType of 'image':", type(image))          # it's a NumPy array (a matrix)
print("\nTop-left corner, first 3x3 pixels (as numbers):")
print(image[0:3, 0:3])                             # print a tiny 3x3 patch of raw numbers

# each pixel = 3 numbers [Blue, Green, Red], each from 0 to 255
print("\nThe single top-left pixel:", image[0, 0])

# now also show it as a picture
cv2.imshow("Picture view", image)
cv2.waitKey(0)
cv2.destroyAllWindows()