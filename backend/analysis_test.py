import cv2
import numpy as np
import sys

p = r"C:\Users\dell5518\OneDrive\Desktop\proj\pro\backend\PlantVillage_Small\Potato___Late_blight\0c2628d4-8d64-48a9-a157-19a9c902e304___RS_LB 4590.JPG"
if len(sys.argv) > 1:
    p = sys.argv[1]

img = cv2.imread(p)
if img is None:
    print('ERROR: could not read', p)
    sys.exit(1)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# leaf mask
lower_leaf = np.array([25,40,20])
upper_leaf = np.array([85,255,255])
mask_leaf = cv2.inRange(hsv, lower_leaf, upper_leaf)
leaf_pixels = int(np.count_nonzero(mask_leaf))
leaf_area_percent = (leaf_pixels / (img.shape[0]*img.shape[1]))*100.0

# disease hsv (yellow/brown)
lower_disease = np.array([10,50,20])
upper_disease = np.array([35,255,255])
mask_disease = cv2.inRange(hsv, lower_disease, upper_disease)
mask_disease = cv2.bitwise_and(mask_disease, mask_leaf)
disease_pixels = int(np.count_nonzero(mask_disease))

severity_fraction = disease_pixels / leaf_pixels if leaf_pixels>0 else 0.0

# detect dark lesions
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
if leaf_pixels>0:
    vals = gray[mask_leaf>0]
    if vals.size>0:
        thresh = int(np.percentile(vals,40))
    else:
        thresh = int(np.percentile(gray.ravel(),40))
else:
    thresh = int(np.percentile(gray.ravel(),40))
dark_mask = (gray <= thresh).astype(np.uint8)*255
if mask_leaf is not None:
    dark_mask = cv2.bitwise_and(dark_mask, mask_leaf)
dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
dark_pixels = int(np.count_nonzero(dark_mask))

dark_severity_frac = dark_pixels / leaf_pixels if leaf_pixels>0 else 0.0

print('leaf_pixels', leaf_pixels)
print('leaf_area_percent', round(leaf_area_percent,2))
print('disease_pixels(hsv)', disease_pixels)
print('severity_fraction(hsv)', round(severity_fraction,4))
print('dark_pixels(grayscale fallback)', dark_pixels)
print('severity_fraction(dark)', round(dark_severity_frac,4))

# Save masks for inspection
cv2.imwrite('static/predictions/test_mask_leaf.png', mask_leaf)
cv2.imwrite('static/predictions/test_mask_disease.png', mask_disease)
cv2.imwrite('static/predictions/test_mask_dark.png', dark_mask)
print('Saved debug masks to static/predictions/')
