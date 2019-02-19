# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 13:25:54 2019
@name: delete_background
@description: delete the background of nike images
@author: vanbr
"""

import os
os.chdir('C:\\Users\\vanbr\\Documents\\Github\\AIAIAI\\Nike GAN')
os.getcwd()

import cv2
import numpy as np

import glob
from pathlib import Path

from matplotlib import pyplot as plt

from PIL import Image
from resizeimage import resizeimage


#== Parameters =======================================================================
BLUR = 25
CANNY_THRESH_1 = 20
CANNY_THRESH_2 = 250
MASK_DILATE_ITER = 2
MASK_ERODE_ITER = 2
MASK_COLOR = (0.0,0.0,1.0) # In BGR format

sigma=0.4

#== Processing =======================================================================

nike_dir = os.path.join(os.getcwd(), 'data_nikes')

# Retrieve all image names in folder
nike_images = glob.glob(nike_dir + '\\originals\\*.png')
filename = nike_images[1]

def deleteBackground (filename, show_image=False):
    #-- Read image -----------------------------------------------------------------------
    img = cv2.imread(filename)
    
    
    # Resize image and store
    img = cv2.resize(img,(128,128))
    new_filename = Path(os.path.basename(filename)).stem + ".png"
    cv2.imwrite(nike_dir + '\\resized\\' + new_filename, img)
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Apply auto threshold selection for edges
    #v = np.median(gray)
    #CANNY_THRESH_1 = int(max(0, (1.0 - sigma) * v))
    #CANNY_THRESH_2 = int(min(255, (1.0 + sigma) * v))
    
    #-- Edge detection -------------------------------------------------------------------
    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2, True)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)
    
    #-- Find contours in edges, sort by area ---------------------------------------------
    contour_info = []
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Previously, for a previous version of cv2, this line was: 
    #  contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Thanks to notes from commenters, I've updated the code but left this note
    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]
    
    #-- Create empty mask, draw filled polygon on it corresponding to largest contour ----
    # Mask is black, polygon is white
    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour[0], (255))
    
    #-- Smooth mask, then blur it --------------------------------------------------------
    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask_preblur = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask_preblur, (BLUR, BLUR), 0)
    mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask
    
    #-- Blend masked img into MASK_COLOR background --------------------------------------
    mask_stack  = mask_stack.astype('float32') / 255.0          # Use float matrices, 
    img         = img.astype('float32') / 255.0                 #  for easy blending
    
    masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR) # Blend
    masked = (masked * 255).astype('uint8')                     # Convert back to 8-bit 
    
    if (show_image == True):
        cv2.imshow('img', masked)                                   # Display
        cv2.waitKey()
    
    # split image into channels
    c_red, c_green, c_blue = cv2.split(img)
    
    # merge with mask got on one of a previous steps
    img_a = cv2.merge((c_red, c_green, c_blue, mask.astype('float32') / 255.0))
    
    new_filename = Path(os.path.basename(filename)).stem + ".png"
    
    ## Save to disk
    # Store image with dropped background in .png
    cv2.imwrite(nike_dir + '\\no_background\\' + new_filename, img_a*255)
    # Store edge detected image
    cv2.imwrite(nike_dir + '\\edges\\' + new_filename, edges)
    # Store shape
    #mask_preblur = mask_preblur*-1 # Invert to make black shape
    cv2.imwrite(nike_dir + '\\shape\\' + new_filename, 255-mask_preblur)
    # Store red backrgounded version also, for quality control
    # cv2.imwrite(nike_dir + '\\no_background\\' + 'red_' + new_filename, masked)           # Save
    
#-- Loop over all images in folder 
for img in nike_images:
    deleteBackground(filename = img)
    
    


        
plt.subplot(122),plt.imshow(edges,cmap = 'gray')

plt.imshow(mask_stack,cmap = 'gray')
plt.imshow(mask_preblur,cmap = 'gray')

plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()

filename = "mysequence.fasta"


filename = img
show_image = False
    



