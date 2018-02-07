import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('bbox-example-image.jpg')

# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy
    
    
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    w,h = img.shape[:2]
    # If x and/or y start/stop positions not defined, set to image size
    if not y_start_stop[0]:
        y_start_stop = (0,h)
    if not x_start_stop[0]:
        x_start_stop = (0,w)
    # Compute the span of the region to be searched
    y_span = y_start_stop[1] - y_start_stop[0]
    x_span = x_start_stop[1] - x_start_stop[0]
    # Compute the number of pixels per step in x/y
    y_pixels = int(xy_window[0]*(1 - xy_overlap[0]))
    x_pixels = int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    y_slides = int(y_span/y_pixels) - 1
    x_slides = int(x_span/x_pixels) - 1
    
    print(y_span, x_span)
    print(y_pixels, x_pixels)
    print(y_slides, x_slides)

    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    for y_slide in range(y_slides):
        for x_slide in range(x_slides):
            # Calculate each window position
            left_top = (y_slide*y_pixels, x_slide*x_pixels)
            bottom_right = (left_top[0]+xy_window[0], left_top[1]+xy_window[1])
            # Append window position to list
            window_list.append((left_top, bottom_right))
    # Return the list of windows
    return window_list

windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(128, 128), xy_overlap=(0.5, 0.5))
                       
window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6)                    
plt.imshow(window_img)