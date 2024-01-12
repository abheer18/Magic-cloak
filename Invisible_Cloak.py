
# '''
# Importing th necessary packages 
# '''
import cv2
import time
import numpy as np


cap = cv2.VideoCapture(0)  #this is used to continous video capture

# Store a single frame as background, before starting the infinite loop (starting video)
_, background = cap.read()
background = cv2.flip(background,1)
time.sleep(2)     #2-second delay between two captures are for adjusting camera auto exposure
_, background = cap.read()
background = cv2.flip(background,1)
#define all the kernels size  
open_kernel = np.ones((5,5),np.uint8)
close_kernel = np.ones((7,7),np.uint8)
dilation_kernel = np.ones((10, 10), np.uint8)

#initial function for the callin of the trackbar
def hello(x):
	#only for referece
	print("")


bars = cv2.namedWindow("bars")

cv2.createTrackbar("upper_hue","bars",110,180,hello)
cv2.createTrackbar("upper_saturation","bars",255, 255, hello)
cv2.createTrackbar("upper_value","bars",255, 255, hello)
cv2.createTrackbar("lower_hue","bars",68,180, hello)
cv2.createTrackbar("lower_saturation","bars",55, 255, hello)
cv2.createTrackbar("lower_value","bars",54, 255, hello)


# Function for remove noise from mask 
def filter_mask(mask):
    
    # Closing: Closing is an operation that removes small holes from the mask. This is done by first eroding the mask, and then dilating it. Erosion shrinks the mask by removing pixels on the edges, while dilation expands the mask by adding pixels on the edges.
    #format = cv2.morphologyEx(input image, type of morphology operation, kernel)
    close_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    
    # Opening: Opening is an operation that removes small bright spots from the mask. This is done by first dilating the mask, and then eroding it. Dilation expands the mask by adding pixels on the edges, while erosion shrinks the mask by removing pixels on the edges.
    open_mask = cv2.morphologyEx(close_mask, cv2.MORPH_OPEN, open_kernel)
    
    # Dilation: Dilation is an operation that expands the mask by adding pixels on the edges. This can be useful for filling in small gaps in the mask.
    dilation = cv2.dilate(open_mask, dilation_kernel, iterations= 1)

    return dilation

# cap.isOpen() function checks if the camera is open or not and returns true if the camera is open and false if the camera is not open.
while cap.isOpened():
    ret, frame = cap.read()  # Capture every frame
    frame  = cv2.flip(frame,1)    
    # convert to hsv colorspace 
    inspect = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    upper_hue = cv2.getTrackbarPos("upper_hue", "bars")
    upper_saturation = cv2.getTrackbarPos("upper_saturation", "bars")
    upper_value = cv2.getTrackbarPos("upper_value", "bars")
    lower_value = cv2.getTrackbarPos("lower_value","bars")
    lower_hue = cv2.getTrackbarPos("lower_hue","bars")
    lower_saturation = cv2.getTrackbarPos("lower_saturation","bars")

    # lower bound and upper bound for Green color  (starting of the green color and ending of the green color in the HSV color space)
    lower_bound = np.array([lower_hue,lower_saturation,lower_value])
    
    upper_bound = np.array([upper_hue,upper_saturation,upper_value])
    
    # find the colors within the boundaries, Creating a mask 
    # cv2.inRange() function returns a segmented binary mask of the frame where the green color is present
    mask = cv2.inRange(inspect, lower_bound, upper_bound)

    # Filter mask
    mask = filter_mask(mask)
    mask = cv2.medianBlur(mask,3)
    # Apply the mask to take only those region from the saved background 
    # where our cloak is present in the current frame
    # We have successfully detected the cloak. We want to show our previously-stored background in the main frame where the cloak is present. First, we need to take the only white region from the background.
    # Now in place of green color, the backgroud will appear in the place of green color
    #format - bitwise_and(source1, source2. mask)
    cloak = cv2.bitwise_and(background, background, mask=mask)

    # create inverse mask 
    # now , we have the visible background in the green color region but black color at the rest of the boundary
    #so now we will just invert the mask, so that whole backround will be show
    # cv2.bitwise_not() inverse the mask pixel value. Where the mask is white it returns black and where is black it returns white.
    inverse_mask = cv2.bitwise_not(mask)  

    # Apply the inverse mask to take those region of the current frame where cloak is not present 
    #now again we are joining the mask (invert mask and the cloak region)
    current_background = cv2.bitwise_and(frame, frame, mask=inverse_mask)

    # Combine cloak region and current_background region to get final frame 
    #this add two frame and return a single frame 
    combined = cv2.add(cloak, current_background)

    cv2.imshow("Final output", combined)


    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

