import numpy as np
import cv2

drawing = False # true if mouse is pressed
ix,iy = -1,-1

# mouse callback function
def draw_rectangle(event, x, y, flags, param):
    """ Draw rectangle on mouse click and drag """
    global ix,iy,drawing,mode
    # if the left mouse button was clicked, record the starting and set the drawing flag to True
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    # mouse is being moved, draw rectangle
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:       
            pass     
            
                      
            # cv2.rectangle(img, (ix, iy), (x, y), (255, 255, 0), -1)
    # if the left mouse button was released, set the drawing flag to False
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img,(ix,iy),(ix,y),  (255, 255, 0))
        cv2.line(img,(ix,y),(x,y),  (255, 255, 0))
        cv2.line(img,(ix,iy),(x,iy),  (255, 255, 0))
        cv2.line(img,(x,iy),(x,y),  (255, 255, 0))
        print(ix,iy,x,y)

# create a black image (height=360px, width=512px), a window and bind the function to window
img = cv2.imread("parrots.PNG")
cv2.namedWindow('image') 
cv2.setMouseCallback('image',draw_rectangle)

while(1):
    cv2.imshow('image',img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
    
cv2.destroyAllWindows()
