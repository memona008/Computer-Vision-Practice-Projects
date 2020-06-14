from transform import four_point_transform
import numpy as np
import argparse
import cv2

# # construct the argument parse and parse the arguments
def warp_image():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help = "path to the image file")
    ap.add_argument("-c", "--coords",
        help = "comma seperated list of source points")
    args = vars(ap.parse_args())

    # load the image and grab the source coordinates (i.e. the list of
    # of (x, y) points)
    # NOTE: using the 'eval' function is bad form, but for this example
    # let's just roll with it -- in future posts I'll show you how to
    # automatically determine the coordinates without pre-supplying them
    # image = cv2.resize(cv2.imread(args["image"]), (500,500))
    # pts = np.array(eval(args["coords"]), dtype = "float32")
    image = cv2.resize(cv2.imread('chess.jpg'), (500,500))
    pts = np.array([(149, 81), (480, 78), (353, 388), (22, 390)], dtype = "float32")
    # apply the four point tranform to obtain a "birds eye view" of
    # the image
    warped = four_point_transform(image, pts)
    # show the original and warped images
    cv2.imshow("Original", image)
    cv2.imshow("Warped", warped)
    cv2.waitKey(0)



def get_coords_by_click():
    pos_list = []
    def draw_circle(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(img,(x,y),1,(255,0,0),-1)
            print("X: {0} , Y: {1}".format(x,y))
            pos_list.append((x,y))


    img = cv2.resize(cv2.imread('chess.jpg'), (500,500))
    # img = np.zeros((512,512,3), np.uint8)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_circle)
    while(1):
        cv2.imshow('image',img)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
        elif k == ord('a'):
            print(pos_list)



warp_image()