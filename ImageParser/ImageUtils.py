import cv2 as cv

# IMPORT IMAGE AS RGB INSTEAD OF BGR
############################################################################
def readImage(name):
    return cv.imread('%s.png'%name)[::1, ::1, ::-1]
