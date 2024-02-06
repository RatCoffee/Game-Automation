import cv2 as cv
import numpy as np
from PIL import ImageGrab
from ImageUtils import *


# LOAD SQUARE TILES
############################################################################
#Load square sprites from a single file
def loadSquares(name, squareSize):
    spriteSheet = readImage(name)
    h, w = spriteSheet[::squareSize,
                       ::squareSize, 0].shape
    sprites = [spriteSheet[i//w*squareSize:(1 + i//w)*squareSize,
                           i%w *squareSize:(1 + i%w) *squareSize]
               for i in range(w*h)]
    return sprites


# FIND GRID ORIGIN
############################################################################
#Origin from a landmark
def originFromLandmark(name, offset):
    landmark = readImage(name)
    screen = np.array(ImageGrab.grab())
    scan = cv.matchTemplate(screen, landmark, 3)
    maxLoc = cv.minMaxLoc(scan)[3]
    return tuple(sum(x) for x in zip(maxLoc, offset))

##def originFromGrid(squareSprites, squareSize):
##    screen = np.array(ImageGrab.grab())
##    scans = [cv.matchTemplate(screen, tile, 3) for tile in squareSprites]
##    return scans
##    maxPoints = [cv.minMaxLoc(scan)[2] for scan in scans]
##    maxVals = [cv.minMaxLoc(scan)[0] for scan in scans]
##    return maxPoints, maxVals


# PARSE GRID
############################################################################
#grid of a known size
def gridOfSize(tiles, origin, gridSize):
    squareSize = tiles[0].shape[0]
    bbox = (origin) + (origin[0] + squareSize * gridSize[0],
                       origin[1] + squareSize * gridSize[1])
    gridScan = np.array(ImageGrab.grab(bbox))
    # TODO: Maybe change this to fit the max value within the cell for each option?
    scans = np.array([cv.matchTemplate(gridScan, tile, 3)[::squareSize,::squareSize]
                      for tile in tiles])
    return np.argmax(scans, axis =0)

