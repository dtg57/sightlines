import numpy as np
import math

from data_classes import Point


def find_starting_points(elevationData, minElevation, regionSize):
    startingPoints = []
    xLimit, yLimit = elevationData.shape
    xRegions, yRegions = math.ceil(xLimit / regionSize), math.ceil(yLimit / regionSize)
    for x in range(xRegions):
        for y in range(yRegions):
            if x == xRegions - 1 or y == yRegions - 1:
                region = elevationData[x * regionSize : , y * regionSize :]
            else:
                region = elevationData[x * regionSize : (x + 1) * regionSize, y * regionSize : (y + 1) * regionSize]
            maxPointRegionX, maxPointRegionY = np.unravel_index(region.argmax(), region.shape)
            maxPointX, maxPointY = maxPointRegionX + x * regionSize, maxPointRegionY + y * regionSize
            maxPointElevation = elevationData[maxPointX, maxPointY]
            if maxPointElevation > minElevation:
                startingPoints.append(Point(maxPointX, maxPointY))
    return startingPoints
