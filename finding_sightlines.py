import numpy as np
import math
from scipy.signal import argrelextrema

from data_classes import Point, PointOnSightline, Sightline

BIG_NUMBER = 10**6


def distance_between_points(pointA, pointB):
    return math.sqrt((pointA.x - pointB.x)**2 + (pointA.y - pointB.y)**2)

def points_on_sightline(startingPoint, sightlineAngle, xLimit, yLimit):
    sightlineGradient = 1/math.tan(math.radians(sightlineAngle))
    octant = math.floor(sightlineAngle / 45) + 1
    #
    # array of points closest to the sightline in order of increasing distance from startingPoint
    #
    pointsOnSightline = []
    #
    # if angle is shallow, fill in distances to squares either side of crossings of vertical gridlines
    #
    if octant in [2,3,6,7]:
        xIncrement = 1 if octant in [2,3] else -1
        yIncrement = sightlineGradient * (1 if octant in [2,3] else -1)
        xPosition = startingPoint.x + xIncrement
        yPosition = startingPoint.y + yIncrement
        upperQuadrant = octant in [2,7]
        while 0 <= xPosition <= xLimit-1 and 0 <= yPosition <= yLimit-1:
            yPositionMin = math.floor(yPosition)
            yPositionMax = math.ceil(yPosition)
            pointA = Point(xPosition, yPositionMin)
            pointB = Point(xPosition, yPositionMax)
            pointsOnSightline += [pointA, pointB] if upperQuadrant else [pointB, pointA]
            xPosition += xIncrement
            yPosition += yIncrement
    #
    # if angle is steep, fill in distances to squares either side of crossings of horizontal gridlines
    #
    elif octant in [1,4,5,8]:
        xIncrement = 1 / sightlineGradient * (1 if octant in [1,8] else -1)
        yIncrement = 1 if octant in [1,8] else -1
        xPosition = startingPoint.x + xIncrement
        yPosition = startingPoint.y + yIncrement
        leftQuadrant = octant in [1,4]
        while 0 <= xPosition <= xLimit-1 and 0 <= yPosition <= yLimit-1:
            xPositionMin = math.floor(xPosition)
            xPositionMax = math.ceil(xPosition)
            pointA = Point(xPositionMin, yPosition)
            pointB = Point(xPositionMax, yPosition)
            pointsOnSightline += [pointA, pointB] if leftQuadrant else [pointB, pointA]
            xPosition += xIncrement
            yPosition += yIncrement
    else:
        raise(Exception("Invalid angle provided:", sightlineAngle))
    return pointsOnSightline

#
#  Returns an array of PointOnSightline objects, each of which contains the Point, elevation, and distance to the starting point
#
def find_local_maxima_along_sightline(startingPoint, pointsOnSightline):
    elevationsOnSightline = np.array([elevationData[point.x, point.y] for point in pointsOnSightline])
    localMaximaPositions = argrelextrema(elevationsOnSightline, np.greater)
    return [
        PointOnSightline(
            pointsOnSightline[index],
            elevationsOnSightline[index],
            distance_between_points(startingPoint, pointsOnSightline[index])
        ) for index in localMaximaPositions[0]
    ]

#
# this works by finding the gradient (elevation difference / horizontal distance) of every theoretical sightline from startingPoint to each local maximum along the sightline
# the point that has the maximum signed gradient is at the end of the longest sightline in this direction
#
def find_longest_sightline_in_direction(xLimit, yLimit, startingPoint, sightlineAngle):
    localMaximaAlongSightline = find_local_maxima_along_sightline(startingPoint, points_on_sightline(startingPoint, sightlineAngle, xLimit, yLimit))
    maxSightlineGradient = - BIG_NUMBER
    startingPointElevation = elevationData[startingPoint.x, startingPoint.y]
    if len(localMaximaAlongSightline) == 0:
        return None
    for pointOnSightline in localMaximaAlongSightline:
        sightlineGradient = (pointOnSightline.elevation - startingPointElevation) / pointOnSightline.distanceToStartingPoint
        if sightlineGradient > maxSightlineGradient:
            maxSightlineGradient = sightlineGradient
            maxPoint = pointOnSightline
    return Sightline(startingPoint, maxPoint.point, maxPoint.distanceToStartingPoint)

def find_longest_sightline_in_all_directions(elevationDataTemp, xLimit, yLimit, startingPoint, angleIncrement):
    global elevationData
    elevationData = elevationDataTemp
    maxSightlineDistance = 0
    for sightlineAngle in np.arange(angleIncrement, 360, angleIncrement):
        # TODO: fix case when angle hits 0, 90, 180, 270 or 360 exactly
        if math.floor(sightlineAngle - angleIncrement) != math.floor(sightlineAngle):
            print("Angle", math.floor(sightlineAngle))
        sightline = find_longest_sightline_in_direction(xLimit, yLimit, startingPoint, sightlineAngle)
        if sightline and sightline.distance > maxSightlineDistance:
            maxSightlineDistance = sightline.distance
            maxSightline = sightline
            print("new max at angle", str(sightlineAngle), sightline)
    return maxSightline