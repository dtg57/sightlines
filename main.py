# 1 unit of height is 1 metre, 1 unit of horizontal distance is 50 metres
import math
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
from scipy.signal import argrelextrema

EARTH_RADIUS = 6371000
BIG_DISTANCE = 10**6

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class PointOnSightline:
    def __init__(self, point, elevation, distanceToStartingPoint):
        self.point = point
        self.elevation = elevation
        self.distanceToStartingPoint = distanceToStartingPoint

def find_max_height(region):
    return 10000

def import_elevation_file(filename):
    with open(filename) as file:
        lines = file.readlines()
    output = {}
    lineNum = 0
    for line in lines:
        splitLine = line.strip('\n').split(" ")
        if splitLine[0].isalpha():
            if len(splitLine) != 2:
                raise Exception("Error: unexpected metadata format:", line)
            output[splitLine[0]] = int(splitLine[1])
        else:
            if lineNum == 0:
                data = np.ndarray(shape = (output['nrows'], output['ncols']))
            line = [float(datum) for datum in splitLine]
            data[lineNum,:] = np.array(line)
            lineNum += 1
    output['data'] = data
    return output

def distance_between_points(pointA, pointB):
    return math.sqrt((pointA.x - pointB.x)**2 + (pointA.y - pointB.y)**2)

def points_on_sightline(startingPoint, sightlineAngle, numColumns, numRows):
    sightlineGradient = 1/math.tan(math.radians(sightlineAngle))
    quadrant = math.floor(sightlineAngle / 90) + 1
    # sightlineIntercept = startingPoint.y - sightlineGradient * startingPoint.x
    # allDistances = np.full((numColumns, numRows), BIG_DISTANCE)
    #
    # array of points closest to the sightline in order of increasing distance from startingPoint
    #
    pointsOnSightline = []
    # if angle is shallow, fill in distances to squares either side of crossings of vertical gridlines
    if 45 <= sightlineAngle <= 135 or 225 <= sightlineAngle <= 315:
        xPosition = startingPoint.x
        yPosition = startingPoint.y
        xIncrement = 1
        yIncrement = sightlineGradient
        upperQuadrant = quadrant in [1,4]
        while xPosition <= numColumns-1 and yPosition <= numRows-1:
            yPositionMin = math.floor(yPosition)
            yPositionMax = math.ceil(yPosition)
            pointA = Point(xPosition, yPositionMin)
            pointB = Point(xPosition, yPositionMax)
            # allDistances[xPosition][yPositionMin] = distance_point_to_line, sightlineGradient, sightlineIntercept)
            # allDistances[xPosition][yPositionMax] = distance_point_to_line(, sightlineGradient, sightlineIntercept)
            pointsOnSightline += [pointA, pointB] if upperQuadrant else [pointB, pointA]
            xPosition += xIncrement
            yPosition += yIncrement
    #
    # if angle is steep, fill in distances to squares either side of crossings of horizontal gridlines
    #
    else:
        xPosition = startingPoint.x
        yPosition = startingPoint.y
        xIncrement = 1/sightlineGradient
        yIncrement = 1
        leftQuadrant = quadrant in [1,2]
        while xPosition <= numColumns-1 and yPosition <= numRows-1:
            xPositionMin = math.floor(xPosition)
            xPositionMax = math.ceil(xPosition)
            pointA = Point(xPositionMin, yPosition)
            pointB = Point(xPositionMax, yPosition)
            # allDistances[xPositionMin][yPosition] = distance_point_to_line(, sightlineGradient, sightlineIntercept)
            # allDistances[xPositionMax][yPosition] = distance_point_to_line(Point(xPositionMax, yPosition), sightlineGradient, sightlineIntercept)
            pointsOnSightline += [pointA, pointB] if leftQuadrant else [pointB, pointA]
            xPosition += xIncrement
            yPosition += yIncrement
    # finalPoint = Point(xPosition, yPosition)
    # sightlineLengthToBoundary = distance_between_points(startingPoint, finalPoint)
    return pointsOnSightline

def find_local_maxima_along_sightline_and_convert(startingPoint, pointsOnSightLine):
    elevationsOnSightLine = np.array([elevationData[point.x, point.y] for point in pointsOnSightLine])
    print(elevationsOnSightLine)
    localMaximaPositions = argrelextrema(elevationsOnSightLine, np.greater)
    print(localMaximaPositions)
    return [
        PointOnSightline(
            pointsOnSightLine[index],
            elevationsOnSightLine[index],
            distance_between_points(startingPoint, pointsOnSightLine[index])
        ) for index in localMaximaPositions[0]
    ]


elevationFile = import_elevation_file('NY55.asc')
elevationData = elevationFile["data"]

startingPoint = Point(100, 100)
print(elevationData)
print(find_local_maxima_along_sightline_and_convert(startingPoint, points_on_sightline(startingPoint, 40, elevationFile['ncols'], elevationFile['nrows'])))
