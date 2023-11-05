# 1 unit of height is 1 metre, 1 unit of horizontal distance is 50 metres
import math
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
from scipy.signal import argrelextrema

EARTH_RADIUS = 6371000
BIG_NUMBER = 10**6

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __str__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"

class PointOnSightline:
    def __init__(self, point, elevation, distanceToStartingPoint):
        self.point = point
        self.elevation = elevation
        self.distanceToStartingPoint = distanceToStartingPoint

    def __str__(self):
        return (("point: " + str(self.point) + "; "
                + "elevation: " + str(self.elevation) + "; ")
                + "distance to starting point: " + str(self.distanceToStartingPoint))

class Sightline:
    def __init__(self, pointA, pointB, distance):
        self.pointA = pointA
        self.pointB = pointB
        self.distance = distance

    def __str__(self):
        return ("starting at " + str(self.pointA)
                + ", ending at " + str(self.pointB)
                + ", spanning a distance of " + str(self.distance))


def find_max_height_point():
    indexTuple = np.unravel_index(elevationData.argmax(), elevationData.shape)
    return Point(indexTuple[0], indexTuple[1])

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
    octant = math.floor(sightlineAngle / 45) + 1
    # sightlineIntercept = startingPoint.y - sightlineGradient * startingPoint.x
    # allDistances = np.full((numColumns, numRows), BIG_NUMBER)
    #
    # array of points closest to the sightline in order of increasing distance from startingPoint
    #
    pointsOnSightline = []
    # if angle is shallow, fill in distances to squares either side of crossings of vertical gridlines
    if octant in [2,3,6,7]:
        xIncrement = 1 if octant in [2,3] else -1
        yIncrement = sightlineGradient * (1 if octant in [2,3] else -1)
        xPosition = startingPoint.x + xIncrement
        yPosition = startingPoint.y + yIncrement
        upperQuadrant = octant in [2,7]
        while 0 <= xPosition <= numColumns-1 and 0 <= yPosition <= numRows-1:
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
    elif octant in [1,4,5,8]:
        xIncrement = 1 / sightlineGradient * (1 if octant in [1,8] else -1)
        yIncrement = 1 if octant in [1,8] else -1
        xPosition = startingPoint.x + xIncrement
        yPosition = startingPoint.y + yIncrement
        leftQuadrant = octant in [1,4]
        while 0 <= xPosition <= numColumns-1 and 0 <= yPosition <= numRows-1:
            xPositionMin = math.floor(xPosition)
            xPositionMax = math.ceil(xPosition)
            pointA = Point(xPositionMin, yPosition)
            pointB = Point(xPositionMax, yPosition)
            # allDistances[xPositionMin][yPosition] = distance_point_to_line(, sightlineGradient, sightlineIntercept)
            # allDistances[xPositionMax][yPosition] = distance_point_to_line(Point(xPositionMax, yPosition), sightlineGradient, sightlineIntercept)
            pointsOnSightline += [pointA, pointB] if leftQuadrant else [pointB, pointA]
            xPosition += xIncrement
            yPosition += yIncrement
    else:
        raise(Exception("Invalid angle provided:", sightlineAngle))
    # finalPoint = Point(xPosition, yPosition)
    # sightlineLengthToBoundary = distance_between_points(startingPoint, finalPoint)
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
def find_longest_sightline_in_direction(startingPoint, sightlineAngle):
    localMaximaAlongSightline = find_local_maxima_along_sightline(startingPoint, points_on_sightline(startingPoint, sightlineAngle, elevationFile['ncols'], elevationFile['nrows']))
    maxSightlineGradient = - BIG_NUMBER
    startingPointElevation = elevationData[startingPoint.x, startingPoint.y]
    if len(localMaximaAlongSightline) == 0:
        return None
    for pointOnSightline in localMaximaAlongSightline:
        sightlineGradient = (startingPointElevation - pointOnSightline.elevation) / pointOnSightline.distanceToStartingPoint
        if sightlineGradient > maxSightlineGradient:
            maxSightlineGradient = sightlineGradient
            maxPoint = pointOnSightline
    return Sightline(startingPoint, maxPoint.point, maxPoint.distanceToStartingPoint)

angleIncrement = 0.5

elevationFile = import_elevation_file('NY55.asc')
# interchange dimensions so that columns are 'x' and rows are 'y'
elevationData = np.flip(np.transpose(elevationFile["data"]), 1)
maxElevationPoint = find_max_height_point()

print("Starting at the highest point:", maxElevationPoint, " with elevation of", elevationData[maxElevationPoint.x,maxElevationPoint.y])
startingPoint = maxElevationPoint

maxSightlineDistance = 0 
for sightlineAngle in np.arange(180.1, 360, angleIncrement):
    sightline = find_longest_sightline_in_direction(startingPoint, sightlineAngle)
    if sightline and sightline.distance > maxSightlineDistance:
        maxSightlineDistance = sightline.distance
        maxSightline = sightline
        print("new max at angle", str(sightlineAngle), sightline)