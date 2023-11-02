# heights and horizontal distances are always given in metres
import math
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

EARTH_RADIUS = 6371000
BIG_DISTANCE = 10**6

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def find_max_height(region):
    return 10000

def find_max_sightline_distance(elevations):
    index = np.argmax(elevations)
    maxElevation = elevations[index]
    return {"index" : index, "elevation" : maxElevation}

def import_elevation_file(filename):
    with open(filename) as file:
        lines = file.readlines()
    output = {}
    data = np.ndarray(shape=(1,1))
    data = []
    for line in lines:
        splitLine = line.strip('\n').split(" ")
        if splitLine[0].isalpha():
            if len(splitLine) != 2:
                raise Exception("Error: unexpected metadata format:", line)
            output[splitLine[0]] = int(splitLine[1])
        else:
            data.append([float(datum) for datum in splitLine])
    output['data'] = data
    return output

def distance_point_to_line(point, lineGradient, lineIntercept):
    distance = abs((-lineGradient * point.x + point.y - lineIntercept) / math.sqrt(1 + lineGradient**2))
    print(point.x, point.y, distance,  lineGradient, lineIntercept)
    return distance

def distance_between_points(pointA, pointB):
    return math.sqrt((pointA.x - pointB.x)**2 + (pointA.y - pointB.y)**2)

def k_min_elements(ndArray, k):
    flattenedArray = np.flatten(ndArray)
    minIndices = np.argpartition(flattenedArray, k)


def distances_to_sightline(startingPoint, sightlineAngle, columns, rows):
    sightlineGradient = 1/math.tan(math.radians(sightlineAngle))
    sightlineIntercept = startingPoint.y - sightlineGradient * startingPoint.x
    distances = np.full((columns, rows), BIG_DISTANCE)
    # if angle is shallow, fill in distances to squares either side of crossings of vertical gridlines
    if 45 <= sightlineAngle <= 135 or 225 <= sightlineAngle <= 315:
        xPosition = startingPoint.x
        yPosition = startingPoint.y
        xIncrement = 1
        yIncrement = sightlineGradient
        while xPosition <= columns-1 and yPosition <= rows-1:
            yPositionMin = math.floor(yPosition)
            yPositionMax = math.ceil(yPosition)
            distances[xPosition][yPositionMin] = distance_point_to_line(Point(xPosition, yPositionMin), sightlineGradient, sightlineIntercept)
            distances[xPosition][yPositionMax] = distance_point_to_line(Point(xPosition, yPositionMax), sightlineGradient, sightlineIntercept)
            xPosition += xIncrement
            yPosition += yIncrement

    # if angle is steep, fill in distances to squares either side of crossings of horizontal gridlines
    else:
        xPosition = startingPoint.x
        yPosition = startingPoint.y
        xIncrement = 1/sightlineGradient
        yIncrement = 1
        while xPosition <= columns-1 and yPosition <= rows-1:
            xPositionMin = math.floor(xPosition)
            xPositionMax = math.ceil(xPosition)
            distances[xPositionMin][yPosition] = distance_point_to_line(Point(xPositionMin, yPosition), sightlineGradient, sightlineIntercept)
            distances[xPositionMax][yPosition] = distance_point_to_line(Point(xPositionMax, yPosition), sightlineGradient, sightlineIntercept)
            xPosition += xIncrement
            yPosition += yIncrement
    finalPoint = Point(xPosition, yPosition)
    sightlineLengthToBoundary = distance_between_points(startingPoint, finalPoint)
    return np.argpartition(distances, math.ceil(sightlineLengthToBoundary))

elevationData = import_elevation_file('NY55.asc')

print(distances_to_sightline(Point(100, 100), 45, elevationData['ncols'], elevationData['nrows']))

