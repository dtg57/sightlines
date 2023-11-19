import numpy as np
import math
from scipy.signal import argrelextrema

from data_classes import Point, PointOnSightline, Sightline

BIG_NUMBER = 10**6
CACHE_CURVATURE = {}
CACHE_DISTANCE = {}
EARTH_RADIUS = 6371000


def cache_curvature(cacheCurvatureStep, minDistance, maxDistance):
    print("Caching curvature")
    global CACHE_CURVATURE
    global CACHE_CURVATURE_STEP
    CACHE_CURVATURE_STEP = cacheCurvatureStep
    CACHE_CURVATURE = {
        distance : EARTH_RADIUS * (1 - math.cos(distance * 50 / EARTH_RADIUS))
        for distance in np.arange(minDistance - cacheCurvatureStep, maxDistance + cacheCurvatureStep, cacheCurvatureStep)
    }

def cache_distance(minDistance, maxDistance):
    print("Caching distance")
    maxDistance += 5
    minDistance -= 5
    # Dictionary of dictionaries, to store all the possible Pythagorean distances between points.
    # The first key is the larger of the two diffs, second is the smaller (or equal)
    global CACHE_DISTANCE
    for xDiff in np.arange(0, maxDistance + 1):
        for yDiff in np.arange(xDiff, maxDistance + 1):
            distance = math.sqrt(xDiff**2 + yDiff**2)
            if distance >= minDistance:
                if distance <= maxDistance:
                    if yDiff in CACHE_DISTANCE:
                        CACHE_DISTANCE[yDiff][xDiff] = distance
                    else:
                        CACHE_DISTANCE[yDiff] = {xDiff : distance}
                else:
                    break
            else:
                continue


def distance_between_points(pointA, pointB):
    xDiff = abs(pointA.x - pointB.x)
    yDiff = abs(pointA.y - pointB.y)
    return CACHE_DISTANCE[xDiff][yDiff] if xDiff >= yDiff else CACHE_DISTANCE[yDiff][xDiff]


def elevation_curvature_correction(distance):
    roundedDistance = CACHE_CURVATURE_STEP * round(distance / CACHE_CURVATURE_STEP)
    return CACHE_CURVATURE[roundedDistance]

def points_on_sightline(startingPoint, sightlineAngle, xLimit, yLimit, minDistance, maxDistance):
    sightlineGradient = 1/math.tan(math.radians(sightlineAngle))
    octant = math.floor(sightlineAngle / 45) + 1
    #
    # array of points closest to the sightline in order of increasing distance from startingPoint
    #
    pointsOnSightline = []
    xPosition = math.floor(startingPoint.x + minDistance * math.sin(math.radians(sightlineAngle)))
    yPosition = math.floor(startingPoint.y + minDistance * math.cos(math.radians(sightlineAngle)))
    #
    # if angle is shallow, fill in distances to squares either side of crossings of vertical gridlines
    #
    if octant in [2,3,6,7]:
        xIncrement = 1 if octant in [2,3] else -1
        yIncrement = sightlineGradient * (1 if octant in [2,3] else -1)
        upperQuadrant = octant in [2,7]
        while 0 <= xPosition <= xLimit-1 and 0 <= yPosition <= yLimit-1:
            yPositionMin = math.floor(yPosition)
            yPositionMax = math.ceil(yPosition)
            pointA = Point(xPosition, yPositionMin)
            pointB = Point(xPosition, yPositionMax)
            distanceA = distance_between_points(startingPoint, pointA)
            distanceB = distance_between_points(startingPoint, pointB)
            if distanceA > maxDistance or distanceB > maxDistance:
                break
            pointOnSightlineA = PointOnSightline(pointA, ELEVATION_DATA[pointA.x, pointA.y], distanceA)
            pointOnSightlineB = PointOnSightline(pointB, ELEVATION_DATA[pointB.x, pointB.y], distanceB)
            pointsOnSightline += [pointOnSightlineA, pointOnSightlineB] if upperQuadrant else [pointOnSightlineB, pointOnSightlineA]
            xPosition += xIncrement
            yPosition += yIncrement
    #
    # if angle is steep, fill in distances to squares either side of crossings of horizontal gridlines
    #
    elif octant in [1,4,5,8]:
        xIncrement = 1 / sightlineGradient * (1 if octant in [1,8] else -1)
        yIncrement = 1 if octant in [1,8] else -1
        leftQuadrant = octant in [1,4]
        while 0 <= xPosition <= xLimit-1 and 0 <= yPosition <= yLimit-1:
            xPositionMin = math.floor(xPosition)
            xPositionMax = math.ceil(xPosition)
            pointA = Point(xPositionMin, yPosition)
            pointB = Point(xPositionMax, yPosition)
            distanceA = distance_between_points(startingPoint, pointA)
            distanceB = distance_between_points(startingPoint, pointB)
            if distanceA > maxDistance or distanceB > maxDistance:
                break
            pointOnSightlineA = PointOnSightline(pointA, ELEVATION_DATA[pointA.x, pointA.y], distanceA)
            pointOnSightlineB = PointOnSightline(pointB, ELEVATION_DATA[pointB.x, pointB.y], distanceB)
            pointsOnSightline += [pointOnSightlineA, pointOnSightlineB] if leftQuadrant else [pointOnSightlineB, pointOnSightlineA]
            xPosition += xIncrement
            yPosition += yIncrement
    else:
        raise(Exception("Invalid angle provided:", sightlineAngle))
    return pointsOnSightline

#
#  Returns an array of PointOnSightline objects, each of which contains the Point, elevation, and distance to the starting point
#
def find_local_maxima_along_sightline(pointsOnSightline):
    correctedElevationsOnSightline = []
    correctedPointsOnSightline = []
    for pointOnSightline in pointsOnSightline:
        correctedElevation = pointOnSightline.elevation - elevation_curvature_correction(pointOnSightline.distanceToStartingPoint)
        correctedPointsOnSightline.append(
            PointOnSightline(
                pointOnSightline.point,
                correctedElevation,
                pointOnSightline.distanceToStartingPoint
            )
        )
        correctedElevationsOnSightline.append(correctedElevation)
    localMaximaPositions = argrelextrema(np.array(correctedElevationsOnSightline), np.greater)
    return [correctedPointsOnSightline[index] for index in localMaximaPositions[0]]

#
# this works by finding the gradient (elevation difference / horizontal distance) of every theoretical sightline from startingPoint to each local maximum along the sightline
# the point that has the maximum signed gradient is at the end of the longest sightline in this direction
#
def find_longest_sightline_in_direction(xLimit, yLimit, minDistance, maxDistance, startingPoint, sightlineAngle):
    localMaximaAlongSightline = find_local_maxima_along_sightline(
        points_on_sightline(startingPoint, sightlineAngle, xLimit, yLimit, minDistance, maxDistance)
    )
    maxSightlineGradient = - BIG_NUMBER
    startingPointElevation = ELEVATION_DATA[startingPoint.x, startingPoint.y]
    if len(localMaximaAlongSightline) == 0:
        return None
    for pointOnSightline in localMaximaAlongSightline:
        sightlineGradient = (pointOnSightline.elevation - startingPointElevation) / pointOnSightline.distanceToStartingPoint
        if sightlineGradient > maxSightlineGradient:
            maxSightlineGradient = sightlineGradient
            maxPoint = pointOnSightline
    return Sightline(startingPoint, maxPoint.point, maxPoint.distanceToStartingPoint)

def find_longest_sightline_in_all_directions(xLimit, yLimit, minDistance, maxDistance, startingPoint, angleIncrement):
    maxSightline = Sightline(None, None, 0)
    for sightlineAngle in np.arange(angleIncrement, 360, angleIncrement):
        if (sightlineAngle - angleIncrement) % 90 > sightlineAngle % 90 or sightlineAngle == angleIncrement:
            print("...angles up to", math.floor(sightlineAngle) + 90, "...")
        # TODO: fix case when angle hits 0, 90, 180, 270 or 360 exactly
        sightline = find_longest_sightline_in_direction(xLimit, yLimit, minDistance, maxDistance, startingPoint, sightlineAngle)
        if sightline and sightline.distance > maxSightline.distance:
            maxSightline = sightline
    return maxSightline

def find_longest_sightline_from_many_starting_points(elevationData, xLimit, yLimit, minDistance, maxDistance, startingPoints, angleIncrement):
    global ELEVATION_DATA
    ELEVATION_DATA = elevationData
    maxSightline = Sightline(None, None, 0)
    for startingPoint in startingPoints:
        print("Starting at", startingPoint)
        sightline = find_longest_sightline_in_all_directions(xLimit, yLimit, minDistance, maxDistance, startingPoint, angleIncrement)
        print("Longest sightline from", str(startingPoint),
              "(elevation " + str(ELEVATION_DATA[startingPoint.x, startingPoint.y]) + ") is to",
              str(sightline.pointB), "(elevation " + str(ELEVATION_DATA[sightline.pointB.x, sightline.pointB.y]) + ")",
              "with a distance of", sightline.distance)
        if sightline.distance > maxSightline.distance:
            maxSightline = sightline
    return maxSightline
