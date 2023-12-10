class Point:
    def __init__(self, x, y):
        # float
        self.x = x
        # float
        self.y = y
    def __str__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"

class PointOnSightline:
    def __init__(self, point, elevation, distanceToStartingPoint):
        # Point
        self.point = point
        # float
        self.elevation = elevation
        # float
        self.distanceToStartingPoint = distanceToStartingPoint

    def __str__(self):
        return (("point: " + str(self.point) + "; "
                + "elevation: " + str(self.elevation) + "; ")
                + "distance to starting point: " + str(self.distanceToStartingPoint))

class Sightline:
    def __init__(self, pointA, pointB, distance):
        # Point
        self.pointA = pointA
        # Point
        self.pointB = pointB
        # float
        self.distance = distance

    def __str__(self):
        return ("starting at " + str(self.pointA)
                + ", ending at " + str(self.pointB)
                + ", spanning a distance of " + str(self.distance))

class ElevationFile:
    def __init__(self, numColumns, numRows, xCorner, yCorner, cellSize, data):
        # all int
        self.numColumns = numColumns
        self.numRows = numRows
        self.xCorner = xCorner
        self.yCorner = yCorner
        self.cellSize = cellSize
        # 2D numpy array
        self.data = data
