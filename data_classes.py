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