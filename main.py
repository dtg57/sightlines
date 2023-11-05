# 1 unit of height is 1 metre, 1 unit of horizontal distance is 50 metres
import math
import numpy as np
import sys

import finding_sightlines

np.set_printoptions(threshold=sys.maxsize)

from data_classes import Point
import data_processing
import plotting

EARTH_RADIUS = 6371000

def find_max_height_point():
    indexTuple = np.unravel_index(elevationData.argmax(), elevationData.shape)
    return Point(indexTuple[0], indexTuple[1])


#
# CONFIG
#
angleIncrement = 1
filename = "NY53.asc"



#
# BRAINS
#
elevationFile = data_processing.import_elevation_file(filename)
elevationData = elevationFile['data']
maxElevationPoint = find_max_height_point()
print("Starting at the highest point:", maxElevationPoint, " with elevation of", elevationData[maxElevationPoint.x,maxElevationPoint.y])
startingPoint = maxElevationPoint

maxSightline = finding_sightlines.find_longest_sightline_in_all_directions(elevationData, elevationFile['ncols'], elevationFile['nrows'], startingPoint, angleIncrement)

plotting.plot_region_3D_with_sightline(elevationData, maxSightline)