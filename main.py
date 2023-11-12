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
# DATA IMPORT
#
elevationFile = data_processing.stitch_together_elevation_files("ny")
elevationData = elevationFile.data


#
# CONFIG
#
angleIncrement = 360 / (elevationFile.numColumns * 4)
fileName = "NY53.asc"


#
# RUN
#
maxElevationPoint = find_max_height_point()
print("Starting at the highest point:", maxElevationPoint, " with elevation of", elevationData[maxElevationPoint.x,maxElevationPoint.y])
startingPoint = maxElevationPoint

maxSightline = finding_sightlines.find_longest_sightline_in_all_directions(
    elevationData,
    elevationFile.numColumns,
    elevationFile.numRows,
    startingPoint,
    angleIncrement
)

plotting.plot_region_3D_with_sightline(elevationData, maxSightline, 20)