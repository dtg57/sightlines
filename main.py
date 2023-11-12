# 1 unit of height is 1 metre, 1 unit of horizontal distance is 50 metres
import math
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

import finding_sightlines
from data_classes import Point
import data_processing
import plotting


def find_max_height_point():
    indexTuple = np.unravel_index(elevationData.argmax(), elevationData.shape)
    return Point(indexTuple[0], indexTuple[1])


#
# DATA IMPORT
#
elevationFile = data_processing.construct_uk()
elevationData = elevationFile.data


#
# CONFIG OPTIONS
#
ANGLE_INCREMENT = 5 * 360 / (elevationFile.numColumns * 4)
CACHE_CURVATURE_STEP = 10
# TODO: restrict distances along sightlines too
MAX_DISTANCE = 1000000


#
# GENERATE CACHES
#
# TODO: cache distances
finding_sightlines.cache_curvature(CACHE_CURVATURE_STEP, MAX_DISTANCE)


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
    ANGLE_INCREMENT
)

plotting.plot_region_3D_with_sightline(elevationData, maxSightline, 25)