# 1 unit of height is 1 metre, 1 unit of horizontal distance is 50 metres
#
# Longest photographed line of sight is Finestrelles, Pyrenees to Pic Gaspard, Alps - 443km - https://beyondrange.wordpress.com/2016/08/03/pic-de-finestrelles-pic-gaspard-ecrins-443-km/
# Longest line of sight in the British Isles is from Merrick to Snowdon - 232km
# Longest known line of sight is from Mount Dankova, Kyrgyzstan to Hingu Tagh, China at 538km
#
# Info here http://www.viewfinderpanoramas.org/panoramas.html#longlinesbrit
# And here https://beyondrange.wordpress.com/lines-of-sight/
# Horizon simulator http://www.peakfinder.org/
#



import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import math

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
CACHE_CURVATURE_STEP = 20
MIN_DISTANCE = 100 / 0.05
MAX_DISTANCE = 250 / 0.05
ANGLE_INCREMENT = 5 * 360 / (2 * math.pi * MAX_DISTANCE)


#
# GENERATE CACHES
#
# TODO: cache distances
finding_sightlines.cache_curvature(CACHE_CURVATURE_STEP,  MIN_DISTANCE, MAX_DISTANCE)
finding_sightlines.cache_distance(MIN_DISTANCE, MAX_DISTANCE)

#
# RUN
#
maxElevationPoint = find_max_height_point()
print("Starting at the highest point:", maxElevationPoint, "with elevation of", elevationData[maxElevationPoint.x,maxElevationPoint.y])
startingPoint = maxElevationPoint

maxSightline = finding_sightlines.find_longest_sightline_in_all_directions(
    elevationData = elevationData,
    xLimit = elevationFile.numColumns,
    yLimit = elevationFile.numRows,
    minDistance = MIN_DISTANCE,
    maxDistance = MAX_DISTANCE,
    startingPoint = startingPoint,
    angleIncrement = ANGLE_INCREMENT
)

plotting.plot_region_3D_with_sightline(elevationData, maxSightline, 25)