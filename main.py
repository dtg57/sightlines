# 1 unit of height is 1 metre, 1 unit of horizontal distance is 50 metres
#
# Longest photographed line of sight is Finestrelles, Pyrenees to Pic Gaspard, Alps - 443km - https://beyondrange.wordpress.com/2016/08/03/pic-de-finestrelles-pic-gaspard-ecrins-443-km/
# Longest line of sight in the British Isles is from Merrick to Snowdon - 232km
# Longest known line of sight is from Mount Dankova, Kyrgyzstan to Hingu Tagh, China at 538km
#
# Info here http://www.viewfinderpanoramas.org/panoramas.html#longlinesbrit
# And here https://beyondrange.wordpress.com/lines-of-sight/
# Horizon simulator http://www.peakfinder.org/
# Coordinate converter https://webapps.bgs.ac.uk/data/webservices/convertForm.cfm#bngToLatLng (multiply by 50)
#


import numpy as np
import sys
import math

np.set_printoptions(threshold=sys.maxsize)

import starting_points
import finding_sightlines
import data_processing
import plotting
import data_classes



#
# DATA IMPORT
#
elevationFile = data_processing.construct_uk()
elevationData = elevationFile.data


#
# CONFIG OPTIONS
#
CACHE_CURVATURE_STEP = 50
MIN_DISTANCE = 150 / 0.05
MAX_DISTANCE = 250 / 0.05
ANGLE_INCREMENT = 10 * 360 / (2 * math.pi * MAX_DISTANCE)
STARTING_POINT_MIN_ELEVATION = 900
STARTING_POINT_REGION_SIZE = 2000


#
# GENERATE CACHES
#
finding_sightlines.cache_curvature(CACHE_CURVATURE_STEP, MAX_DISTANCE)
finding_sightlines.cache_distance(MAX_DISTANCE)

#
# RUN
#
# Find viable starting points by splitting datatype into square regions of size STARTING_POINT_REGION_SIZE and finding
# the maximum in each region, returning it if it is larger than STARTING_POINT_MIN_ELEVATION
startingPoints = starting_points.find_starting_points(
    elevationData,
    STARTING_POINT_MIN_ELEVATION,
    STARTING_POINT_REGION_SIZE,
)
startingPointsString = ' '.join([str(startingPoint) for startingPoint in startingPoints])
print(len(startingPoints), "starting points found:", startingPointsString)

maxSightline = finding_sightlines.find_longest_sightline_from_many_starting_points(
    elevationData = elevationData,
    xLimit = elevationFile.numColumns,
    yLimit = elevationFile.numRows,
    minDistance = MIN_DISTANCE,
    maxDistance = MAX_DISTANCE,
    startingPoints = startingPoints,
    angleIncrement = ANGLE_INCREMENT
)

print("Overall longest sight line is", maxSightline)

plotting.plot_region_3D_with_sightline(elevationData, maxSightline, 25)