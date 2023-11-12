import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cbook, cm
from matplotlib.colors import LightSource


def plot_region_3D_with_sightline(elevationData, sightline = None, step = 1):
    elevationDataAbridged = elevationData[::step, ::step]
    numRows, numColumns = elevationDataAbridged.shape
    print("Plotting surface with " + str(numRows) + " * " + str(numColumns) + " = " + str(numRows * numColumns) + " points")

    x = np.linspace(0, numColumns * step - 1, numColumns)
    y = np.linspace(0, numRows * step - 1, numRows)
    x, y = np.meshgrid(x, y)
    figure, axes = plt.subplots(subplot_kw=dict(projection='3d'))
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    lightsource = LightSource(270, 45)
    rgb = lightsource.shade(elevationDataAbridged, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    surf = axes.plot_surface(y, x, elevationDataAbridged, rstride=1, cstride=1, facecolors=rgb,
                           linewidth=0, antialiased=False, shade=False, zorder = 0)
    if sightline:
        sightlineX = [sightline.pointA.x, sightline.pointB.x]
        sightlineY = [sightline.pointA.y, sightline.pointB.y]
        sightlineZ = [elevationData[sightline.pointA.x, sightline.pointA.y], elevationData[sightline.pointB.x, sightline.pointB.y]]
        axes.plot(sightlineX, sightlineY, sightlineZ, zorder = 10)
    plt.show()