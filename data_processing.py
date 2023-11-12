import numpy as np
from os import listdir
from re import findall
from data_classes import ElevationFile

def import_elevation_file(fileName):
    with open(fileName) as file:
        lines = file.readlines()
    elevationFile = ElevationFile(0, 0, 0, 0, 0, np.array([]))
    lineNumData = 0
    for line in lines:
        splitLine = line.strip('\n').split(" ")
        if splitLine[0].isalpha():
            if len(splitLine) != 2 or not splitLine[1].isdigit():
                raise Exception("Error: unexpected metadata format:", line)
            field = splitLine[0]
            value = int(splitLine[1])
            if field == "ncols":
                elevationFile.numColumns = value
            if field == "nrows":
                elevationFile.numRows = value
            if field == "xllcorner":
                elevationFile.xCorner = value
            if field == "yllcorner":
                elevationFile.yCorner = value
            if field == "cellsize":
                elevationFile.cellSize = value
        else:
            if lineNumData == 0:
                data = np.ndarray(shape = (elevationFile.numRows, elevationFile.numColumns))
            line = [float(datum) for datum in splitLine]
            data[lineNumData,:] = np.array(line)
            lineNumData += 1
    #
    # interchange dimensions so that columns are 'x' and rows are 'y'
    # also flip y-data so that origin is in the bottom-left (south-west)
    #
    elevationFile.data = np.flip(np.transpose(data), 1)
    return elevationFile

def get_asc_file_from_folder(folderPath):
    fileNames = listdir(folderPath)
    for fileName in fileNames:
        if fileName[-4:] == ".asc":
            return fileName
    raise Exception("There are no .asc files in the folder " + folderPath)

def stitch_together_elevation_files(folderPath):
    subFolderNames = listdir(folderPath)
    elevationFiles = {}
    nFile = 0
    for subFolderName in subFolderNames:
        fileName = get_asc_file_from_folder(folderPath + "/" + subFolderName)
        gridReference = fileName[2:4]
        if not gridReference.isdigit:
            raise Exception("A file in the folder " + subFolderName + " is not named correctly: " + fileName)
        else:
            nFile += 1
            elevationFile = import_elevation_file(folderPath + "/" + subFolderName + "/" + fileName)
            elevationFiles[gridReference] = elevationFile
            numColumns = elevationFile.numColumns
            numRows = elevationFile.numRows
            if nFile != 1:
                if numColumns != elevationFile.numColumns or numRows != elevationFile.numRows:
                    #
                    # check all the files have the same format
                    #
                    raise Exception("Not all of the files are the same size")
    #
    # the code in this block assumes we want to create an elevation file for a large grid square made from 10 * 10 small grid squares
    #
    stitchedElevationFile = ElevationFile(
        numColumns = numColumns * 10,
        numRows = numRows * 10,
        xCorner = elevationFiles["00"].xCorner,
        yCorner = elevationFiles["00"].yCorner,
        cellSize = elevationFiles["00"].cellSize,
        data = np.zeros((numRows * 10, numRows * 10))
    )
    for (gridReference,elevationFile) in elevationFiles.items():
        bigX = int(gridReference[0])
        bigY = int(gridReference[1])
        stitchedElevationFile.data[numColumns * bigX : numColumns * (bigX + 1), numRows * bigY : numRows * (bigY + 1)] = elevationFile.data
    return stitchedElevationFile