import numpy as np

def import_elevation_file(filename):
    with open(filename) as file:
        lines = file.readlines()
    output = {}
    lineNum = 0
    for line in lines:
        splitLine = line.strip('\n').split(" ")
        if splitLine[0].isalpha():
            if len(splitLine) != 2:
                raise Exception("Error: unexpected metadata format:", line)
            output[splitLine[0]] = int(splitLine[1])
        else:
            if lineNum == 0:
                data = np.ndarray(shape = (output['nrows'], output['ncols']))
            line = [float(datum) for datum in splitLine]
            data[lineNum,:] = np.array(line)
            lineNum += 1
    #
    # interchange dimensions so that columns are 'x' and rows are 'y'
    # also flip y-data so that origin is in the bottom-left (south-west)
    #
    output['data'] = np.flip(np.transpose(data), 1)
    return output