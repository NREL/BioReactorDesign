import os
import sys

import numpy as np


def readMesh(file):
    A = np.loadtxt(file, usecols=(1, 2, 3))
    return A


def readOFScal(file, nCells, nHeader=None):
    # Check that the field is not a uniform field
    try:
        f = open(file, "r")
        for i in range(20):
            line = f.readline()
        f.close()
        lineList = line.split()
        if len(lineList) == 3 and lineList[1] == "uniform":
            # Uniform field
            val = float(lineList[2][:-1])
            Array = val * np.ones(nCells)
        else:
            # Not a uniform field
            f = open(file, "r")
            if nHeader is None:
                # Find header
                lines = f.readlines()
                for iline, line in enumerate(lines[:-2]):
                    if str(nCells) in lines[iline] and "(" in lines[iline + 1]:
                        break
                nHeader = iline + 2
                f.close()
            Array = np.loadtxt(file, skiprows=nHeader, max_rows=nCells)
    except:
        print("Issue when reading %s" % file)
        sys.exit()
    return Array


def ofvec2arr(vec):
    vec_list = vec[1:-1].split()
    vec_float = [float(entry) for entry in vec_list]
    return np.array(vec_float)


def readOFVec(file, nCells, nHeader=None):
    # Check that the field is not a uniform field
    try:
        f = open(file, "r")
        for i in range(20):
            line = f.readline()
        f.close()
        lineList = line.split()
        if len(lineList) == 3 and lineList[1] == "uniform":
            # Uniform field
            raise NotImplementedError
            val = ofvec2arr(lineList[2][:-1])
            Array = val * np.ones(nCells)
        else:
            # Not a uniform field
            f = open(file, "r")
            if nHeader is None:
                # Find header
                lines = f.readlines()
                for iline, line in enumerate(lines[:-2]):
                    if str(nCells) in lines[iline] and "(" in lines[iline + 1]:
                        break
                nHeader = iline + 2
                f.close()
            Array = np.loadtxt(
                file, dtype=tuple, skiprows=nHeader, max_rows=nCells
            )
            for i in range(nCells):
                Array[i, 0] = float(Array[i, 0][1:])
                Array[i, 1] = float(Array[i, 1])
                Array[i, 2] = float(Array[i, 2][:-1])
            Array = np.array(Array).astype(float)
    except:
        print("Issue when reading %s" % file)
        sys.exit()

    return Array


def readSizeGroups(file):
    sizeGroup = {}
    f = open(file, "r")
    lines = f.readlines()
    f.close()
    begin = None
    for iline, line in enumerate(lines):
        if "sizeGroups" in line:
            begin = iline + 2
        if not begin is None and ");" in line:
            end = iline - 1
            break
    for line in lines[begin : end + 1]:
        tmp = line.split("{")
        name = tmp[0].strip()
        size = tmp[1].split(";")[0].split()[1]
        sizeGroup[name] = float(size)
    # Sort by size
    sizeGroup = dict(sorted(sizeGroup.items(), key=lambda item: item[1]))
    binGroup = {}
    groups = list(sizeGroup.keys())
    for igroup, group in enumerate(groups):
        if igroup == 0:
            bin_size = (
                sizeGroup[groups[igroup + 1]] - sizeGroup[groups[igroup]]
            )
        elif igroup == len(groups) - 1:
            bin_size = (
                sizeGroup[groups[igroup]] - sizeGroup[groups[igroup - 1]]
            )
        else:
            bin_size_p = (
                sizeGroup[groups[igroup + 1]] - sizeGroup[groups[igroup]]
            )
            bin_size_m = (
                sizeGroup[groups[igroup]] - sizeGroup[groups[igroup - 1]]
            )
            assert abs(bin_size_p - bin_size_m) < 1e-12
            bin_size = bin_size_m
        binGroup[group] = bin_size
    return sizeGroup, binGroup


def getCaseTimes(casePath):
    # Read Time
    times_tmp = os.listdir(casePath)
    # remove non floats
    for i, entry in reversed(list(enumerate(times_tmp))):
        try:
            a = float(entry)
        except ValueError:
            a = times_tmp.pop(i)
            # print('removed ', a)
    time_float = [float(entry) for entry in times_tmp]
    time_str = [entry for entry in times_tmp]
    index_sort = np.argsort(time_float)
    time_float_sorted = [time_float[i] for i in list(index_sort)]
    time_str_sorted = [time_str[i] for i in list(index_sort)]

    return time_float_sorted, time_str_sorted


def getMeshTime(casePath):
    files_tmp = os.listdir(casePath)
    for entry in files_tmp:
        if entry.startswith("meshFaceCentres"):
            return entry[16:-4]
