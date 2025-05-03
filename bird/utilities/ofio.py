import os
import re
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
    except Exception as err:
        print("Issue when reading %s" % file)
        print(err)
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
    except Exception as err:
        print("Issue when reading %s" % file)
        print(err)
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
            print(f"{entry} not a time folder, removing")
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


def remove_comments(text):
    text = re.sub(
        r"/\*.*?\*/", "", text, flags=re.DOTALL
    )  # Remove /* */ comments
    text = re.sub(r"//.*", "", text)  # Remove // comments
    return text


def tokenize(text):
    # Add spaces around braces and semicolons to make them separate tokens
    text = re.sub(r"([{}();])", r" \1 ", text)
    return text.split()


def parse_tokens(tokens):
    def parse_block(index):
        result = {}
        while index < len(tokens):
            token = tokens[index]
            if token == "}":
                return result, index + 1
            elif token == "{":
                raise SyntaxError("Unexpected '{'")
            else:
                key = token
                index += 1
                if tokens[index] == "{":
                    index += 1
                    value, index = parse_block(index)
                    result[key] = value
                elif tokens[index] == "(":
                    # Parse list
                    index += 1
                    lst = []
                    while tokens[index] != ")":
                        lst.append(tokens[index])
                        index += 1
                    index += 1  # Skip ')'
                    result[key] = lst
                    if tokens[index] == ";":
                        index += 1
                else:
                    # Parse scalar value
                    value = tokens[index]
                    index += 1
                    if tokens[index] == ";":
                        index += 1
                    result[key] = value
        return result, index

    parsed, _ = parse_block(0)
    return parsed


def parse_openfoam_dict(text):
    text = remove_comments(text)
    tokens = tokenize(text)
    return parse_tokens(tokens)


def read_properties(filename: str):
    with open(filename, "r+") as f:
        text = f.read()
    foam_dict = parse_openfoam_dict(text)
    return foam_dict


def write_openfoam_dict(d, filename, indent=0):
    lines = []

    indent_str = " " * indent

    for key, value in d.items():
        if isinstance(value, dict):
            lines.append(f"{indent_str}{key}")
            lines.append(f"{indent_str}{{")
            lines.extend(write_openfoam_dict(value, indent + 4))
            lines.append(f"{indent_str}}}")
        elif isinstance(value, list):
            lines.append(f"{indent_str}{key}")
            lines.append(f"{indent_str}(")
            for item in value:
                lines.append(f"{indent_str}    {item}")
            lines.append(f"{indent_str});")
        else:
            lines.append(f"{indent_str}{key}    {value};")

    with open(filename, "w") as f:
        lines = write_openfoam_dict(foam_dict)
        f.write("\n".join(lines))
        f.write(
            "\n\n// ************************************************************************* //\n"
        )
