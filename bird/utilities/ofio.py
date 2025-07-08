import os
import re
import sys

import numpy as np


def readMesh(filename: str) -> np.ndarray:
    """
    Reads cell center location from meshCellCentres_X.obj

    Parameters
    ----------
    filename: str
        meshCellCentres_X.obj filename

    returns
    ----------
    cell_centers: np.ndarray
        Array (N,3) representing the cell centers (N is number of cells)

    """
    assert "meshCellCentres" in filename
    assert ".obj" in filename
    cell_centers = np.loadtxt(filename, usecols=(1, 2, 3))
    return cell_centers


def ofvec2arr(vec: str) -> np.ndarray:
    """
    Converts a vector written as a string into a numpy array

    Parameters
    ----------
    vec: str
        Vector written as string
        Must start with "("
        Must end with ")"

    returns
    ----------
    vec_array: np.ndarray
        Array (3,) representing the vector


    """
    vec = vec.strip()
    assert vec[0] == "("
    assert vec[-1] == ")"

    vec_list = vec[1:-1].split()
    vec_array = np.array([float(entry) for entry in vec_list])
    return vec_array


def is_comment(line: str) -> bool:
    """
    Checks if line is a comment

    Parameters
    ----------
    line: str
        Line of file

    returns
    ----------
    is_comment: bool
        True if line is a comment
        False if line is not a comment
    """
    is_comment = False

    sline = line.strip()
    if sline.startswith("//"):
        is_comment = True
    elif sline.startswith("/*"):
        is_comment = True
    return is_comment


def read_meta_data(filename: str, mode: str | None = None) -> dict:
    """
    Read meta data from field outputted by OpenFOAM in ASCII format

    Parameters
    ----------
    filename: str
        Field filename
    mode: str | None
        If "scalar", expects a scalar field
        If "vector", expects a vector field
        If None, obtained from field header

    returns
    ----------
    meta_data: dict
        Dictionary that contain info about the scalar field
            ============= =====================================================
            Key           Description (type)
            ============= =====================================================
            name          Name of the field (*str*)
            n_cells       Number of computational cells (*int*)
            uniform       Whether the field is uniform (*bool*)
            uniform_value Uniform value if uniform field (*float* | *np.ndarray*)
            type          "vector" or "scalar" (*str*)
            ============= =====================================================
    """
    meta_data = {}
    meta_data["type"] = mode

    # Read meta data
    with open(filename, "r") as f:
        header_done = False
        iline = 0
        while not header_done:
            line = f.readline()
            if not is_comment(line):
                # Get field type
                if (line.strip().startswith("class")) and (";" in line):
                    sline = line.strip().split()
                    field_type = sline[1][:-1]
                    if field_type == "volVectorField":
                        field_type = "vector"
                    elif field_type == "volScalarField":
                        field_type = "scalar"
                    else:
                        raise NotImplementedError
                    if mode is not None:
                        assert field_type == mode
                    meta_data["type"] = field_type
                # Get field name
                if (line.strip().startswith("object")) and (";" in line):
                    sline = line.strip().split()
                    field_name = sline[1][:-1]
                    meta_data["name"] = field_name

                # Check if uniform
                if line.strip().startswith("internalField"):
                    if "nonuniform" in line:
                        meta_data["uniform"] = False
                        iline += 1
                        # read until no comments
                        comment = True
                        while comment:
                            count_line = f.readline().strip()
                            if not is_comment(count_line):
                                comment = False
                        try:
                            n_cells = int(count_line)
                            meta_data["n_cells"] = n_cells
                            header_done = True
                        except ValueError:
                            raise ValueError(
                                f"Expected integer number of cells on line {iline}, got: '{count_line}'"
                            )
                    elif "uniform" in line:
                        meta_data["uniform"] = True
                        sline = line.split()
                        if meta_data["type"] == "scalar":
                            unif_value = float(sline[-1].strip(";"))
                        elif meta_data["type"] == "vector":
                            unif_value = ofvec2arr(sline[-1].strip(";"))
                        else:
                            raise NotImplementedError(
                                f"Mode {mode} is unknown"
                            )
                        meta_data["uniform_value"] = unif_value
                        header_done = True

            if len(line) == 0 and (not header_done):
                raise ValueError(
                    f"File {filename} ends before meta-data found"
                )

    return meta_data


def readOFScal(
    filename: str,
    n_cells: int | None = None,
    n_header: int | None = None,
    meta_data: dict | None = None,
) -> dict:
    """
    Read a scalar field outputted by OpenFOAM in ASCII format

    Parameters
    ----------
    filename: str
        Field filename
    n_cells : int | None
        Number of computational cells in the domain
    n_header : int | None
        Number of header lines
    meta_data : dict | None
        meta data dictionary
        If None, it is read from filename

    returns
    ----------
    data: dict
        Dictionary that contain info about the scalar field
            ======== =====================================================
            Key      Description (type)
            ======== =====================================================
            field    For nonuniform fields, array of size the number of cells (*np.ndarray*).
                     For uniform fields with a specified n_cells,
                          array of size the number of cells (*np.ndarray*).
                     For uniform fields, a scalar value (*float*)
            name     Name of the scalar field (*str*)
            n_cells  Number of computational cells (*int*)
            n_header Number of header lines (*int*)
            ======== =====================================================
    """
    field = None

    if meta_data is None:
        # Read meta data
        meta_data = read_meta_data(filename, mode="scalar")

    # Set field
    if meta_data["uniform"]:
        if n_cells is None:
            field = meta_data["uniform_value"]
        else:
            field = meta_data["uniform_value"] * np.ones(n_cells)
    else:
        n_cells = meta_data["n_cells"]

        # Get header size
        if n_header is None:
            oldline = ""
            newline = ""
            iline = 0
            eof = False
            with open(filename, "r") as f:
                while iline < 100 and (not eof):
                    line = f.readline()
                    if len(line) == 0:
                        eof = True
                    oldline = newline
                    newline = line
                    if str(n_cells) in oldline and "(" in newline:
                        n_header = iline + 1
                        break
                    iline += 1
                else:
                    raise ValueError(
                        "Could not find a sequence {n_cells} and '(' in file ({filename})"
                    )

        # Rapid field read
        try:
            field = np.loadtxt(filename, skiprows=n_header, max_rows=n_cells)
        except Exception as err:
            print(f"Issue when reading {filename}")
            print(err)
            sys.exit()

    return {
        "field": field,
        "name": meta_data["name"],
        "n_cells": n_cells,
        "n_header": n_header,
    }


def readOFVec(
    filename: str,
    n_cells: int | None = None,
    n_header: int | None = None,
    meta_data: dict | None = None,
) -> dict:
    """
    Read a vector field outputted by OpenFOAM in ASCII format

    Parameters
    ----------
    filename: str
        Vector field filename
    n_cells : int | None
        Number of computational cells in the domain
    n_header : int | None
        Number of header lines
    meta_data : dict | None
        meta data dictionary
        If None, it is read from filename

    returns
    ----------
    data: dict
        Dictionary that contain info about the scalar field
            ======== =====================================================
            Key      Description (type)
            ======== =====================================================
            field    For nonuniform fields, array of size the number of cells by 3 (*np.ndarray*).
                     For uniform fields with a specified n_cells,
                          array of size the number of cells by 3 (*np.ndarray*).
                     For uniform fields, a scalar value (*float*)
            name     Name of the field (*str*)
            n_cells  Number of computational cells (*int*)
            n_header Number of header lines (*int*)
            ======== =====================================================
    """
    field = None

    if meta_data is None:
        # Read meta data
        meta_data = read_meta_data(filename, mode="vector")

    # Set field
    if meta_data["uniform"]:
        if n_cells is None:
            field = meta_data["uniform_value"]
        else:
            field = meta_data["uniform_value"] * np.ones((n_cells, 3))
    else:
        n_cells = meta_data["n_cells"]
        if n_header is None:
            oldline = ""
            newline = ""
            iline = 0
            eof = False
            with open(filename, "r") as f:
                while iline < 100 and (not eof):
                    line = f.readline()
                    if len(line) == 0:
                        eof = True
                    oldline = newline
                    newline = line
                    if str(n_cells) in oldline and "(" in newline:
                        n_header = iline + 1
                        break
                    iline += 1
                else:
                    raise ValueError(
                        "Could not find a sequence {n_cells} and '(' in file ({filename})"
                    )

        try:
            field = np.loadtxt(
                filename, dtype=tuple, skiprows=n_header, max_rows=n_cells
            )
            for i in range(n_cells):
                field[i, 0] = float(field[i, 0][1:])
                field[i, 1] = float(field[i, 1])
                field[i, 2] = float(field[i, 2][:-1])
            field = np.array(field).astype(float)

        except Exception as err:
            print(f"Issue when reading {filename}")
            print(err)
            sys.exit()

    return {
        "field": field,
        "name": meta_data["name"],
        "n_cells": n_cells,
        "n_header": n_header,
    }


def readOF(
    filename: str,
    n_cells: int | None = None,
    n_header: int | None = None,
    meta_data: dict | None = None,
) -> dict:
    """
    Read an OpenFOAM field outputted in ASCII format

    Parameters
    ----------
    filename: str
        Field filename
    n_cells : int | None
        Number of computational cells in the domain
    n_header : int | None
        Number of header lines
    meta_data : dict | None
        meta data dictionary
        If None, it is read from filename


    returns
    ----------
    data: dict
        Dictionary that contain info about the scalar field
            ======== =====================================================
            Key      Description (type)
            ======== =====================================================
            field    For nonuniform fields, array of size the number of cells (*np.ndarray*).
                     For uniform fields with a specified n_cells,
                          array of size the number of cells (*np.ndarray*).
                     For uniform fields, a scalar value (*float*)
            name     Name of the field (*str*)
            n_cells  Number of computational cells (*int*)
            n_header Number of header lines (*int*)
            ======== =====================================================
    """
    if meta_data is None:
        # Read meta data
        meta_data = read_meta_data(filename)
    if meta_data["type"] == "scalar":
        return readOFScal(filename=filename, meta_data=meta_data)
    if meta_data["type"] == "vector":
        return readOFVec(filename=filename, meta_data=meta_data)


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


def getCaseTimes(
    casePath: str, remove_zero: bool = False, verbose: bool = True
) -> tuple:
    """
    Get list of all time folders from an OpenFOAM case

    Parameters
    ----------
    casePath: str
        Path to case folder
    remove_zero : bool
        Whether to remove zero from the time folder list
    verbose : bool
        Whether to print what time folders are included

    returns
    ----------
    time_float_sorted: list[float]
        List of time folder values in ascending order
    time_str_sorted: list[str]
        List of time folder names in ascending order

    """
    # Read Time
    times_tmp = os.listdir(casePath)
    # remove non floats
    for i, entry in reversed(list(enumerate(times_tmp))):
        try:
            a = float(entry)
            if remove_zero:
                if abs(a) < 1e-12:
                    _ = times_tmp.pop(i)
        except ValueError:
            if verbose:
                print(f"{entry} not a time folder, removing")
            a = times_tmp.pop(i)
            # print('removed ', a)
    time_float = [float(entry) for entry in times_tmp]
    time_str = [entry for entry in times_tmp]
    index_sort = np.argsort(time_float)
    time_float_sorted = [time_float[i] for i in list(index_sort)]
    time_str_sorted = [time_str[i] for i in list(index_sort)]

    return time_float_sorted, time_str_sorted


def getMeshTime(casePath: str) -> str:
    """
    Get the time at which the mesh was printed

    Parameters
    ----------
    casePath: str
        Path to case folder

    Returns
    ----------
    time_mesh: str
        The name of the time at which "meshFaceCentresXXX" was created
    """

    files_tmp = os.listdir(casePath)
    for entry in files_tmp:
        if "meshFaceCentres" in entry:
            time_mesh = entry[16:-4]
            return time_mesh


def remove_comments(text: str) -> str:
    """
    Remove C++-style comments (// and /*) from the input and markers like #{ #}

    Parameters
    ----------
    text: str
        Raw input text containing comments

    Returns
    ----------
    text: str
        Text with all comments removed
    """

    # text = re.sub(
    #    r"/\*.*?\*/", "", text, flags=re.DOTALL
    # )  # Remove /* */ comments
    # text_unc = re.sub(r"//.*", "", text)  # Remove // comments

    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//.*", "", text)
    text = re.sub(r"#\{", "{", text)
    text = re.sub(r"#\};", "}", text)
    text = re.sub(r"#codeStream", "", text)
    return text


def tokenize(text: str) -> list[str]:
    """
    Add spaces around special characters (brace and semicolons) to make them separate tokens

    Parameters
    ----------
    text: str
        The cleaned (comment-free) OpenFOAM-style text.

    Returns:
    ----------
    token_list: list[str]
        List of tokens.
    """
    text = re.sub(r"([{}();])", r" \1 ", text)
    token_list = text.split()
    return token_list


def parse_tokens(tokens: list[str]) -> dict:
    """
    Parse OpenFOAM tokens into a nested Python dictionary.
    Special handling for `code { ... }` blocks to be stored as raw strings.

    Parameters
    ----------
    tokens: list[str]
        A list of tokens produced by `tokenize`.

    Returns
    ----------
    parsed: dict
        A nested dictionary that represents the OpenFOAM dictionary.
    """

    def parse_block(index: int) -> tuple:
        result = {}
        while index < len(tokens):
            token = tokens[index]
            if token == "}":
                return result, index + 1
            elif token == "{":
                raise SyntaxError("Unexpected '{'")

            key = token
            index += 1

            # key followed by dictionary
            if index < len(tokens) and tokens[index] == "{":
                index += 1
                if key == "code":
                    code_lines = []
                    while tokens[index] != "}":
                        code_lines.append(tokens[index])
                        index += 1
                    index += 1
                    if index < len(tokens) and tokens[index] == ";":
                        index += 1
                    result[key] = " ".join(code_lines).strip()
                else:
                    subdict, index = parse_block(index)
                    result[key] = subdict

            # key followed by list
            elif index < len(tokens) and tokens[index] == "(":
                index += 1

                # Peek to check if it's a dict-list (starts with '(' then '{')
                if tokens[index] == "(":
                    dictlist = {}
                    while tokens[index] != ")":
                        if tokens[index] != "(":
                            raise SyntaxError(
                                f"Expected '(' for label in dict-list, got {tokens[index]}"
                            )
                        # Read full label (e.g., "(gas and liquid)")
                        label_tokens = []
                        while tokens[index] != ")":
                            label_tokens.append(tokens[index])
                            index += 1
                        label_tokens.append(tokens[index])  # include ')'
                        index += 1
                        label = " ".join(label_tokens)

                        if tokens[index] != "{":
                            raise SyntaxError(
                                f"Expected '{{' after label {label}"
                            )
                        index += 1
                        subdict, index = parse_block(index)
                        dictlist[label] = subdict
                    index += 1  # skip final ')'
                    if index < len(tokens) and tokens[index] == ";":
                        index += 1
                    result[key] = dictlist
                else:
                    # Standard list
                    lst = []
                    while tokens[index] != ")":
                        lst.append(tokens[index])
                        index += 1
                    index += 1
                    if index < len(tokens) and tokens[index] == ";":
                        index += 1
                    result[key] = lst

            # key followed by scalar
            elif index < len(tokens):
                value = tokens[index]
                index += 1
                if index < len(tokens) and tokens[index] == ";":
                    index += 1
                result[key] = value

        return result, index

    parsed, _ = parse_block(0)
    return parsed


def parse_openfoam_dict(filename: str) -> dict:
    """
    Parse OpenFOAM dictionary into a python dictionary

    Parameters
    ----------
    filename: str
        OpenFOAM dictionary filename

    Returns
    ----------
    dict_of: dict
        A Python dictionary representing the structure of the OpenFOAM dictionary.
    """
    with open(filename, "r+") as f:
        text = f.read()
    text = remove_comments(text)
    tokens = tokenize(text)
    foam_dict = parse_tokens(tokens)
    return foam_dict


def write_openfoam_dict(d: dict, filename: str, indent: int = 0) -> None:
    """
    Save a Python dictionary back to an OpenFOAM-style file.

    Parameters
    ----------
    d: dict
        Python dictionary to save
    filename: str
        The file that will contain the saved dictionary
    indent: int
        Number of indentation space
    """

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
