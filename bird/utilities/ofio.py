import math
import os
import re

import numpy as np

from bird import logger


def _read_mesh(filename: str) -> np.ndarray:
    """
    Reads cell center location from meshCellCentres_X.obj

    Parameters
    ----------
    filename: str
        meshCellCentres_X.obj filename

    Returns
    -------
    cell_centers: np.ndarray
        Array (N,3) representing the cell centers (N is number of cells)
    """

    assert "meshCellCentres" in filename
    assert ".obj" in filename
    cell_centers = np.loadtxt(filename, usecols=(1, 2, 3))
    return cell_centers


def _ofvec2arr(vec: str) -> np.ndarray:
    """
    Converts a vector written as a string into a numpy array

    Parameters
    ----------
    vec: str
        Vector written as string
        Must start with "("
        Must end with ")"

    Returns
    -------
    vec_array: np.ndarray
        Array (3,) representing the vector
    """

    vec = vec.strip()
    assert vec[0] == "("
    assert vec[-1] == ")"

    vec_list = vec[1:-1].split()
    vec_array = np.array([float(entry) for entry in vec_list])
    return vec_array


def _is_comment(line: str) -> bool:
    """
    Checks if line is a comment

    Parameters
    ----------
    line: str
        Line of file

    Returns
    -------
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


def _read_meta_data(filename: str, mode: str | None = None) -> dict:
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

    Returns
    -------
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
            if not _is_comment(line):
                # Get field type
                if (line.strip().startswith("class")) and (";" in line):
                    sline = line.strip().split()
                    field_type = sline[1][:-1]
                    if field_type.lower() == "volvectorfield":
                        field_type = "vector"
                    elif field_type.lower() == "volscalarfield":
                        field_type = "scalar"
                    else:
                        err_msg = f"Field type {field_type} not recognized"
                        err_msg = "Only 'volVectorField' and 'volScalarField' are supported"
                        raise NotImplementedError(err_msg)
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
                            if not _is_comment(count_line):
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
                        if meta_data["type"].lower() == "scalar":
                            sline = line.split()
                            unif_value = float(sline[-1].strip(";"))
                        elif meta_data["type"].lower() == "vector":
                            sline = line.split()
                            for ientry, entry in enumerate(sline):
                                if ";" in entry:
                                    ind_end = ientry
                                    break
                            line_cropped = " ".join(sline[2 : ientry + 1])
                            unif_value = _ofvec2arr(line_cropped.strip(";"))
                        else:
                            raise NotImplementedError(
                                f"Mode {mode} is unknown (must be 'scalar', 'vector' or None)"
                            )
                        meta_data["uniform_value"] = unif_value
                        header_done = True

            if len(line) == 0 and (not header_done):
                raise ValueError(
                    f"File {filename} ends before meta-data found"
                )

    return meta_data


def _readOFScal(
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

    Returns
    -------
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
        meta_data = _read_meta_data(filename, mode="scalar")

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
            error_msg = f"Issue when reading {filename}"
            logger.error(error_msg)
            raise err(error_msg)

    return {
        "field": field,
        "name": meta_data["name"],
        "n_cells": n_cells,
        "n_header": n_header,
    }


def _readOFVec(
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

    Returns
    -------
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
        meta_data = _read_meta_data(filename, mode="vector")

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
            error_msg = f"Issue when reading {filename}"
            logger.error(error_msg)
            raise err(error_msg)

    return {
        "field": field,
        "name": meta_data["name"],
        "n_cells": n_cells,
        "n_header": n_header,
    }


def _readOF(
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


    Returns
    -------
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
        meta_data = _read_meta_data(filename)
    if meta_data["type"] == "scalar":
        return _readOFScal(filename=filename, meta_data=meta_data)
    if meta_data["type"] == "vector":
        return _readOFVec(filename=filename, meta_data=meta_data)


def read_field(
    case_folder: str,
    time_folder: str,
    field_name: str,
    n_cells: int | None = None,
    field_dict: dict | None = None,
) -> tuple[np.ndarray | float, dict]:
    """
    Read field at a given time and store it in dictionary for later reuse

    Parameters
    ----------
    case_folder: str
        Path to case folder
    time_folder: str
        Name of time folder to analyze
    field_name: str
        Name of the field file to read
    n_cells : int | None
        Number of cells in the domain.
        If None, it will deduced from the field reading
    field_dict : dict | None
        Dictionary of fields used to avoid rereading the same fields to calculate different quantities

    Returns
    ----------
    field : np.ndarray | float
        Field read
    field_dict : dict
        Dictionary of fields read
    """
    if field_dict is None:
        field_dict = {}

    if not (field_name in field_dict) or field_dict[field_name] is None:
        # Read field if it had not been read before
        field_file = os.path.join(case_folder, time_folder, field_name)
        field = _readOF(field_file, n_cells=n_cells)["field"]
        field_dict[field_name] = field
    else:
        # Get field from dict if it has been read before
        field = field_dict[field_name]

    return field, field_dict


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


def get_case_times(
    case_folder: str, remove_zero: bool = False
) -> tuple[list[float], list[str]]:
    """
    Get list of all time folders from an OpenFOAM case

    Parameters
    ----------
    case_folder: str
        Path to case folder
    remove_zero : bool
        Whether to remove zero from the time folder list

    Returns
    -------
    time_float_sorted: list[float]
        List of time folder values in ascending order
    time_str_sorted: list[str]
        List of time folder names in ascending order

    """
    # Read Time
    times_tmp = os.listdir(case_folder)
    # remove non floats
    for i, entry in reversed(list(enumerate(times_tmp))):
        try:
            a = float(entry)
            if remove_zero:
                if abs(a) < 1e-12:
                    _ = times_tmp.pop(i)
        except ValueError:
            logger.debug(f"{entry} not a time folder, removing")
            a = times_tmp.pop(i)
            # print('removed ', a)
    time_float = [float(entry) for entry in times_tmp]
    time_str = [entry for entry in times_tmp]
    index_sort = np.argsort(time_float)
    time_float_sorted = [time_float[i] for i in list(index_sort)]
    time_str_sorted = [time_str[i] for i in list(index_sort)]

    return time_float_sorted, time_str_sorted


def _get_mesh_time(case_folder: str) -> str | None:
    """
    Get the time at which the mesh was printed

    Parameters
    ----------
    case_folder: str
        Path to case folder

    Returns
    ----------
    time_mesh: str | None
        The name of the time at which "meshCellCentresXXX" was created
        If None, nothing was found
    """

    files_tmp = os.listdir(case_folder)
    time_mesh = None
    for entry in files_tmp:
        if "meshCellCentres" in entry:
            time_mesh = entry[16:-4]

    return time_mesh


def _get_volume_time(case_folder: str) -> str | None:
    """
    Get the time at which the volume was printed

    Parameters
    ----------
    case_folder: str
        Path to case folder

    Returns
    ----------
    time_volume: str | None
        The name of the time at which "V" was created
        If None, nothing was found
    """

    time_float, time_str = get_case_times(case_folder)
    time_volume = None
    for entry in time_str:
        if os.path.exists(os.path.join(case_folder, entry, "V")):
            logger.debug(f"Volume time found to be {entry}")
            time_volume = entry
            break

    return time_volume


def _remove_comments(text: str) -> str:
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


def _tokenize(text: str) -> list[str]:
    """
    Add spaces around special characters (brace and semicolons) to make them separate tokens

    Parameters
    ----------
    text: str
        The cleaned (comment-free) OpenFOAM-style text.

    Returns
    ----------
    token_list: list[str]
        List of tokens.
    """
    text = re.sub(r"([{}();])", r" \1 ", text)
    text = re.sub(r'"\s*\(\s*([^)]+?)\s*\)\s*"', r'"(\1)"', text)

    token_list = text.split()

    # print(token_list)
    # print(text)

    return token_list


def _parse_tokens(tokens: list[str]) -> dict:
    """
    Parse OpenFOAM tokens into a nested Python dictionary.
    Special handling for `code { ... }` blocks to be stored as raw strings.

    Parameters
    ----------
    tokens: list[str]
        A list of tokens produced by `_tokenize`.

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


def read_openfoam_dict(filename: str) -> dict:
    """
    Parse OpenFOAM dictionary into a python dictionary

    Parameters
    ----------
    filename: str
        OpenFOAM dictionary filename

    Returns
    -------
    dict_of: dict
        A Python dictionary representing the structure of the OpenFOAM dictionary.
    """
    with open(filename, "r+") as f:
        text = f.read()
    text = _remove_comments(text)
    tokens = _tokenize(text)
    foam_dict = _parse_tokens(tokens)
    return foam_dict


def write_openfoam_dict(data: dict, filename: str, indent: int = 0) -> None:
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

    def write_block(f, key, value, indent=0):
        pad = " " * indent
        if isinstance(value, dict):
            f.write(f"{pad}{key}\n{pad}{{\n")
            for k, v in value.items():
                write_block(f, k, v, indent + 4)
            f.write(f"{pad}}}\n")
        elif isinstance(value, list):
            if all(isinstance(v, str) for v in value):
                f.write(f"{pad}{key}\n{pad}(\n")
                for v in value:
                    f.write(f"{pad}    {v}\n")
                f.write(f"{pad});\n")
            else:
                # assume list of numbers for OpenFOAM vectors
                joined = " ".join(value)
                f.write(f"{pad}{key}    ( {joined} );\n")
        else:
            f.write(f"{pad}{key}    {value};\n")

    with open(filename, "w") as f:
        # Write OpenFOAM header
        f.write(
            r"""/*--------------------------------*- C++ -*----------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  9
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/
"""
        )
        # Write FoamFile block first
        foam_file = data.pop("FoamFile", None)
        if foam_file:
            write_block(f, "FoamFile", foam_file)
        f.write(
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
        )

        # Then write the rest of the blocks
        for key, value in data.items():
            write_block(f, key, value)
            f.write("\n")

        # Write OpenFOAM footer
        f.write(
            "// ************************************************************************* //\n"
        )


def read_cell_centers(
    case_folder: str,
    cell_centers_file: str | None = None,
    field_dict: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Read field of cell centers and store it in dictionary for later reuse

    Parameters
    ----------
    case_folder: str
        Path to case folder
    cell_centers_file : str
        Filename of cell center data
        If None, find the cell center file automoatically
    field_dict : dict | None
        Dictionary of fields used to avoid rereading the same fields to calculate different quantities

    Returns
    ----------
    cell_centers : np.ndarray
        cell centers read from file
    field_dict : dict
        Dictionary of fields read
    """
    if field_dict is None:
        field_dict = {}

    if (
        not ("cell_centers" in field_dict)
        or field_dict["cell_centers"] is None
    ):
        if cell_centers_file is None:
            # try to find the mesh time
            mesh_time = _get_mesh_time(case_folder)
            if mesh_time is not None:
                cell_centers_file = f"meshCellCentres_{mesh_time}.obj"

        try:
            cell_centers = _read_mesh(
                os.path.join(case_folder, cell_centers_file)
            )
            field_dict["cell_centers"] = cell_centers

        except FileNotFoundError:

            error_msg = f"Could not find {cell_centers_file}"
            error_msg += "You can generate it with\n\t"
            error_msg += f"`writeMeshObj -case {case_folder}`\n"
            time_float, time_str = get_case_times(case_folder)
            correct_path = f"meshCellCentres_{time_str[0]}.obj"
            if not correct_path == cell_centers_file:
                error_msg += (
                    f"And adjust the cell center file path to {correct_path}"
                )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
    else:
        cell_centers = field_dict["cell_centers"]

    return cell_centers, field_dict


def read_cell_volumes(
    case_folder: str,
    time_folder: str | None = None,
    n_cells: int | None = None,
    field_dict: dict | None = None,
) -> tuple[np.ndarray | float, dict]:
    """
    Read volume at a given time and store it in dictionary for later reuse

    Parameters
    ----------
    case_folder: str
        Path to case folder
    time_folder: str | None
        Name of time folder to analyze.
        If None, it will be found automatically
    n_cells : int | None
        Number of cells in the domain.
        If None, it will deduced from the field reading
    field_dict : dict | None
        Dictionary of fields used to avoid rereading the same fields to calculate different quantities

    Returns
    ----------
    cell_volumes : np.ndarray | float
        Field of cell volumes
    field_dict : dict
        Dictionary of fields read
    """
    if field_dict is None:
        field_dict = {}

    kwargs_vol = {
        "case_folder": case_folder,
        "time_folder": time_folder,
        "n_cells": n_cells,
    }

    if not ("V" in field_dict) or field_dict["V"] is None:
        if time_folder is None:
            # Find the time at which the volume was printed
            time_folder = _get_volume_time(case_folder)
            kwargs_vol["time_folder"] = time_folder
        try:
            cell_volumes, field_dict = read_field(
                field_name="V", field_dict=field_dict, **kwargs_vol
            )

        except FileNotFoundError:
            error_msg = f"Could not find {os.path.join(case_folder, time_folder, 'V')}\n"
            time_float, time_str = get_case_times(case_folder)
            error_msg += "You can generate V with\n\t"
            error_msg += f"`postProcess -func writeCellVolumes -time {time_str[0]} -case {case_folder}`"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
    else:
        # Get field from dict if it has been read before
        cell_volumes = field_dict["V"]

    return cell_volumes, field_dict


def read_bubble_diameter(
    case_folder: str,
    time_folder: str | None = None,
    n_cells: int | None = None,
    field_dict: dict | None = None,
) -> tuple[np.ndarray | float, dict]:
    """
    Read bubble diameter at a given time and store it in dictionary for later reuse.
    A specific function is constructed so that if d.gas is not available, the bubble
    diameter is read from phaseProperties.

    Parameters
    ----------
    case_folder: str
        Path to case folder
    time_folder: str | None
        Name of time folder to analyze.
        If None, it will be found automatically
    n_cells : int | None
        Number of cells in the domain.
        If None, it will deduced from the field reading
    field_dict : dict | None
        Dictionary of fields used to avoid rereading the same fields to calculate different quantities

    Returns
    ----------
    cell_volumes : np.ndarray | float
        Field of cell volumes
    field_dict : dict
        Dictionary of fields read
    """

    if field_dict is None:
        field_dict = {}

    kwargs = {
        "case_folder": case_folder,
        "time_folder": time_folder,
        "n_cells": n_cells,
    }

    if not ("d.gas" in field_dict) or field_dict["d.gas"] is None:
        try:
            d_gas, field_dict = read_field(
                field_name="d.gas", field_dict=field_dict, **kwargs
            )

        except FileNotFoundError as err:
            logger.debug(
                "Could not find d.gas, checking if bubble size model is constant"
            )
            # d.gas does not exist, it might be because a constant bubble diameter model is used
            phaseProperties_dict = read_openfoam_dict(
                os.path.join(case_folder, "constant", "phaseProperties")
            )
            # If the bubble size model is not constant, raise original error
            if (
                not phaseProperties_dict["gas"]["diameterModel"].lower()
                == "constant"
            ):
                logger.error(
                    f"Bubble size model is not constant ({phaseProperties_dict['gas']['diameterModel']}), yet could not find d.gas"
                )
                raise FileNotFoundError(err)
            else:
                # Get the constant bubble diameter
                logger.debug("Reading bubble diameter from phaseProperties")
                d_gas = float(
                    phaseProperties_dict["gas"]["constantCoeffs"]["d"]
                )

    else:
        # Get field from dict if it has been read before
        d_gas = field_dict["d.gas"]

    return d_gas, field_dict


def _cross_reference_global_vars(
    globalVars_dict: dict,
) -> dict:
    """
    Resolve cross referencing in the globalVars dictionary read
    so that all the values are float
    """

    cross_referenced_globalVars_dict = globalVars_dict.copy()

    # capture $Variable references
    var_pattern = re.compile(r"\$([A-Za-z0-9_]+)")

    for key, value in globalVars_dict.items():
        # Check if we need to do cross referencing
        if isinstance(value, str) and value.startswith("#calc"):

            # Extract expression inside quotes
            expr = value.split('"', 1)[1].rsplit('"', 1)[0]

            # Replace $Var with its numeric value from globalVars_dict
            def repl(match):
                varname = match.group(1)
                if varname not in cross_referenced_globalVars_dict:
                    raise KeyError(
                        f"Variable '{varname}' not found for expression {expr}"
                    )
                return str(cross_referenced_globalVars_dict[varname])

            expr = var_pattern.sub(repl, expr)

            # Replace 'Foam::pow' with 'math.pow'
            if "Foam::pow" in expr:
                expr = expr.replace("Foam::pow", "math.pow")

            if "pow" in expr:
                indstr = [m.start() for m in re.finditer("pow", expr)]
                new_expr = expr
                count_replace = 0
                for ind in indstr:
                    if not "mat" in new_expr[ind - 6 : ind + 3]:
                        tmp = new_expr
                        new_expr = (
                            tmp[: ind + 5 * count_replace]
                            + "math.pow"
                            + tmp[ind + 5 * count_replace + 3 :]
                        )
                        count_replace += 1
                expr = new_expr
            # Replace 'exp' with 'math.exp'
            expr = expr.replace("exp", "math.exp")

            try:
                result = eval(expr, {"math": math})
            except Exception as e:
                raise RuntimeError(
                    f"Error evaluating globalVars expression for {key}: {expr}"
                ) from e

            # Convert to int if whole number
            if isinstance(result, float) and result.is_integer():
                result = int(result)

            cross_referenced_globalVars_dict[key] = result

    return cross_referenced_globalVars_dict


def read_global_vars(
    case_folder: str | None = None,
    filename: str | None = None,
    cross_ref: bool = True,
) -> dict:
    """
    Read globalVars into a python dictionary

    Parameters
    ----------
    case_folder: str | None
        Path to case folder
        If None, using filename
    filename: str | None
        Path to the globalVars file
        If None, using case_folder
    cross_ref: bool
        Do cross referencing or no
        If True, all the `#calc` values from the python dict are replaced by their numeral values
        If False, all the `#calc` values are stored as string

    Returns
    ----------
    globalVars_dict : dict
        Dictionary that contains the globals vars variable values
    """

    # Set the filename to read
    if case_folder is None:
        if filename is None or not os.path.isfile(filename):
            error_msg = f"Global Vars file ({filename}) not found, but is needed if case folder not specified"
            raise FileNotFoundError(error_msg)
    else:
        if not os.path.exists(case_folder):
            error_msg = f"Case folder ({case_folder}) not found, but is needed if global vars filename not specified"
            raise FileNotFoundError(error_msg)
        filename = os.path.join(case_folder, "constant", "globalVars")

    logger.debug(f"Reading {filename}")

    # Now read globalVars
    globalVars_dict = {}
    with open(filename, "r") as f:
        for line in f:
            # Remove comments
            line = line.split("//", 1)[0].strip()
            if not line:
                continue  # skip empty/comment-only lines

            # Ensure it ends with ;
            if not line.endswith(";"):
                continue
            line = line[:-1].strip()  # remove trailing ;

            # Split key and value
            parts = line.split(None, 1)  # split on first whitespace
            if len(parts) != 2:
                continue
            key, value = parts
            logger.debug(f"\tReading {key}")

            # Keep calc expressions as-is
            if value.startswith("#calc"):
                globalVars_dict[key] = value
            else:
                # Try numeric conversion otherwise store value as str
                try:
                    val = float(value)
                    if val.is_integer():
                        val = int(val)
                    globalVars_dict[key] = val
                except ValueError:
                    globalVars_dict[key] = value
    if cross_ref:
        # Remove the #calc stuff
        globalVars_dict = _cross_reference_global_vars(globalVars_dict)

    return globalVars_dict


def _species_name_to_mw(case_folder: str, species_name: str) -> float:
    """
    Get species molecular weight from thermophysical properties or globalVars
    Return species molecular weight in kg/mol
    """
    thermo_gas_filename = os.path.join(
        case_folder, "constant", "thermophysicalProperties.gas"
    )
    thermo_liq_filename = os.path.join(
        case_folder, "constant", "thermophysicalProperties.liquid"
    )
    globalVars_filename = os.path.join(case_folder, "constant", "globalVars")

    assert isinstance(species_name, str)

    # Special case for water
    if species_name.lower() in ["water", "h2o"]:
        species_names = ["water", "h2o", "H2O", "WATER"]
    else:
        species_names = [
            species_name,
            species_name.upper(),
            species_name.lower(),
        ]

    mw = None
    # Try finding the molecular weight from thermo liquid
    if os.path.isfile(thermo_liq_filename):
        thermo = read_openfoam_dict(thermo_liq_filename)
        for name in species_names:
            if name in thermo:
                mw = float(thermo[name]["specie"]["molWeight"]) * 1e-3
                break
        if mw is not None:
            logger.debug(
                f"Read the {species_name} molecular weight ({mw}) from thermophysicalProperties.liquid"
            )
        else:
            logger.debug(
                f"Could not read the {species_name} molecular weight from thermophysicalProperties.liquid"
            )

    # Try finding the molecular weight from thermo gas
    if mw is None and os.path.isfile(thermo_gas_filename):
        thermo = read_openfoam_dict(thermo_gas_filename)
        for name in species_names:
            if name in thermo:
                mw = float(thermo[name]["specie"]["molWeight"]) * 1e-3
                break
        if mw is not None:
            logger.debug(
                f"Read the {species_name} molecular weight ({mw}) from thermophysicalProperties.gas"
            )
        else:
            logger.debug(
                f"Could not read the {species_name} molecular weight from thermophysicalProperties.gas"
            )

    # Last resort: try finding the molecular weight from thermo gas
    if mw is None and os.path.isfile(globalVars_filename):
        globalVars = read_global_vars(case_folder=case_folder, cross_ref=True)
        for name in species_names:
            if f"Mw_{name}" in globalVars:
                mw = float(globalVars[f"Mw_{name}"])
                break
        if mw is not None:
            mw = globalVars[f"Mw_{name}"]
        else:
            err_msg = f"Mw_{species_name} was not found in globalVars."
            err_msg += "\nIf you add it, it should be [kg/mol]"
            raise KeyError(err_msg)

    return mw
