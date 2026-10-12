import os

import numpy as np

from bird import logger


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


def _find_header_size(
    filename: str, n_cells: int, max_lines: int = 100
) -> int:
    """
    Count the lines before the internal field values start

    The values of a nonuniform field are preceded by the number of cells on
    its own line, then a line opening the list with "(".

    Parameters
    ----------
    filename: str
        Field filename
    n_cells : int
        Number of computational cells in the domain
    max_lines : int
        Number of lines to scan before giving up

    Returns
    -------
    n_header: int
        Number of header lines to skip
    """
    oldline = ""
    newline = ""
    with open(filename, "r") as f:
        for iline in range(max_lines):
            line = f.readline()
            if len(line) == 0:
                # End of file reached before the values started
                break
            oldline = newline
            newline = line
            if str(n_cells) in oldline and "(" in newline:
                return iline + 1

    error_msg = f"Could not find the sequence {n_cells} then '('"
    error_msg += f" in the first {max_lines} lines of {filename}"
    logger.error(error_msg)
    raise ValueError(error_msg)


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
            n_header = _find_header_size(filename, n_cells)

        # Rapid field read
        try:
            field = np.loadtxt(filename, skiprows=n_header, max_rows=n_cells)
        except Exception as err:
            error_msg = f"Issue when reading {filename}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from err

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
            n_header = _find_header_size(filename, n_cells)

        # Each line reads "(x y z)", so strip the parentheses off the first
        # and last column as they are parsed
        try:
            field = np.loadtxt(
                filename,
                skiprows=n_header,
                max_rows=n_cells,
                converters={
                    # remove ( in the first entry
                    # Don't touch the middle entry
                    # Remove ) from the last entry
                    0: lambda entry: float(entry[1:]),
                    2: lambda entry: float(entry[:-1]),
                },
            )

        except Exception as err:
            error_msg = f"Issue when reading {filename}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from err

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
