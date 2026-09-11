import math
import os
import re

import numpy as np

from bird import logger

from .foam_dict_io import read_openfoam_dict


def _cross_reference_math(expr):
    """
    Resolve math in cross referencing in the globalVars dictionary
    """
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
    # Replace 'sin' with 'math.sin'
    expr = expr.replace("sin", "math.sin")
    # Replace 'cos' with 'math.cos'
    expr = expr.replace("cos", "math.cos")
    # Replace 'tan' with 'math.tan'
    expr = expr.replace("tan", "math.tan")

    degToRad = lambda x: x * np.pi / 180.0
    return expr, degToRad


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
        if isinstance(value, str):
            # CALC case
            if value.startswith("#calc"):
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

                expr, degToRad = _cross_reference_math(expr)
                try:
                    # The builtins key allow to not replace anything else
                    # than the math
                    result = eval(
                        expr,
                        {
                            "__builtins__": {},
                            "math": math,
                            "degToRad": degToRad,
                        },
                    )
                    result = float(result)
                    # Convert to int if whole number
                    # if result.is_integer():
                    #    result = int(result)
                    cross_referenced_globalVars_dict[key] = result

                except Exception as e:
                    logger.warning(
                        f"Could not evaluate globalVars expression for {key}: {expr}"
                    )
            # Direct variable references
            elif "$" in value:

                def repl_direct(match):
                    varname = match.group(1)
                    return str(cross_referenced_globalVars_dict[varname])

                try:
                    # Substitute the variable
                    expr = var_pattern.sub(repl_direct, value)
                    expr, degToRad = _cross_reference_math(expr)
                    # The builtins key allow to not replace anything else
                    # than the math
                    result = eval(
                        expr,
                        {
                            "__builtins__": {},
                            "math": math,
                            "degToRad": degToRad,
                        },
                    )
                    # Convert to int if whole number
                    result = float(result)
                    # if result.is_integer():
                    #    result = int(result)
                    cross_referenced_globalVars_dict[key] = result
                except Exception as e:
                    logger.warning(
                        f"Could not evaluate globalVars expression for {key}: {expr}"
                    )

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
                    # if val.is_integer():
                    #    val = int(val)
                    globalVars_dict[key] = val
                except ValueError:
                    globalVars_dict[key] = value
    if cross_ref:
        # Remove the #calc stuff
        globalVars_dict = _cross_reference_global_vars(globalVars_dict)

    return globalVars_dict


def read_surface_tension(case_folder: str) -> float:
    """Surface tension [N/m] from constant/phaseProperties.

    Resolves a ``$var`` reference (e.g. ``sigma $sigmaLiq;``) against globalVars.
    """
    phase_properties = read_openfoam_dict(
        os.path.join(case_folder, "constant", "phaseProperties")
    )
    sigma = None
    for entry in phase_properties["surfaceTension"].values():
        if isinstance(entry, dict) and "sigma" in entry:
            sigma = entry["sigma"]
            break
    if sigma is None:
        raise KeyError("No sigma found in surfaceTension of phaseProperties")

    if isinstance(sigma, str) and sigma.startswith("$"):
        return float(read_global_vars(case_folder, cross_ref=True)[sigma[1:]])
    return float(sigma)
