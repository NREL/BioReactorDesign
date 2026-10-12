import os
import re


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
                    # Standard list or dict-like list (e.g. sizeGroups)
                    lst = []
                    dictlist = {}

                    while tokens[index] != ")":
                        label = tokens[index]
                        index += 1

                        if index < len(tokens) and tokens[index] == "{":
                            # Inline dict entry like: f1 { dSph 1e-3; value 0.0; }
                            index += 1
                            subdict, index = parse_block(index)
                            dictlist[label] = subdict
                        else:
                            # Skip semicolons if present (e.g. in lists like species)
                            if label != ";":
                                lst.append(label)

                    index += 1  # skip ')'
                    if index < len(tokens) and tokens[index] == ";":
                        index += 1

                    # Choose dictlist only if it has content; otherwise, use lst
                    result[key] = dictlist if dictlist else lst

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


def read_gravity(case_folder: str) -> float:
    """Gravity magnitude [m/s2] from constant/g; 9.81 if the file is absent."""
    try:
        g_dict = read_openfoam_dict(os.path.join(case_folder, "constant", "g"))
    except FileNotFoundError:
        return 9.81
    gravity_vector = [float(component) for component in g_dict["value"]]
    return sum(component**2 for component in gravity_vector) ** 0.5


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
