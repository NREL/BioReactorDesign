import re
from collections import defaultdict

from ruamel.yaml import YAML

from bird import BIRD_CONST_DIR


def parse_yaml(filename: str):
    yaml = YAML()
    with open(filename, "r+") as f:
        spec = yaml.load(f)

    return spec


def global_vars_to_dict(filename: str):
    with open(filename, "r+") as f:
        lines = f.readlines()
    global_vars_dict = {}
    for line in lines:
        if not line.startswith("//"):
            line_l = line.split()
            name = line_l[0]
            val = line[len(name) + 1 :]
            ind_end = val.index(";")
            val = val[:ind_end] + ";"
            global_vars_dict[name] = val

    return global_vars_dict


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


def write_openfoam_dict(d, indent=0):
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

    return lines


def save_openfoam_dict(foam_dict, path, header=None):
    with open(path, "w") as f:
        if header:
            f.write(header.strip() + "\n\n")
        lines = write_openfoam_dict(foam_dict)
        f.write("\n".join(lines))
        f.write(
            "\n\n// ************************************************************************* //\n"
        )
