import json
from pathlib import Path
from typing import Callable

SCRIPT_DIR = Path(__file__).parent
STUDY_DIR = SCRIPT_DIR / "data" / "study"

# QoI key -> LaTeX row label. Order matches QOI_COLS in get_optimal_all_qois.py.
QOI_LABELS = {
    "qoi": r"$\rm{QOI}_1 (3.6\, \rm{L}~/~608\, \rm{m}^3)$",
    "qoi_kla": r"$\rm{QOI}_2 (3.6\, \rm{L}~/~608\, \rm{m}^3)$",
    "qoi_sum": r"$\rm{QOI}_3 (3.6\, \rm{L}~/~608\, \rm{m}^3)$",
    "qoi_sum_kla": r"$\rm{QOI}_4 (3.6\, \rm{L}~/~608\, \rm{m}^3)$",
}

# Surrogate label -> key inside the hyperparams JSON.
SURROGATE_COLUMNS = {"RBF": "rbf", "RF": "rf", "MLP": "nn"}

LEVELS = ("lev1", "lev6")


def format_int(value: float) -> str:
    return str(int(value))


def format_float(value: float) -> str:
    """3 significant figures; scientific notation rewritten to LaTeX math."""
    text = f"{value:.3g}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        return rf"{mantissa}\times10^{{{int(exponent)}}}"
    return text


def format_kernel(value: str) -> str:
    return rf"\texttt{{{value.replace('_', r'\_')}}}"


# Per-surrogate column spec: (json param key, LaTeX header, formatter, is_math).
Formatter = Callable[[object], str]
SURROGATE_PARAMS: dict[str, list[tuple[str, str, Formatter, bool]]] = {
    "RBF": [
        ("epsilon", r"$\epsilon$", format_float, True),
        ("kernel", "Kernel", format_kernel, False),
    ],
    "RF": [
        ("n_estimators", r"$n_{\rm trees}$", format_int, True),
        ("max_depth", "depth", format_int, True),
        ("min_samples_leaf", "min. leaf", format_int, True),
        ("max_features", "max. feat.", format_float, True),
    ],
    "MLP": [
        ("n_units", "units", format_int, True),
        ("n_layers", "layers", format_int, True),
        ("lr", "lr", format_float, True),
    ],
}


def read_params(level: str, qoi_key: str, model_key: str) -> dict:
    """Optimal hyperparameters for one level/QoI/surrogate."""
    json_path = (
        STUDY_DIR
        / f"study_0_4vvm_{level}"
        / qoi_key
        / f"hyperparams_{qoi_key}.json"
    )
    with open(json_path) as json_file:
        hyperparams = json.load(json_file)
    return hyperparams[model_key]["params"]


def build_cell(lev1_text: str, lev6_text: str, is_math: bool) -> str:
    inner = f"{lev1_text} / {lev6_text}"
    body = f"${inner}$" if is_math else inner
    return rf"\rev{{{body}}}"


def build_table(surrogate_label: str) -> str:
    model_key = SURROGATE_COLUMNS[surrogate_label]
    columns = SURROGATE_PARAMS[surrogate_label]
    headers = [header for _, header, _, _ in columns]

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{ |l|" + "c|" * len(headers) + " }",
        r"\hline",
        r"\rev{QoI} & "
        + " & ".join(rf"\rev{{{header}}}" for header in headers)
        + r" \\",
        r"\hline",
    ]

    for qoi_key, row_label in QOI_LABELS.items():
        lev1_params = read_params("lev1", qoi_key, model_key)
        lev6_params = read_params("lev6", qoi_key, model_key)
        cells = [
            build_cell(
                formatter(lev1_params[param_key]),
                formatter(lev6_params[param_key]),
                is_math,
            )
            for param_key, _, formatter, is_math in columns
        ]
        lines.append(
            rf"\rev{{{row_label}}} & " + " & ".join(cells) + r" \\ \hline"
        )

    lines += [
        r"\end{tabular}",
        rf"\caption{{\rev{{Optimal {surrogate_label} hyperparameters for "
        r"each QoI, at reactor scales $3.6\,$L (lev1) $/$ $608\,$m$^3$ "
        r"(lev6).}}",
        rf"\label{{tab:hyperparams_{model_key}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def write_wrapper(table_file_names: list[str], out_path: Path) -> None:
    """Standalone document that defines \\rev (red) and inputs each table."""
    inputs = [rf"\input{{{name}}}" for name in table_file_names]
    wrapper = "\n".join(
        [
            r"\documentclass{article}",
            r"\usepackage[margin=1in]{geometry}",
            r"\usepackage{xcolor}",
            r"\newcommand{\rev}[1]{\textcolor{red}{#1}}",
            r"\begin{document}",
            *inputs,
            r"\end{document}",
        ]
    )
    with open(out_path, "w") as wrapper_file:
        wrapper_file.write(wrapper)
    print(f"Wrote {out_path}")


def main() -> None:
    table_file_names = []
    for surrogate_label, model_key in SURROGATE_COLUMNS.items():
        table = build_table(surrogate_label)
        print(table)
        print()

        table_file_name = f"hyperparams_{model_key}_table.tex"
        table_path = SCRIPT_DIR / table_file_name
        with open(table_path, "w") as table_file:
            table_file.write(table)
        print(f"Wrote {table_path}")
        table_file_names.append(table_file_name)

    write_wrapper(table_file_names, SCRIPT_DIR / "view_hyperparam_tables.tex")


if __name__ == "__main__":
    main()
