import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
STUDY_DIR = SCRIPT_DIR / "data" / "study"

# QoI key -> LaTeX row label. Order matches QOI_COLS in get_optimal_all_qois.py.
QOI_LABELS = {
    "qoi": r"$\rm{QOI}_1 (3.6\, \rm{L}~/~608\, \rm{m}^3)$",
    "qoi_kla": r"$\rm{QOI}_2 (3.6\, \rm{L}~/~608\, \rm{m}^3)$",
    "qoi_sum": r"$\rm{QOI}_3 (3.6\, \rm{L}~/~608\, \rm{m}^3)$",
    "qoi_sum_kla": r"$\rm{QOI}_4 (3.6\, \rm{L}~/~608\, \rm{m}^3)$",
}

# Surrogate column header -> key inside the hyperparams JSON.
SURROGATE_COLUMNS = {"RBF": "rbf", "RF": "rf", "MLP": "nn"}

LEVELS = ("lev1", "lev6")
METRIC_KEY = "val_nrmse_amp_pct"


def read_val_nrmse(level: str, qoi_key: str) -> dict[str, float]:
    """Validation nRMSE (% of amplitude) for each surrogate at one level/QoI."""
    json_path = (
        STUDY_DIR
        / f"study_0_4vvm_{level}"
        / qoi_key
        / f"hyperparams_{qoi_key}.json"
    )
    with open(json_path) as json_file:
        hyperparams = json.load(json_file)
    return {
        column: float(hyperparams[model_key][METRIC_KEY])
        for column, model_key in SURROGATE_COLUMNS.items()
    }


def format_number(value: float, is_lowest: bool) -> str:
    text = f"{value:.1f}"
    return rf"\textbf{{{text}}}" if is_lowest else text


def build_cell(
    lev1_value: float,
    lev6_value: float,
    lev1_is_lowest: bool,
    lev6_is_lowest: bool,
) -> str:
    lev1_text = format_number(lev1_value, lev1_is_lowest)
    lev6_text = format_number(lev6_value, lev6_is_lowest)
    return rf"${lev1_text}\% / {lev6_text}\%$"


def build_table() -> str:
    column_headers = list(SURROGATE_COLUMNS)
    lines = [
        r"\begin{tabular}{l" + "c" * len(column_headers) + "}",
        r"\hline",
        "QoI & " + " & ".join(column_headers) + r" \\",
        r"\hline",
    ]

    for qoi_key, row_label in QOI_LABELS.items():
        lev1_values = read_val_nrmse("lev1", qoi_key)
        lev6_values = read_val_nrmse("lev6", qoi_key)
        lev1_lowest_column = min(lev1_values, key=lev1_values.get)
        lev6_lowest_column = min(lev6_values, key=lev6_values.get)

        cells = [
            build_cell(
                lev1_values[column],
                lev6_values[column],
                column == lev1_lowest_column,
                column == lev6_lowest_column,
            )
            for column in column_headers
        ]
        lines.append(f"{row_label} & " + " & ".join(cells) + r" \\")

    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines)


def write_wrapper(table_file_name: str, out_path: Path) -> None:
    wrapper = "\n".join(
        [
            r"\documentclass{article}",
            r"\usepackage[margin=1in]{geometry}",
            r"\begin{document}",
            r"\begin{table}[h]",
            r"\centering",
            rf"\input{{{table_file_name}}}",
            r"\caption{Validation nRMSE relative to amplitude "
            r"(lev1 $3.6\,$L / lev6 $608\,$m$^3$); lowest per level in bold.}",
            r"\end{table}",
            r"\end{document}",
        ]
    )
    with open(out_path, "w") as wrapper_file:
        wrapper_file.write(wrapper)
    print(f"Wrote {out_path}")


def main() -> None:
    table = build_table()
    print(table)

    table_path = SCRIPT_DIR / "val_rmse_table.tex"
    with open(table_path, "w") as table_file:
        table_file.write(table)
    print(f"Wrote {table_path}")

    write_wrapper("val_rmse_table.tex", SCRIPT_DIR / "view_val_rmse_table.tex")


if __name__ == "__main__":
    main()
