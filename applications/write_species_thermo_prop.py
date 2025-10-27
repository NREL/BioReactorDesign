import argparse

from bird.preprocess.species_gen.setup_thermo_prop import (
    write_species_properties,
)


def main():
    parser = argparse.ArgumentParser(
        description="Write species thermo properties"
    )
    parser.add_argument(
        "-cf",
        "--case_folder",
        type=str,
        metavar="",
        required=True,
        help="Case folder path",
        default=".",
    )
    args = parser.parse_args()
    write_species_properties(args.case_folder, phase="gas")
    write_species_properties(args.case_folder, phase="liquid")


if __name__ == "__main__":
    main()
