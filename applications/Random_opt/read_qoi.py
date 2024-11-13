import argparse
import os

import numpy as np


def count_sim(dirlist):
    count = 0
    for entry in dirlist:
        if entry.startswith("Sim"):
            count += 1
    return count


def read_qoi(case_folder):
    filen = os.path.join(case_folder, "qoi.txt")
    if os.path.isfile(filen):
        with open(filen, "r+") as f:
            try:
                line = f.readline()
                qois = line.split(",")
            except:
                qois = [np.nan, np.nan]
        return float(qois[0]), float(qois[1])
    else:
        return np.nan, np.nan


parser = argparse.ArgumentParser(description="QOI reader")
parser.add_argument(
    "-bf",
    "--batch_folder",
    type=str,
    metavar="",
    required=True,
    help="Folder that contains the batch of sims",
)
args, unknown = parser.parse_known_args()


dir_list = os.listdir(args.batch_folder)
n_sim = count_sim(dir_list)
qoi = np.zeros(n_sim)
qoi_err = np.zeros(n_sim)
for isim in range(n_sim):
    case_folder = os.path.join(args.batch_folder, f"Sim_{isim}")
    qoi[isim], qoi_err[isim] = read_qoi(case_folder)

np.savez(
    os.path.join(args.batch_folder, "results.npz"), qoi=qoi, qoi_err=qoi_err
)
