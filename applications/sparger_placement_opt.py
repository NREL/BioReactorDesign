import os
import pickle
import shutil

import numpy as np

from bird.preprocess.json_gen.design_io import *
from bird.preprocess.json_gen.generate_designs import *

if __name__ == "__main__":

    generate_single_scaledup_reactor_sparger_cases(
        sparger_locs=[0.3, 0.5, 1.4],
        sim_id=0,
        vvm=0.4,
        study_folder=".",
    )
