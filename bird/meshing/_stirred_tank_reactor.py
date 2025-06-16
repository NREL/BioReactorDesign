import os
from pathlib import Path

import numpy as np

from bird.meshing._mesh_tools import parseYAMLFile


class StirredTankReactor:
    def __init__(
        self,
        Dt,
        Da,
        H,
        nimpellers,
        C,
        W,
        L,
        Lin,
        J,
        Wh,
        polyrad,
        Z0,
        nr,
        nz,
        Npoly,
        Na,
        nbaffles,
    ):
        # Loop through params and setattr v to self.k
        for k, v in locals().items():
            if k != "self":
                setattr(self, k, v)
        self.Dh = Da - 2 * L
        self.Dmrf = (Da + Dt - 2 * J) / 2
        self.nsplits = 2 * nbaffles  # we need twice the number of splits
        self.dangle = 2.0 * np.pi / float(self.nsplits)

        self.circradii = np.array(
            [
                self.Dh / 2 - Lin,
                self.Dh / 2,
                Da / 2,
                self.Dmrf / 2,
                Dt / 2 - J,
                Dt / 2,
            ]
        )
        self.ncirc = len(self.circradii)
        self.hub_circ = 1
        self.inhub_circ = self.hub_circ - 1  # circle inside hub
        self.rot_circ = self.hub_circ + 1
        self.mrf_circ = self.rot_circ + 1
        self.tank_circ = self.ncirc - 1

        self.reacthts = [Z0]
        self.baff_sections = []
        self.baff_volumes = []
        self.hub_volumes = []
        count = 1
        for n_imp in range(self.nimpellers):
            self.reacthts.append(Z0 + C[n_imp] - W / 2)

            self.baff_sections.append(count)
            self.baff_volumes.append(count)
            count = count + 1

            self.reacthts.append(Z0 + C[n_imp] - Wh / 2)

            self.baff_sections.append(count)
            self.baff_volumes.append(count)
            self.hub_volumes.append(count)
            count = count + 1

            self.reacthts.append(Z0 + C[n_imp] + Wh / 2)

            self.baff_sections.append(count)
            self.baff_volumes.append(count)
            count = count + 1

            self.reacthts.append(Z0 + C[n_imp] + W / 2)
            self.baff_sections.append(count)
            count = count + 1

        self.reacthts.append(Z0 + H)

        self.nsections = len(self.reacthts)
        self.nvolumes = self.nsections - 1
        self.meshz = nz * np.diff(self.reacthts)
        self.meshz = self.meshz.astype(int) + 1  # avoid zero mesh elements

        self.all_volumes = range(self.nvolumes)
        self.nonbaff_volumes = [
            sec for sec in self.all_volumes if sec not in self.baff_volumes
        ]
        self.nonstem_volumes = [
            0,
            1,
        ]  # this is 0,1 no matter how many impellers are there

        # note: stem_volumes include hub volumes also
        # these are volumes where we miss out polygon block
        self.stem_volumes = [
            sec for sec in self.all_volumes if sec not in self.nonstem_volumes
        ]

        # removes hub_volumes here for declaring patches
        self.only_stem_volumes = [
            sec for sec in self.stem_volumes if sec not in self.hub_volumes
        ]

        # to define mrf region
        # not that [1] is not a stem volume but baffles are there
        self.mrf_volumes = [1] + self.stem_volumes

        # increase grid points in the impeller section
        for i in self.baff_volumes:
            self.meshz[i] *= 2

        self.meshr = nr * np.diff(self.circradii)

        # adding polygon to hub mesh resolution
        self.meshr = np.append(nr * polyrad, self.meshr)
        self.meshr = self.meshr.astype(int)
        self.meshr += 1  # to avoid being zero

        self.centeroffset = 1  # one point on the axis
        self.polyoffset = self.nsplits  # number of points on polygon
        self.npts_per_section = (
            self.centeroffset + self.polyoffset + self.ncirc * self.nsplits
        )  # center+polygon+circles

    @classmethod
    def from_file(cls, yamlfile):
        if ".yaml" not in yamlfile:
            yamlfile += ".yaml"
        in_dict = parseYAMLFile(yamlfile)
        react_dict = {**in_dict["geometry"], **in_dict["mesh"]}
        return cls(**react_dict)
