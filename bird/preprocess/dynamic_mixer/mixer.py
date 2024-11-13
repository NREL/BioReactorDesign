import numpy as np


class Mixer:
    def __init__(self):
        self.x = None
        self.y = None
        self.z = None
        self.rad = 0.015
        self.power = 300
        self.start_time = 1.0
        self.smear = 3
        self.sign = None
        self.normal_dir = None
        self.ready = False

    def update_from_expl_dict(self, mixer_dict):
        if "x" in mixer_dict:
            self.x = mixer_dict["x"]
        if "y" in mixer_dict:
            self.y = mixer_dict["y"]
        if "z" in mixer_dict:
            self.z = mixer_dict["z"]
        if "rad" in mixer_dict:
            self.rad = mixer_dict["rad"]
        if "power" in mixer_dict:
            self.power = mixer_dict["power"]
        if "sign" in mixer_dict:
            self.sign = mixer_dict["sign"]
        if "smear" in mixer_dict:
            self.smear = mixer_dict["smear"]
        if "start_time" in mixer_dict:
            self.start_time = mixer_dict["start_time"]
        if "normal_dir" in mixer_dict:
            self.normal_dir = mixer_dict["normal_dir"]
        self.check_status()

    def update_from_loop_dict(self, mixer_dict, geom_dict, mesh_dict=None):
        segment = geom_dict["segments"][mixer_dict["branch_id"]]
        pos = segment["start"] + mixer_dict["frac_space"] * segment["conn"]
        self.x = pos[0]
        self.y = pos[1]
        self.z = pos[2]
        self.normal_dir = segment["normal_dir"]
        if "rad" in mixer_dict:
            self.rad = min(mixer_dict["rad"], segment["max_rad"])
        else:
            self.rad = segment["max_rad"] * 0.7
        if "power" in mixer_dict:
            self.power = mixer_dict["power"]
        if "sign" in mixer_dict:
            self.sign = mixer_dict["sign"]
        if "start_time" in mixer_dict:
            self.start_time = mixer_dict["start_time"]
        if "normal_dir" in mixer_dict:
            self.normal_dir = mixer_dict["normal_dir"]
        if mesh_dict is not None:
            if self.normal_dir == 0:
                min_mesh_transv = min(
                    mesh_dict["Blockwise"]["y"], mesh_dict["Blockwise"]["z"]
                )
            elif self.normal_dir == 1:
                min_mesh_transv = min(
                    mesh_dict["Blockwise"]["x"], mesh_dict["Blockwise"]["z"]
                )
            elif self.normal_dir == 2:
                min_mesh_transv = min(
                    mesh_dict["Blockwise"]["x"], mesh_dict["Blockwise"]["y"]
                )
            self.smear = min_mesh_transv // 3
        self.check_status(blocks=segment["blocks"])

    def check_status(self, blocks=None):
        if (
            self.x is None
            or self.y is None
            or self.z is None
            or self.normal_dir is None
            or ((not self.sign == "+") and (not self.sign == "-"))
        ):
            self.ready = False
        else:
            print(
                f"\n\tpos({self.x:.2g}, {self.y:.2g}, {self.z:.2g})"
                + f"\n\tnormal_dir {self.normal_dir}"
                + f"\n\trad {self.rad:.2g}"
                + f"\n\tpower {self.power:.2g}"
                + f"\n\tsign {self.sign}"
                + f"\n\tsmear {self.smear}"
                + f"\n\tstart_time {self.start_time:.2g}"
            )
            if blocks is not None:
                print(f"\tbranch = {blocks}")

            self.ready = True
