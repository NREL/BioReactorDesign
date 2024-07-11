import numpy as np


class Mixer:
    def __init__(self):
        self.x = None
        self.y = None
        self.z = None
        self.rad = 0.015
        self.power = 300
        self.start_time = 1.0
        self.normal_dir = None
        self.ready = False

    def update_from_dict(self, mixer_dict):
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
        if "start_time" in mixer_dict:
            self.start_time = mixer_dict["start_time"]
        if "normal_dir" in mixer_dict:
            self.normal_dir = mixer_dict["normal_dir"]
        self.check_status()

    def check_status(self):
        if (
            self.x is None
            or self.y is None
            or self.z is None
            or self.normal_dir is None
        ):
            self.ready = False
        else:
            self.ready = True
