import json


def generate_stl_patch(filename, bc_dict, geom_dict):
    final_dict = {}
    final_dict["Geometry"] = geom_dict["Geometry"]
    for patch in bc_dict:
        final_dict[patch] = bc_dict[patch]
    with open(filename, "w+") as f:
        json.dump(final_dict, f, indent=2)


def generate_dynamic_mixer(filename, mixers_list, geom_dict):
    final_dict = {}
    final_dict["Meshing"] = geom_dict["Meshing"]
    final_dict["Geometry"] = geom_dict["Geometry"]
    final_dict["mixers"] = mixers_list
    with open(filename, "w+") as f:
        json.dump(final_dict, f, indent=2)


def make_default_geom_dict_from_file(filename, rescale=2.7615275385627096):
    with open(filename, "r+") as f:
        geom_dict = json.load(f)
    if "rescale" not in geom_dict["Geometry"]["OverallDomain"]["x"]:
        geom_dict["Geometry"]["OverallDomain"]["x"]["rescale"] = rescale
        geom_dict["Geometry"]["OverallDomain"]["y"]["rescale"] = rescale
        geom_dict["Geometry"]["OverallDomain"]["z"]["rescale"] = rescale
    assert "Meshing" in geom_dict
    assert "Geometry" in geom_dict
    return geom_dict
