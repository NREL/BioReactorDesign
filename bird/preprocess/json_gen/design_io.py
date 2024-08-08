import json


def generate_stl_patch(filename, bc_dict, geom_dict):
    final_dict = {}
    final_dict['Geometry'] = geom_dict['Geometry']
    for patch in bc_dict:
        final_dict[patch] = bc_dict[patch]
    with open(filename, 'w+') as f:
        json.dump(final_dict,f,indent=2)

def generate_dynamic_mixer(filename, mixers_list, geom_dict):
    final_dict = {}
    final_dict['Meshing'] = geom_dict['Meshing']
    final_dict['Geometry'] = geom_dict['Geometry']
    final_dict['mixers'] = mixers_list
    with open(filename, 'w+') as f:
       json.dump(final_dict, f, indent=2)

def make_default_geom_dict_from_file(filename):
    with open(filename, "r+") as f:
        geom_dict = json.load(f)
    if "rescale" not in geom_dict["Geometry"]["OverallDomain"]["x"]:
        geom_dict["Geometry"]["OverallDomain"]["x"]["rescale"] = 2.7615275385627096
        geom_dict["Geometry"]["OverallDomain"]["y"]["rescale"] = 2.7615275385627096
        geom_dict["Geometry"]["OverallDomain"]["z"]["rescale"] = 2.7615275385627096
    assert "Meshing" in geom_dict
    assert "Geometry" in geom_dict
    return geom_dict

if __name__ == "__main__":
    bc_dict = {}
    bc_dict["inlets"]=[]
    bc_dict["outlets"]=[]
    tmp_dict = {}
    tmp_dict["type"] = "circle"
    tmp_dict["centx"] = 5.0
    tmp_dict["centy"] = 0.0
    tmp_dict["centz"] = 0.5
    tmp_dict["radius"] = 0.4
    tmp_dict["normal_dir"] = 1
    tmp_dict["nelements"] = 50
    bc_dict["inlets"].append(tmp_dict)
    tmp_dict = {}
    tmp_dict["type"] = "circle"
    tmp_dict["centx"] = 2.5
    tmp_dict["centy"] = 0.0
    tmp_dict["centz"] = 0.5
    tmp_dict["radius"] = 0.4
    tmp_dict["normal_dir"] = 1
    tmp_dict["nelements"] = 50
    bc_dict["inlets"].append(tmp_dict)
    tmp_dict = {}
    tmp_dict["type"] = "circle"
    tmp_dict["centx"] = 7.5
    tmp_dict["centy"] = 0.0
    tmp_dict["centz"] = 0.5
    tmp_dict["radius"] = 0.4
    tmp_dict["normal_dir"] = 1
    tmp_dict["nelements"] = 50
    bc_dict["inlets"].append(tmp_dict)
    
    tmp_dict = {}
    tmp_dict["type"] = "circle"
    tmp_dict["centx"] = 0.5
    tmp_dict["centy"] = 5.0
    tmp_dict["centz"] = 0.5
    tmp_dict["radius"] = 0.4
    tmp_dict["normal_dir"] = 1
    tmp_dict["nelements"] = 50
    bc_dict["outlets"].append(tmp_dict)
    tmp_dict = {}
    tmp_dict["type"] = "circle"
    tmp_dict["centx"] = 0.5
    tmp_dict["centy"] = 5.0
    tmp_dict["centz"] = 0.5
    tmp_dict["radius"] = 0.4
    tmp_dict["normal_dir"] = 1
    tmp_dict["nelements"] = 50
    bc_dict["outlets"].append(tmp_dict)

    generate_stl_patch('test.json', bc_dict)
