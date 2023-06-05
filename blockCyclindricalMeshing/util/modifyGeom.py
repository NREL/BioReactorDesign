import json
import os


def modify_multiring(width, spacing, template_folder, target_folder):
    
    with open(os.path.join(template_folder,'input.json'), 'r+') as f:
        data_input = json.load(f)
    with open(os.path.join(template_folder,'topology.json'), 'r+') as f:
        data_topo = json.load(f)
    
    
    os.makedirs(target_folder, exist_ok=True)
    with open(os.path.join(target_folder, 'topology.json'), 'w+') as f:
        json.dump(data_topo, f, indent=4)
    with open(os.path.join(target_folder, 'input.json'), 'w+') as f:
        ring_width = width
        ring_spacing = spacing
        r_outterRing = data_input['Geometry']['Radial']["outter_ring5"]
        data_input['Geometry']['Radial']["inner_ring5"] = r_outterRing - ring_width
        data_input['Geometry']['Radial']["outter_ring4"] = r_outterRing - ring_width - ring_spacing
        data_input['Geometry']['Radial']["inner_ring4"] = r_outterRing - 2*ring_width - ring_spacing
        json.dump(data_input, f, indent=4)

def modify_flatDonut(width, template_folder, target_folder):

    with open(os.path.join(template_folder,'input.json'), 'r+') as f:
        data_input = json.load(f)
    with open(os.path.join(template_folder,'topology.json'), 'r+') as f:
        data_topo = json.load(f)
   
   
    os.makedirs(target_folder, exist_ok=True)
    with open(os.path.join(target_folder, 'topology.json'), 'w+') as f:
        json.dump(data_topo, f, indent=4)
    with open(os.path.join(target_folder, 'input.json'), 'w+') as f:
        r_outter = data_input['Geometry']['Radial']["outter_sparger"]
        data_input['Geometry']['Radial']["inner_sparger"] = r_outter - width
        json.dump(data_input, f, indent=4)


def modify_sideSparger(height, template_folder, target_folder):

    with open(os.path.join(template_folder,'input.json'), 'r+') as f:
        data_input = json.load(f)
    with open(os.path.join(template_folder,'topology.json'), 'r+') as f:
        data_topo = json.load(f)
  
    os.makedirs(target_folder, exist_ok=True)
    with open(os.path.join(target_folder, 'topology.json'), 'w+') as f:
        json.dump(data_topo, f, indent=4)
    with open(os.path.join(target_folder, 'input.json'), 'w+') as f:
        data_input['Geometry']['Longitudinal']["sparger_top"] = 500 + height/2
        data_input['Geometry']['Longitudinal']["sparger_bottom"] = 500 - height/2
        data_input['Meshing']['NVertSmallest'] = max(round(1*height/20), 1)
        vertCoarse = 0.7*(height/data_input['Meshing']['NVertSmallest'])/(20/2) 
        data_input["Meshing"]['verticalCoarsening'][2]['ratio'] = vertCoarse 
        data_input["Meshing"]['verticalCoarsening'][4]['ratio'] = vertCoarse
        json.dump(data_input, f, indent=4)


if __name__=="__main__":
    modify_multiring(width=32, spacing=100, template_folder='template_multiRing', target_folder='testMR')
    #modify_flatDonut(width=32, template_folder='template_flatDonut', target_folder='testMR')
    #modify_sideSparger(height=100, template_folder='template_sideSparger', target_folder='testMR')
