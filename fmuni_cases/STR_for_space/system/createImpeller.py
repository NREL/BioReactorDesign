import math

def addSurface(surface, i, cx, cy, cz, e1x, e1y, e1z, e2x, e2y, e2z, e3x, e3y, e3z):
    with open("impeller", "a", encoding="utf-8") as file:
        file.write(f"{surface}_{i}\n")
        file.write("{\n")
        file.write(f"    surface {surface};\n")
        file.write( "    scale (1.0 1.0 1.0);\n")
        file.write( "    transform\n")
        file.write( "    {\n")
        file.write( "       coordinateSystem\n")
        file.write( "       {\n")
        file.write( "           type cartesian;\n")
        file.write(f"           origin ( {cx} {cy} {cz} );\n")
        file.write( "           coordinateRotation\n")
        file.write( "           {\n")
        file.write( "               type axesRotation;\n")
        file.write(f"               e1 ( {e1x} {e1y} {e1z} );\n")
        file.write(f"               e2 ( {e2x} {e2y} {e2z} );\n")
        file.write(f"               e3 ( {e3x} {e3y} {e3z} );\n")
        file.write( "           }\n")
        file.write( "       }\n")

        file.write( "    }\n")
        file.write("}\n")
        
        
shaft_min =   1
shaft_R   =   0.15
disk_h =      0.1
disk_R =      0.5
blade_H =     0.5
blade_L =     0.6
blade_t =     0.05
blade_attack = 0.3

impeller_sep = 1

disk_max = shaft_min + disk_h
disk_mid = (disk_max + shaft_min)/2.0

blade_x0 =  blade_attack
blade_x1 = blade_attack + blade_L


blade_y0 =  - blade_t/2.0
blade_y1 = blade_t/2.0

blade_z0 = disk_mid - blade_H/2.0
blade_z1 = disk_mid + blade_H/2.0

rzone_min = blade_z0 - blade_H/4.0
rzone_max = blade_z1 + blade_H/4.0 + impeller_sep
rzone_R = blade_x1 + blade_L/4.0



with open("impeller", "w", encoding="utf-8") as file:
    file.write( "impeller_rotatingZone\n")
    file.write( "{\n")
    file.write( "   type cylinder;\n")
    file.write(f"   point1 ( 0 0 {rzone_min} );\n")
    file.write(f"   point2 ( 0 0 {rzone_max} );\n")
    file.write(f"   radius {rzone_R};\n")
    file.write( "}\n")
    
with open("impeller", "a", encoding="utf-8") as file:
    file.write( "impeller_shaft\n")
    file.write( "{\n")
    file.write( "   type cylinder;\n")
    file.write(f"   point1 ( 0 0 {shaft_min} );\n")
    file.write(f"   point2 ( 0 0 100 );\n")
    file.write(f"   radius {shaft_R};\n")
    file.write( "}\n")

with open("impeller", "a", encoding="utf-8") as file:
    file.write( "impeller_disk\n")
    file.write( "{\n")
    file.write( "   type cylinder;\n")
    file.write(f"   point1 ( 0 0 {shaft_min} );\n")
    file.write(f"   point2 ( 0 0 {disk_max} );\n")
    file.write(f"   radius {disk_R};\n")
    file.write( "}\n")
    
    
with open("impeller", "a", encoding="utf-8") as file:
    file.write( "impeller_blade\n")
    file.write( "{\n")
    file.write( "   type box;\n")
    file.write(f"   min ( {blade_x0} {blade_y0} {blade_z0} );\n")
    file.write(f"   max ( {blade_x1} {blade_y1} {blade_z1} );\n")
    file.write( "}\n")
    
with open("impeller", "a", encoding="utf-8") as file:
    file.write( "impeller\n")
    file.write( "{\n")
    file.write( "   type collection;\n")
    file.write( "   mergeSubRegions true;\n")
    

addSurface("impeller_disk", 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1 )

for i in range(6):
    angle = i * 2 * math.pi / 6.0
    e1x = math.cos(angle)
    e1y = math.sin(angle)
    e2x = -math.sin(angle)
    e2y = math.cos(angle)
    addSurface("impeller_blade", i, 0, 0, 0, e1x, e1y, 0, e2x, e2y, 0, 0, 0, 1 )

with open("impeller", "a", encoding="utf-8") as file:
    file.write( "}\n")

## Impeller 2    
with open("impeller", "a", encoding="utf-8") as file:
    file.write( "impeller2\n")
    file.write( "{\n")
    file.write( "   type collection;\n")
    file.write( "   mergeSubRegions true;\n")
    

addSurface("impeller_disk", 0, 0, 0, impeller_sep, 1, 0, 0, 0, 1, 0, 0, 0, 1 )

for i in range(6):
    angle = i * 2 * math.pi / 6.0
    e1x = math.cos(angle)
    e1y = math.sin(angle)
    e2x = -math.sin(angle)
    e2y = math.cos(angle)
    addSurface("impeller_blade", i, 0, 0, impeller_sep, e1x, e1y, 0, e2x, e2y, 0, 0, 0, 1 )

with open("impeller", "a", encoding="utf-8") as file:
    file.write( "}\n")
    
    
