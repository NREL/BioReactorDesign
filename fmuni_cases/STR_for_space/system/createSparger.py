import math

def addSurface(surface, i, cx, cy, cz, e1x, e1y, e1z, e2x, e2y, e2z, e3x, e3y, e3z):
    with open("sparger", "a", encoding="utf-8") as file:
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
        
        
sparger_cyl_max = 0.1
sparger_R = 0.1
sparger_Rreal = 0.025
sparger_xpos = 0.75

with open("sparger", "w", encoding="utf-8") as file:
    file.write( "\n")

for i in range(6):
    angle = i * 2 * math.pi / 6.0
    e1x = math.cos(angle)
    e1y = math.sin(angle)
    e2x = -math.sin(angle)
    e2y = math.cos(angle)
    xpos = sparger_xpos * e1x 
    ypos = sparger_xpos * e1y 
    with open("sparger", "a", encoding="utf-8") as file:
        file.write(f"sparger_{i}\n")
        file.write( "{\n")
        file.write( "   type cylinder;\n")
        file.write(f"   point1 ( {xpos} {ypos} -10 );\n")
        file.write(f"   point2 ( {xpos} {ypos} {sparger_cyl_max} );\n")
        file.write(f"   radius {sparger_R};\n")
        file.write( "}\n")

with open("spargerPatch", "w", encoding="utf-8") as file:
    file.write( "\n")


for i in range(6):
    angle = i * 2 * math.pi / 6.0
    e1x = math.cos(angle)
    e1y = math.sin(angle)
    e2x = -math.sin(angle)
    e2y = math.cos(angle)
    xpos = sparger_xpos * e1x 
    ypos = sparger_xpos * e1y 
    with open("spargerPatch", "a", encoding="utf-8") as file:
        file.write(f"sparger_{i}\n")
        file.write( "{\n")
        file.write( "   patchInfo { type patch; };\n")
        file.write( "   constructFrom zone;\n")
        file.write( "   zone {\n")
        file.write( "   type cylinder;\n")
        file.write(f"   point1 ( {xpos} {ypos} -10 );\n")
        file.write(f"   point2 ( {xpos} {ypos} 0.001 );\n")
        file.write(f"   radius {sparger_Rreal};\n")
        file.write( "   }\n")
        file.write( "}\n")


    
