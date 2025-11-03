#!/bin/bash

# Cylinder parameters
RADIUS=0.03175  # Outer radius of the cylinder
INNER_RADIUS=0.005  # Inner circular face radius
r=0.05  # Radius of the coil
PI=3.14159265359  # Pi value
NCOILS=3  # Number of coils
HEIGHT=0.3  # Cylinder height
NR=12
NZ=$(echo "0.5 * $NR * $HEIGHT / $RADIUS" | bc -l)
SEGMENTS=36  # Circle segmentation
OUTPUT_FILE="constant/triSurface/cylinder.stl"

RTUBE=0.0025
PITCH=$(echo "2 * $RTUBE" | bc -l)

# Helper function to calculate coordinates
calc_coords() {
  local radius=$1
  local angle=$(echo "2 * $2 * $PI / $SEGMENTS" | bc -l)
  local x=$(echo "$radius * c($angle)" | bc -l)
  local y=$(echo "$radius * s($angle)" | bc -l)
  echo "$x $y"
}

# Start the STL file
echo "solid cylinder_top" > $OUTPUT_FILE

# Generate the cylinder top face
for ((i=0; i<$SEGMENTS; i++)); do
  next=$(( (i + 1) % SEGMENTS ))
  coords1=$(calc_coords $RADIUS $i)
  coords2=$(calc_coords $RADIUS $next)

  # Top face (z = HEIGHT)
  echo "  facet normal 0 0 1" >> $OUTPUT_FILE
  echo "    outer loop" >> $OUTPUT_FILE
  echo "      vertex 0 0 $HEIGHT" >> $OUTPUT_FILE
  echo "      vertex $coords1 $HEIGHT" >> $OUTPUT_FILE
  echo "      vertex $coords2 $HEIGHT" >> $OUTPUT_FILE
  echo "    endloop" >> $OUTPUT_FILE
  echo "  endfacet" >> $OUTPUT_FILE
done

# End the top face region
echo "endsolid cylinder_top" >> $OUTPUT_FILE

# Start the corona region
echo "solid corona" >> $OUTPUT_FILE

# Generate the corona region on bottom face (between INNER_RADIUS and RADIUS)
for ((i=0; i<$SEGMENTS; i++)); do
  next=$(( (i + 1) % SEGMENTS ))

  # Define inner circle coordinates
  inner_coords1=$(calc_coords $INNER_RADIUS $i)
  inner_coords2=$(calc_coords $INNER_RADIUS $next)

  # Define outer circle coordinates
  outer_coords1=$(calc_coords $RADIUS $i)
  outer_coords2=$(calc_coords $RADIUS $next)

  # Corona facets (between inner circle and outer circle)
  echo "  facet normal 0 0 -1" >> $OUTPUT_FILE
  echo "    outer loop" >> $OUTPUT_FILE
  echo "      vertex $inner_coords1 0" >> $OUTPUT_FILE
  echo "      vertex $outer_coords1 0" >> $OUTPUT_FILE
  echo "      vertex $outer_coords2 0" >> $OUTPUT_FILE
  echo "    endloop" >> $OUTPUT_FILE
  echo "  endfacet" >> $OUTPUT_FILE

  echo "  facet normal 0 0 -1" >> $OUTPUT_FILE
  echo "    outer loop" >> $OUTPUT_FILE
  echo "      vertex $inner_coords1 0" >> $OUTPUT_FILE
  echo "      vertex $outer_coords2 0" >> $OUTPUT_FILE
  echo "      vertex $inner_coords2 0" >> $OUTPUT_FILE
  echo "    endloop" >> $OUTPUT_FILE
  echo "  endfacet" >> $OUTPUT_FILE
done

# End the corona region
echo "endsolid corona" >> $OUTPUT_FILE

# Start the inner circular face region
echo "solid inner_circle" >> $OUTPUT_FILE

# Generate inner circular face (disk at z = 0 with INNER_RADIUS)
for ((i=0; i<$SEGMENTS; i++)); do
  next=$(( (i + 1) % SEGMENTS ))
  inner_coords1=$(calc_coords $INNER_RADIUS $i)
  inner_coords2=$(calc_coords $INNER_RADIUS $next)

  echo "  facet normal 0 0 -1" >> $OUTPUT_FILE
  echo "    outer loop" >> $OUTPUT_FILE
  echo "      vertex 0 0 0" >> $OUTPUT_FILE
  echo "      vertex $inner_coords2 0" >> $OUTPUT_FILE
  echo "      vertex $inner_coords1 0" >> $OUTPUT_FILE
  echo "    endloop" >> $OUTPUT_FILE
  echo "  endfacet" >> $OUTPUT_FILE
done

# End the inner circular face region
echo "endsolid inner_circle" >> $OUTPUT_FILE

# Start the cylinder sides region
echo "solid cylinder_sides" >> $OUTPUT_FILE

# Generate the cylinder side faces
for ((i=0; i<$SEGMENTS; i++)); do
  next=$(( (i + 1) % SEGMENTS ))
  coords1=$(calc_coords $RADIUS $i)
  coords2=$(calc_coords $RADIUS $next)

  # Side facets
  echo "  facet normal 0 0 0" >> $OUTPUT_FILE
  echo "    outer loop" >> $OUTPUT_FILE
  echo "      vertex $coords1 0" >> $OUTPUT_FILE
  echo "      vertex $coords2 0" >> $OUTPUT_FILE
  echo "      vertex $coords2 $HEIGHT" >> $OUTPUT_FILE
  echo "    endloop" >> $OUTPUT_FILE
  echo "  endfacet" >> $OUTPUT_FILE

  echo "  facet normal 0 0 0" >> $OUTPUT_FILE
  echo "    outer loop" >> $OUTPUT_FILE
  echo "      vertex $coords1 0" >> $OUTPUT_FILE
  echo "      vertex $coords2 $HEIGHT" >> $OUTPUT_FILE
  echo "      vertex $coords1 $HEIGHT" >> $OUTPUT_FILE
  echo "    endloop" >> $OUTPUT_FILE
  echo "  endfacet" >> $OUTPUT_FILE
done

# End the sides region
echo "endsolid cylinder_sides" >> $OUTPUT_FILE

echo "Cylinder STL file with regions generated: $OUTPUT_FILE"

# Create input file for blockMeshDict
echo "x0 -$RADIUS;" > "system/meshParams.H"
echo "x1 $RADIUS;" >> "system/meshParams.H"
echo "y0 -$RADIUS;" >> "system/meshParams.H"
echo "y1 $RADIUS;" >> "system/meshParams.H"
echo "z0 0.000001;" >> "system/meshParams.H"
echo "z1 $HEIGHT;" >> "system/meshParams.H"
echo "nx $NR;" >> "system/meshParams.H"
echo "ny $NR;" >> "system/meshParams.H"
echo "nz #calc \" ceil($NZ) \";" >> "system/meshParams.H"
echo "R $r;" >> "system/meshParams.H"
echo "p $PITCH;" >> "system/meshParams.H"

