#!/bin/bash

# Cylinder parameters
RADIUS=1.0  # Radius of the cylinder
HEIGHT=1.0 # Height of the cylinder
SEGMENTS=36 # Number of segments for the circle approximation
OUTPUT_FILE="cylinder.stl"

# Helper function to calculate coordinates
calc_coords() {
  local angle=$(echo "2 * $1 * 3.141592653589793 / $SEGMENTS" | bc -l)
  local x=$(echo "$RADIUS * c($angle)" | bc -l)
  local y=$(echo "$RADIUS * s($angle)" | bc -l)
  echo "$x $y"
}

# Start the STL file
echo "solid cylinder" > $OUTPUT_FILE

# Generate the cylinder top and bottom faces
for ((i=0; i<$SEGMENTS; i++)); do
  next=$(( (i + 1) % SEGMENTS ))
  coords1=$(calc_coords $i)
  coords2=$(calc_coords $next)

  # Top face (z = HEIGHT)
  echo "  facet normal 0 0 1" >> $OUTPUT_FILE
  echo "    outer loop" >> $OUTPUT_FILE
  echo "      vertex 0 0 $HEIGHT" >> $OUTPUT_FILE
  echo "      vertex $coords1 $HEIGHT" >> $OUTPUT_FILE
  echo "      vertex $coords2 $HEIGHT" >> $OUTPUT_FILE
  echo "    endloop" >> $OUTPUT_FILE
  echo "  endfacet" >> $OUTPUT_FILE

  # Bottom face (z = 0)
  echo "  facet normal 0 0 -1" >> $OUTPUT_FILE
  echo "    outer loop" >> $OUTPUT_FILE
  echo "      vertex 0 0 0" >> $OUTPUT_FILE
  echo "      vertex $coords2 0" >> $OUTPUT_FILE
  echo "      vertex $coords1 0" >> $OUTPUT_FILE
  echo "    endloop" >> $OUTPUT_FILE
  echo "  endfacet" >> $OUTPUT_FILE

  # Side faces
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

# End the STL file
echo "endsolid cylinder" >> $OUTPUT_FILE

echo "Cylinder STL file generated: $OUTPUT_FILE"

