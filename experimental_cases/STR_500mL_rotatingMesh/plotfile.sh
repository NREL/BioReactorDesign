#!/bin/bash

# Check input
if [ $# -lt 2 ]; then
    echo "Usage: ./plotfile.sh <filename.csv> <y_column_index>"
    exit 1
fi

FILENAME="$1"
YCOLUMN="$2"

# Create a temporary gnuplot script
GNUPLOT_SCRIPT=$(mktemp)

cat <<EOF > "$GNUPLOT_SCRIPT"
set datafile separator ","
set xlabel "Column 1"
set ylabel "Column $YCOLUMN"
set key off
plot "$FILENAME" using 1:$YCOLUMN with linespoints title "Col $YCOLUMN vs Col 1"
EOF

# Run gnuplot
gnuplot -p "$GNUPLOT_SCRIPT"

# Clean up
rm "$GNUPLOT_SCRIPT"

