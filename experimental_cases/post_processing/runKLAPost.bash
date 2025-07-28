#!/bin/bash

# File to copy
FILE_TO_COPY="writeKlA"

# Loop over all subdirectories
for dir in */ ; do
    if [ -d "$dir" ]; then
        cp "$FILE_TO_COPY" "$dir/system/$FILE_TO_COPY"
        cd $dir
        timedirs=($(ls -1d [0-9]* | sort -n | tail -19))
        starttime="${timedirs[2]}:" 
        echo $starttime
        rm kLA.csv
        foamPostProcess -time $starttime -func $FILE_TO_COPY
        cd ..
    fi
done

