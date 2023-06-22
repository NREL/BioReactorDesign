post_conditional_mean () {

    postProcess  -func writeCellVolumes -time $2 -case $1
    writeMeshObj -case $1
}

for d in BC_2850L/Case*/ ; do
    echo Doing $d
    post_conditional_mean $d 0
done
