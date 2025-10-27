#Compile solver

conda activate bird
BIRD_HOME=`python -c "import bird; print(bird.BIRD_DIR)"`
cd ${BIRD_HOME}/../OFsolvers/birdmultiphaseEulerFoam
export WM_COMPILE_OPTION=Debug
./Allwmake
cd ../../

# Run all tests

## Run deckwer17 PBE
cd experimental_cases/deckwer17
bash run.sh 
cd ../../
## Run deckwer17 constantD
cd experimental_cases/deckwer17
cp constant/phaseProperties_constantd constant/phaseProperties
bash run.sh 
cd ../../
## Run deckwer19 PBE
cd experimental_cases/deckwer19
bash run.sh 
cd ../../
## Run side sparger tutorial
cd tutorial_cases/side_sparger
bash run.sh 
cd ../../
## Run bubble column tutorial
cd tutorial_cases/bubble_column_20L
bash run.sh 
cd ../../
## Run stirred-tank tutorial
cd tutorial_cases/stirred_tank
bash run.sh 
cd ../../
## Run reactive loop reactor tutorial
cd tutorial_cases/loop_reactor_reacting
bash run.sh 
cd ../../
## Run mixing loop reactor tutorial
cd tutorial_cases/loop_reactor_mixing
bash run.sh 
cd ../../
## Run airlift reactor tutorial
cd tutorial_cases/airlift_40m
bash run.sh
cd ../../
## Run flat panel reactor tutorial
cd tutorial_cases/FlatPanel_250L_ASU
bash run.sh
cd ../../

