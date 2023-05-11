## Generate STL of spider sparger

### Execute without plotting

`python main.py -cr 0.25 -na 12 -aw 0.1 -al 0.5`

### Execute with plotting

`python main.py -v -cr 0.25 -na 12 -aw 0.1 -al 0.5`

Generates

<p float="left">
  <img src="../image/simpleOutput.png" width="250"/>
</p>


### Manual

```
usage: main.py [-h] [-cr] [-na] [-aw] [-al] [-v]

Generate Spider Sparger STL

optional arguments:
  -h, --help            show this help message and exit
  -cr , --centerRadius 
                        Radius of the center distributor
  -na , --nArms         Number of spider arms
  -aw , --armsWidth     Width of spider arms
  -al , --armsLength    Length of spider arms
  -v, --verbose         plot on screen

```
