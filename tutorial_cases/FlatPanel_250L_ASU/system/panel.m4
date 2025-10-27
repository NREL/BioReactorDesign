//--------------------------------*- C++ -*----------------------------------
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
FoamFile
{
  version  2.0;
  format   ascii;
  class dictionary;
  object blockMeshDict;
}
// ************************************
changecom(//)changequote([,])
define(calc, [esyscmd(perl -e 'printf ($1)')])
define(calcint, [esyscmd(perl -e 'printf int($1)')])
define(VCOUNT, 0)
define(vlabel, [[// ]Vertex $1 = VCOUNT define($1, VCOUNT)define([VCOUNT], incr(VCOUNT))])

convertToMeters 1;
   
define(h, 0.02925) //wall-sparge distance
define(b, 0.005) // sparger diameter
define(L, 1.1176) // total lenght
define(D, 0.0635) // total depth
//define(H, 0.9144) //total height
define(H, 1.1) //total height
define(Lmh, calc(L-h))
define(hpb, calc(h+b))
define(NPX, 50)
define(NPZ, 5)
define(NPY, 50)

vertices
(
    ( 0   0   0 ) vlabel(block0_0)
    ( L   0   0 ) vlabel(block0_1)
    ( L   H   0 ) vlabel(block0_2)
    ( 0   H   0 ) vlabel(block0_3)
    ( h   0   h ) vlabel(block0_4)
    ( Lmh 0   h ) vlabel(block0_5)
    ( Lmh H   h ) vlabel(block0_6)
    ( h   H   h ) vlabel(block0_7)
    ( h   0   hpb) vlabel(block0_8)
    ( Lmh 0   hpb) vlabel(block0_9)
    ( Lmh H   hpb) vlabel(block0_10)
    ( h   H   hpb) vlabel(block0_11)
    ( 0   0   D ) vlabel(block0_12)
    ( L   0   D ) vlabel(block0_13)
    ( L   H   D ) vlabel(block0_14)
    ( 0   H   D ) vlabel(block0_15)

);

blocks
(
    //block 0
    hex ( block0_0 block0_1 block0_2 block0_3 block0_4 block0_5 block0_6 block0_7 ) (NPX NPY NPZ) simpleGrading (1 1 1)
    //block 1
    hex ( block0_5 block0_1 block0_2 block0_6 block0_9 block0_13 block0_14 block0_10) ( NPZ NPY NPZ ) simpleGrading (1 1 1)
    //block 2
    hex ( block0_8 block0_9 block0_10 block0_11 block0_12 block0_13 block0_14 block0_15) ( NPX NPY NPZ) simpleGrading (1 1 1)
    //block 3
    hex ( block0_0 block0_4 block0_7 block0_3 block0_12 block0_8 block0_11 block0_15 ) ( NPZ NPY NPZ ) simpleGrading (1 1 1)
    //block 4
    hex (block0_4 block0_5 block0_6 block0_7 block0_8 block0_9 block0_10 block0_11) ( NPX NPY NPZ ) simpleGrading (1 1 1)
);

edges
(
);

patches
(
    patch inlet
    (
        ( block0_4 block0_5 block0_9 block0_8 ) 
    )
      
    patch outlet
    (
        ( block0_3 block0_2 block0_6 block0_7 )
	( block0_11 block0_10 block0_14 block0_15 )
	( block0_3 block0_7 block0_11 block0_15 )
	( block0_2 block0_14 block0_10 block0_6 )
	( block0_7 block0_6 block0_10 block0_11)
    )

    wall wall_sides
    (
        ( block0_0 block0_3 block0_15 block0_12 )
    	( block0_1 block0_13 block0_14 block0_2 )
    )

    wall wall_frontback
    (
        ( block0_12 block0_15 block0_14 block0_13 )
    	( block0_0 block0_1 block0_2 block0_3 )
    )

    wall wall_botttom
    (
        ( block0_0 block0_1 block0_5 block0_4 )
	( block0_8 block0_9 block0_13 block0_12 )
	( block0_0 block0_4 block0_8 block0_12 )
	( block0_1 block0_13 block0_9 block0_5)
    )
);

mergePatchPairs
(
);

