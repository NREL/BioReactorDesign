//--------------------------------*- C++ -*----------------------------------
// blockMesh :  Block mesh description file
//
// adapted from:
// http://www.cfd-online.com/Forums/openfoam-meshing-blockmesh/61796-help-could-anyone-post-simple-cylinder-mesh.html
//
// JJS, 1/8/16
//
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

   define(D1, 5.0)   // outer cylinder diameter
   define(L1, 40.0)  // outer cylinder length
   define(D2, calc(0.707*D1))   //  inner cylinder diameter
   
   define(PI, 3.14159265)
   
   define(R1, calc(D1/2))
   define(R2, calc(D2/2))
   define(CW, calc(D2/4)) //Width of middle square section
   
   define(CX1, calc(R1*cos((PI/180)*45)))
   define(CZ1, calc(R1*sin((PI/180)*45)))

   define(CX2, calc(R2*cos((PI/180)*45)))
   define(CZ2, calc(R2*sin((PI/180)*45)))

   define(NPS, 8) //how many cells in the square section
   define(NPD, 8) //how many cells from square section to perimeter
   define(NPY, 200) // how many cells from top to bottom

   vertices
   (
    //Bottom surface
    //==========

    //square
    ( CW 0.0  CW) vlabel(fiveoclock_sq1)
    (-CW 0.0  CW) vlabel(sevenoclock_sq1)
    (-CW 0.0 -CW) vlabel(elevenoclock_sq1)
    ( CW 0.0 -CW) vlabel(oneoclock_sq1)

    //outer circle
    ( CX1 0.0  CZ1) vlabel(fiveoclock_c11)
    (-CX1 0.0  CZ1) vlabel(sevenoclock_c11)
    (-CX1 0.0 -CZ1) vlabel(elevenoclock_c11)
    ( CX1 0.0 -CZ1) vlabel(oneoclock_c11)

    //inner circle
    ( CX2 0.0  CZ2) vlabel(fiveoclock_c21)
    (-CX2 0.0  CZ2) vlabel(sevenoclock_c21)
    (-CX2 0.0 -CZ2) vlabel(elevenoclock_c21)
    ( CX2 0.0 -CZ2) vlabel(oneoclock_c21)
   

    //Top surface

    //square
    ( CW L1  CW) vlabel(fiveoclock_sq2)
    (-CW L1  CW) vlabel(sevenoclock_sq2)
    (-CW L1 -CW) vlabel(elevenoclock_sq2)
    ( CW L1 -CW) vlabel(oneoclock_sq2)

    //outer circle
    ( CX1 L1  CZ1) vlabel(fiveoclock_c12)
    (-CX1 L1  CZ1) vlabel(sevenoclock_c12)
    (-CX1 L1 -CZ1) vlabel(elevenoclock_c12)
    ( CX1 L1 -CZ1) vlabel(oneoclock_c12)

    //inner circle
    ( CX2 L1  CZ2) vlabel(fiveoclock_c22)
    (-CX2 L1  CZ2) vlabel(sevenoclock_c22)
    (-CX2 L1 -CZ2) vlabel(elevenoclock_c22)
    ( CX2 L1 -CZ2) vlabel(oneoclock_c22)

   );				

   blocks
   (
    //square block
    hex (
       sevenoclock_sq1 fiveoclock_sq1 oneoclock_sq1 elevenoclock_sq1
       sevenoclock_sq2 fiveoclock_sq2 oneoclock_sq2 elevenoclock_sq2
       )
    (NPS NPS NPY)
    simpleGrading (1 1 1)

    //slice1
    hex (
       sevenoclock_c21 fiveoclock_c21 fiveoclock_sq1 sevenoclock_sq1
       sevenoclock_c22 fiveoclock_c22 fiveoclock_sq2 sevenoclock_sq2
       )
    (NPS NPD NPY)
    simpleGrading (1 1 1)

    //slice2
    hex (
       elevenoclock_c21 sevenoclock_c21 sevenoclock_sq1 elevenoclock_sq1
       elevenoclock_c22 sevenoclock_c22 sevenoclock_sq2 elevenoclock_sq2
       )
   (NPS NPD NPY)
simpleGrading (1 1 1)

   //slice3
   hex (
         oneoclock_c21 elevenoclock_c21 elevenoclock_sq1 oneoclock_sq1
         oneoclock_c22 elevenoclock_c22 elevenoclock_sq2 oneoclock_sq2
       )
   (NPS NPD NPY)
simpleGrading (1 1 1)

   //slice4
   hex (
         fiveoclock_c21 oneoclock_c21 oneoclock_sq1 fiveoclock_sq1
         fiveoclock_c22 oneoclock_c22 oneoclock_sq2 fiveoclock_sq2
       )
   (NPS NPD  NPY)
   simpleGrading (1 1 1)

    //slice1
    hex (
       fiveoclock_c11 fiveoclock_c21 sevenoclock_c21 sevenoclock_c11
       fiveoclock_c12 fiveoclock_c22 sevenoclock_c22 sevenoclock_c12
       )
    (NPD NPS NPY)
    simpleGrading (1 1 1)

    //slice2
    hex (
       sevenoclock_c11 sevenoclock_c21 elevenoclock_c21 elevenoclock_c11
       sevenoclock_c12 sevenoclock_c22 elevenoclock_c22 elevenoclock_c12
       )
   (NPD NPS NPY)
simpleGrading (1 1 1)

   //slice3
   hex (
       elevenoclock_c11 elevenoclock_c21 oneoclock_c21  oneoclock_c11
       elevenoclock_c12 elevenoclock_c22 oneoclock_c22  oneoclock_c12
       )
   (NPD NPS NPY)
simpleGrading (1 1 1)

   //slice4
   hex (
       oneoclock_c11 oneoclock_c21 fiveoclock_c21  fiveoclock_c11
       oneoclock_c12 oneoclock_c22 fiveoclock_c22  fiveoclock_c12
       )
   (NPD NPS  NPY)
   simpleGrading (1 1 1)
    
   );


   //create the quarter circles
   edges
   (
    arc fiveoclock_c11 sevenoclock_c11   (0.0 0.0 R1)
    arc sevenoclock_c11 elevenoclock_c11 (-R1 0.0 0.0)
    arc elevenoclock_c11 oneoclock_c11   (0.0 0.0 -R1)
    arc oneoclock_c11 fiveoclock_c11     (R1 0.0 0.0)

    arc fiveoclock_c12 sevenoclock_c12   (0.0 L1 R1)
    arc sevenoclock_c12 elevenoclock_c12 (-R1 L1 0.0)
    arc elevenoclock_c12 oneoclock_c12   (0.0 L1 -R1)
    arc oneoclock_c12 fiveoclock_c12     (R1  L1 0.0)
    
    arc fiveoclock_c21 sevenoclock_c21   (0.0 0.0 R2)
    arc sevenoclock_c21 elevenoclock_c21 (-R2 0.0 0.0)
    arc elevenoclock_c21 oneoclock_c21   (0.0 0.0 -R2)
    arc oneoclock_c21 fiveoclock_c21     (R2 0.0 0.0)

    arc fiveoclock_c22 sevenoclock_c22   (0.0 L1 R2)
    arc sevenoclock_c22 elevenoclock_c22 (-R2 L1 0.0)
    arc elevenoclock_c22 oneoclock_c22   (0.0 L1 -R2)
    arc oneoclock_c22 fiveoclock_c22     (R2  L1 0.0)
   );

   patches
   (
    patch inlet
    (
     (fiveoclock_sq1 oneoclock_sq1 elevenoclock_sq1 sevenoclock_sq1)
     (fiveoclock_sq1 fiveoclock_c21 sevenoclock_c21 sevenoclock_sq1)
     (sevenoclock_sq1 sevenoclock_c21 elevenoclock_c21 elevenoclock_sq1)
     (elevenoclock_sq1 elevenoclock_c21 oneoclock_c21 oneoclock_sq1)
     (oneoclock_sq1 oneoclock_c21 fiveoclock_c21 fiveoclock_sq1)
    )

    patch outlet
    (
     (fiveoclock_sq2 oneoclock_sq2 elevenoclock_sq2 sevenoclock_sq2)
     (fiveoclock_sq2 fiveoclock_c22 sevenoclock_c22 sevenoclock_sq2)
     (sevenoclock_sq2 sevenoclock_c22 elevenoclock_c22 elevenoclock_sq2)
     (elevenoclock_sq2 elevenoclock_c22 oneoclock_c22 oneoclock_sq2)
     (oneoclock_sq2 oneoclock_c22 fiveoclock_c22 fiveoclock_sq2)

     (fiveoclock_c22 fiveoclock_c12 sevenoclock_c12 sevenoclock_c22)
     (sevenoclock_c22 sevenoclock_c12 elevenoclock_c12 elevenoclock_c22)
     (elevenoclock_c22 elevenoclock_c12 oneoclock_c12 oneoclock_c22)
     (oneoclock_c22 oneoclock_c12 fiveoclock_c12 fiveoclock_c22)

    )

    wall walls
    (
     (fiveoclock_c21 fiveoclock_c11 sevenoclock_c11 sevenoclock_c21)
     (sevenoclock_c21 sevenoclock_c11 elevenoclock_c11 elevenoclock_c21)
     (elevenoclock_c21 elevenoclock_c11 oneoclock_c11 oneoclock_c21)
     (oneoclock_c21 oneoclock_c11 fiveoclock_c11 fiveoclock_c21)
     
     (fiveoclock_c11 fiveoclock_c12 sevenoclock_c12 sevenoclock_c11)
     (sevenoclock_c11 sevenoclock_c12 elevenoclock_c12 elevenoclock_c11)
     (elevenoclock_c11 elevenoclock_c12 oneoclock_c12 oneoclock_c11)
     (oneoclock_c11 oneoclock_c12 fiveoclock_c12 fiveoclock_c11)
    )

);

mergePatchPairs
(
);
