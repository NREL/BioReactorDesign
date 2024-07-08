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
   define(L2, 21.0)  //  inner cylinder length
   define(T2, 0.1)   //  inner cylinder thickness
   define(OF2, D2) //  inner cylinder offset from bottom
   
   define(PI, 3.14159265)
   
   define(R1, calc(D1/2))
   define(R2, calc(D2/2))
   define(Rt, calc(R2+T2))
   define(CW, calc(D2/4)) //Width of middle square section
   
   define(CX1, calc(R1*cos((PI/180)*45)))
   define(CZ1, calc(R1*sin((PI/180)*45)))

   define(CX2, calc(R2*cos((PI/180)*45)))
   define(CZ2, calc(R2*sin((PI/180)*45)))

   define(CXt, calc(Rt*cos((PI/180)*45)))
   define(CZt, calc(Rt*sin((PI/180)*45)))

   define(OFF1,calc(OF2))
   define(OFF2,calc(OF2+L2))
   define(OFF3,calc(L1))
   
   define(NPS, 9) //how many cells in the square section
   define(NPD, 11) //how many cells from square section to perimeter
   define(NPT, 2) //how many cells in the thickness
   define(NPY, 200) // how many cells from top to bottom

   define(NPY1,calcint(NPY*OF2/L1))
   define(NPY2,calcint(NPY*L2/L1))
   define(NPY3,calcint(NPY-NPY1-NPY2))

   vertices
   (
    ( CW 0.0  CW) vlabel(fiveoclock_sq1)
    (-CW 0.0  CW) vlabel(sevenoclock_sq1)
    (-CW 0.0 -CW) vlabel(elevenoclock_sq1)
    ( CW 0.0 -CW) vlabel(oneoclock_sq1)

    ( CW OFF1  CW) vlabel(fiveoclock_sq2)
    (-CW OFF1  CW) vlabel(sevenoclock_sq2)
    (-CW OFF1 -CW) vlabel(elevenoclock_sq2)
    ( CW OFF1 -CW) vlabel(oneoclock_sq2)

    ( CW OFF2  CW) vlabel(fiveoclock_sq3)
    (-CW OFF2  CW) vlabel(sevenoclock_sq3)
    (-CW OFF2 -CW) vlabel(elevenoclock_sq3)
    ( CW OFF2 -CW) vlabel(oneoclock_sq3)

    ( CW OFF3  CW) vlabel(fiveoclock_sq4)
    (-CW OFF3  CW) vlabel(sevenoclock_sq4)
    (-CW OFF3 -CW) vlabel(elevenoclock_sq4)
    ( CW OFF3 -CW) vlabel(oneoclock_sq4)
   
    ( CX1 0.0  CZ1) vlabel(fiveoclock_c11)
    (-CX1 0.0  CZ1) vlabel(sevenoclock_c11)
    (-CX1 0.0 -CZ1) vlabel(elevenoclock_c11)
    ( CX1 0.0 -CZ1) vlabel(oneoclock_c11)

    ( CX1 OFF1  CZ1) vlabel(fiveoclock_c12)
    (-CX1 OFF1  CZ1) vlabel(sevenoclock_c12)
    (-CX1 OFF1 -CZ1) vlabel(elevenoclock_c12)
    ( CX1 OFF1 -CZ1) vlabel(oneoclock_c12)

    ( CX1 OFF2  CZ1) vlabel(fiveoclock_c13)
    (-CX1 OFF2 CZ1) vlabel(sevenoclock_c13)
    (-CX1 OFF2 -CZ1) vlabel(elevenoclock_c13)
    ( CX1 OFF2 -CZ1) vlabel(oneoclock_c13)

    ( CX1 OFF3  CZ1) vlabel(fiveoclock_c14)
    (-CX1 OFF3  CZ1) vlabel(sevenoclock_c14)
    (-CX1 OFF3 -CZ1) vlabel(elevenoclock_c14)
    ( CX1 OFF3 -CZ1) vlabel(oneoclock_c14)

    ( CX2 0.0  CZ2) vlabel(fiveoclock_c21)
    (-CX2 0.0  CZ2) vlabel(sevenoclock_c21)
    (-CX2 0.0 -CZ2) vlabel(elevenoclock_c21)
    ( CX2 0.0 -CZ2) vlabel(oneoclock_c21)

    ( CX2 OFF1  CZ2) vlabel(fiveoclock_c22)
    (-CX2 OFF1  CZ2) vlabel(sevenoclock_c22)
    (-CX2 OFF1 -CZ2) vlabel(elevenoclock_c22)
    ( CX2 OFF1 -CZ2) vlabel(oneoclock_c22)

    ( CX2 OFF2  CZ2) vlabel(fiveoclock_c23)
    (-CX2 OFF2  CZ2) vlabel(sevenoclock_c23)
    (-CX2 OFF2 -CZ2) vlabel(elevenoclock_c23)
    ( CX2 OFF2 -CZ2) vlabel(oneoclock_c23)

    ( CX2 OFF3  CZ2) vlabel(fiveoclock_c24)
    (-CX2 OFF3  CZ2) vlabel(sevenoclock_c24)
    (-CX2 OFF3 -CZ2) vlabel(elevenoclock_c24)
    ( CX2 OFF3 -CZ2) vlabel(oneoclock_c24)

    ( CXt 0.0  CZt) vlabel(fiveoclock_ct1)
    (-CXt 0.0  CZt) vlabel(sevenoclock_ct1)
    (-CXt 0.0 -CZt) vlabel(elevenoclock_ct1)
    ( CXt 0.0 -CZt) vlabel(oneoclock_ct1)

    ( CXt OFF1  CZt) vlabel(fiveoclock_ct2)
    (-CXt OFF1  CZt) vlabel(sevenoclock_ct2)
    (-CXt OFF1 -CZt) vlabel(elevenoclock_ct2)
    ( CXt OFF1 -CZt) vlabel(oneoclock_ct2)

    ( CXt OFF2  CZt) vlabel(fiveoclock_ct3)
    (-CXt OFF2  CZt) vlabel(sevenoclock_ct3)
    (-CXt OFF2 -CZt) vlabel(elevenoclock_ct3)
    ( CXt OFF2 -CZt) vlabel(oneoclock_ct3)

    ( CXt OFF3  CZt) vlabel(fiveoclock_ct4)
    (-CXt OFF3  CZt) vlabel(sevenoclock_ct4)
    (-CXt OFF3 -CZt) vlabel(elevenoclock_ct4)
    ( CXt OFF3 -CZt) vlabel(oneoclock_ct4)

   );				

   blocks
   (
    //square blocks
    hex (
       sevenoclock_sq1 fiveoclock_sq1 oneoclock_sq1 elevenoclock_sq1
       sevenoclock_sq2 fiveoclock_sq2 oneoclock_sq2 elevenoclock_sq2
       )
    (NPS NPS NPY1)
    simpleGrading (1 1 1)

    hex (
       sevenoclock_sq2 fiveoclock_sq2 oneoclock_sq2 elevenoclock_sq2
       sevenoclock_sq3 fiveoclock_sq3 oneoclock_sq3 elevenoclock_sq3
       )
    (NPS NPS NPY2)
    simpleGrading (1 1 1)

    hex (
       sevenoclock_sq3 fiveoclock_sq3 oneoclock_sq3 elevenoclock_sq3
       sevenoclock_sq4 fiveoclock_sq4 oneoclock_sq4 elevenoclock_sq4
       )
    (NPS NPS NPY3)
    simpleGrading (1 1 1)

    //slices (outer to thickness)*************************************

    // 1 to 2=========================================================
    //slice1
    hex (
       sevenoclock_c11 fiveoclock_c11 fiveoclock_ct1 sevenoclock_ct1
       sevenoclock_c12 fiveoclock_c12 fiveoclock_ct2 sevenoclock_ct2
       )
    (NPS NPD NPY1)
    simpleGrading (1 1 1)

    //slice2
    hex (
       elevenoclock_c11 sevenoclock_c11 sevenoclock_ct1 elevenoclock_ct1
       elevenoclock_c12 sevenoclock_c12 sevenoclock_ct2 elevenoclock_ct2
       )
   (NPS NPD NPY1)
simpleGrading (1 1 1)

   //slice3
   hex (
         oneoclock_c11 elevenoclock_c11 elevenoclock_ct1 oneoclock_ct1
         oneoclock_c12 elevenoclock_c12 elevenoclock_ct2 oneoclock_ct2
       )
   (NPS NPD NPY1)
simpleGrading (1 1 1)

   //slice4
   hex (
         fiveoclock_c11 oneoclock_c11 oneoclock_ct1 fiveoclock_ct1
         fiveoclock_c12 oneoclock_c12 oneoclock_ct2 fiveoclock_ct2
       )
   (NPS NPD  NPY1)
simpleGrading (1 1 1)
   //====================================================================
    
   // 2 to 3=========================================================
    //slice1
    hex (
       sevenoclock_c12 fiveoclock_c12 fiveoclock_ct2 sevenoclock_ct2
       sevenoclock_c13 fiveoclock_c13 fiveoclock_ct3 sevenoclock_ct3
       )
    (NPS NPD NPY2)
    simpleGrading (1 1 1)

    //slice2
    hex (
       elevenoclock_c12 sevenoclock_c12 sevenoclock_ct2 elevenoclock_ct2
       elevenoclock_c13 sevenoclock_c13 sevenoclock_ct3 elevenoclock_ct3
       )
   (NPS NPD NPY2)
simpleGrading (1 1 1)

   //slice3
   hex (
         oneoclock_c12 elevenoclock_c12 elevenoclock_ct2 oneoclock_ct2
         oneoclock_c13 elevenoclock_c13 elevenoclock_ct3 oneoclock_ct3
       )
   (NPS NPD NPY2)
simpleGrading (1 1 1)

   //slice4
   hex (
         fiveoclock_c12 oneoclock_c12 oneoclock_ct2 fiveoclock_ct2
         fiveoclock_c13 oneoclock_c13 oneoclock_ct3 fiveoclock_ct3
       )
   (NPS NPD  NPY2)
simpleGrading (1 1 1)
   //====================================================================

   // 3 to 4=========================================================
    //slice1
    hex (
       sevenoclock_c13 fiveoclock_c13 fiveoclock_ct3 sevenoclock_ct3
       sevenoclock_c14 fiveoclock_c14 fiveoclock_ct4 sevenoclock_ct4
       )
    (NPS NPD NPY3)
    simpleGrading (1 1 1)

    //slice2
    hex (
       elevenoclock_c13 sevenoclock_c13 sevenoclock_ct3 elevenoclock_ct3
       elevenoclock_c14 sevenoclock_c14 sevenoclock_ct4 elevenoclock_ct4
       )
   (NPS NPD NPY3)
simpleGrading (1 1 1)

   //slice3
   hex (
         oneoclock_c13 elevenoclock_c13 elevenoclock_ct3 oneoclock_ct3
         oneoclock_c14 elevenoclock_c14 elevenoclock_ct4 oneoclock_ct4
       )
   (NPS NPD NPY3)
simpleGrading (1 1 1)

   //slice4
   hex (
         fiveoclock_c13 oneoclock_c13 oneoclock_ct3 fiveoclock_ct3
         fiveoclock_c14 oneoclock_c14 oneoclock_ct4 fiveoclock_ct4
       )
   (NPS NPD  NPY3)
simpleGrading (1 1 1)
   //====================================================================

    //slices (inner to square)*************************************

    // 1 to 2=========================================================
    //slice1
    hex (
       sevenoclock_c21 fiveoclock_c21 fiveoclock_sq1 sevenoclock_sq1
       sevenoclock_c22 fiveoclock_c22 fiveoclock_sq2 sevenoclock_sq2
       )
    (NPS NPD NPY1)
    simpleGrading (1 1 1)

    //slice2
    hex (
       elevenoclock_c21 sevenoclock_c21 sevenoclock_sq1 elevenoclock_sq1
       elevenoclock_c22 sevenoclock_c22 sevenoclock_sq2 elevenoclock_sq2
       )
   (NPS NPD NPY1)
simpleGrading (1 1 1)

   //slice3
   hex (
         oneoclock_c21 elevenoclock_c21 elevenoclock_sq1 oneoclock_sq1
         oneoclock_c22 elevenoclock_c22 elevenoclock_sq2 oneoclock_sq2
       )
   (NPS NPD NPY1)
simpleGrading (1 1 1)

   //slice4
   hex (
         fiveoclock_c21 oneoclock_c21 oneoclock_sq1 fiveoclock_sq1
         fiveoclock_c22 oneoclock_c22 oneoclock_sq2 fiveoclock_sq2
       )
   (NPS NPD  NPY1)
simpleGrading (1 1 1)
   //====================================================================

    // 2 to 3=========================================================
    //slice1
    hex (
       sevenoclock_c22 fiveoclock_c22 fiveoclock_sq2 sevenoclock_sq2
       sevenoclock_c23 fiveoclock_c23 fiveoclock_sq3 sevenoclock_sq3
       )
    (NPS NPD NPY2)
    simpleGrading (1 1 1)

    //slice2
    hex (
       elevenoclock_c22 sevenoclock_c22 sevenoclock_sq2 elevenoclock_sq2
       elevenoclock_c23 sevenoclock_c23 sevenoclock_sq3 elevenoclock_sq3
       )
   (NPS NPD NPY2)
simpleGrading (1 1 1)

   //slice3
   hex (
         oneoclock_c22 elevenoclock_c22 elevenoclock_sq2 oneoclock_sq2
         oneoclock_c23 elevenoclock_c23 elevenoclock_sq3 oneoclock_sq3
       )
   (NPS NPD NPY2)
simpleGrading (1 1 1)

   //slice4
   hex (
         fiveoclock_c22 oneoclock_c22 oneoclock_sq2 fiveoclock_sq2
         fiveoclock_c23 oneoclock_c23 oneoclock_sq3 fiveoclock_sq3
       )
   (NPS NPD  NPY2)
simpleGrading (1 1 1)
   //====================================================================

    // 3 to 4=========================================================
    //slice1
    hex (
       sevenoclock_c23 fiveoclock_c23 fiveoclock_sq3 sevenoclock_sq3
       sevenoclock_c24 fiveoclock_c24 fiveoclock_sq4 sevenoclock_sq4
       )
    (NPS NPD NPY3)
    simpleGrading (1 1 1)

    //slice2
    hex (
       elevenoclock_c23 sevenoclock_c23 sevenoclock_sq3 elevenoclock_sq3
       elevenoclock_c24 sevenoclock_c24 sevenoclock_sq4 elevenoclock_sq4
       )
   (NPS NPD NPY3)
simpleGrading (1 1 1)

   //slice3
   hex (
         oneoclock_c23 elevenoclock_c23 elevenoclock_sq3 oneoclock_sq3
         oneoclock_c24 elevenoclock_c24 elevenoclock_sq4 oneoclock_sq4
       )
   (NPS NPD NPY3)
simpleGrading (1 1 1)

   //slice4
   hex (
         fiveoclock_c23 oneoclock_c23 oneoclock_sq3 fiveoclock_sq3
         fiveoclock_c24 oneoclock_c24 oneoclock_sq4 fiveoclock_sq4
       )
   (NPS NPD  NPY3)
simpleGrading (1 1 1)
   //====================================================================


    //slices (thickness to inner)*************************************

    // 1 to 2=========================================================
    //slice1
    hex (
       sevenoclock_ct1 fiveoclock_ct1 fiveoclock_c21 sevenoclock_c21
       sevenoclock_ct2 fiveoclock_ct2 fiveoclock_c22 sevenoclock_c22
       )
    (NPS NPT NPY1)
    simpleGrading (1 1 1)

    //slice2
    hex (
       elevenoclock_ct1 sevenoclock_ct1 sevenoclock_c21 elevenoclock_c21
       elevenoclock_ct2 sevenoclock_ct2 sevenoclock_c22 elevenoclock_c22
       )
   (NPS NPT NPY1)
simpleGrading (1 1 1)

   //slice3
   hex (
         oneoclock_ct1 elevenoclock_ct1 elevenoclock_c21 oneoclock_c21
         oneoclock_ct2 elevenoclock_ct2 elevenoclock_c22 oneoclock_c22
       )
   (NPS NPT NPY1)
simpleGrading (1 1 1)

   //slice4
   hex (
         fiveoclock_ct1 oneoclock_ct1 oneoclock_c21 fiveoclock_c21
         fiveoclock_ct2 oneoclock_ct2 oneoclock_c22 fiveoclock_c22
       )
   (NPS NPT  NPY1)
simpleGrading (1 1 1)
   //====================================================================

    // 3 to 4=========================================================
    //slice1
    hex (
       sevenoclock_ct3 fiveoclock_ct3 fiveoclock_c23 sevenoclock_c23
       sevenoclock_ct4 fiveoclock_ct4 fiveoclock_c24 sevenoclock_c24
       )
    (NPS NPT NPY3)
    simpleGrading (1 1 1)

    //slice2
    hex (
       elevenoclock_ct3 sevenoclock_ct3 sevenoclock_c23 elevenoclock_c23
       elevenoclock_ct4 sevenoclock_ct4 sevenoclock_c24 elevenoclock_c24
       )
   (NPS NPT NPY3)
simpleGrading (1 1 1)

   //slice3
   hex (
         oneoclock_ct3 elevenoclock_ct3 elevenoclock_c23 oneoclock_c23
         oneoclock_ct4 elevenoclock_ct4 elevenoclock_c24 oneoclock_c24
       )
   (NPS NPT NPY3)
simpleGrading (1 1 1)

   //slice4
   hex (
         fiveoclock_ct3 oneoclock_ct3 oneoclock_c23 fiveoclock_c23
         fiveoclock_ct4 oneoclock_ct4 oneoclock_c24 fiveoclock_c24
       )
   (NPS NPT  NPY3)
simpleGrading (1 1 1)
   //====================================================================

   );


   //create the quarter circles
   edges
   (
    arc fiveoclock_c11 sevenoclock_c11   (0.0 0.0 R1)
    arc sevenoclock_c11 elevenoclock_c11 (-R1 0.0 0.0)
    arc elevenoclock_c11 oneoclock_c11   (0.0 0.0 -R1)
    arc oneoclock_c11 fiveoclock_c11     (R1 0.0 0.0)

    arc fiveoclock_c12 sevenoclock_c12   (0.0 OFF1 R1)
    arc sevenoclock_c12 elevenoclock_c12 (-R1 OFF1 0.0)
    arc elevenoclock_c12 oneoclock_c12   (0.0 OFF1 -R1)
    arc oneoclock_c12 fiveoclock_c12     (R1  OFF1 0.0)
    
    arc fiveoclock_c13 sevenoclock_c13   (0.0 OFF2 R1)
    arc sevenoclock_c13 elevenoclock_c13 (-R1 OFF2 0.0)
    arc elevenoclock_c13 oneoclock_c13   (0.0 OFF2 -R1)
    arc oneoclock_c13 fiveoclock_c13     (R1  OFF2 0.0)

    arc fiveoclock_c14 sevenoclock_c14   (0.0 OFF3 R1)
    arc sevenoclock_c14 elevenoclock_c14 (-R1 OFF3 0.0)
    arc elevenoclock_c14 oneoclock_c14   (0.0 OFF3 -R1)
    arc oneoclock_c14 fiveoclock_c14     (R1  OFF3 0.0)

    arc fiveoclock_c21 sevenoclock_c21   (0.0 0.0 R2)
    arc sevenoclock_c21 elevenoclock_c21 (-R2 0.0 0.0)
    arc elevenoclock_c21 oneoclock_c21   (0.0 0.0 -R2)
    arc oneoclock_c21 fiveoclock_c21     (R2 0.0 0.0)

    arc fiveoclock_c22 sevenoclock_c22   (0.0 OFF1 R2)
    arc sevenoclock_c22 elevenoclock_c22 (-R2 OFF1 0.0)
    arc elevenoclock_c22 oneoclock_c22   (0.0 OFF1 -R2)
    arc oneoclock_c22 fiveoclock_c22     (R2  OFF1 0.0)
    
    arc fiveoclock_c23 sevenoclock_c23   (0.0 OFF2 R2)
    arc sevenoclock_c23 elevenoclock_c23 (-R2 OFF2 0.0)
    arc elevenoclock_c23 oneoclock_c23   (0.0 OFF2 -R2)
    arc oneoclock_c23 fiveoclock_c23     (R2  OFF2 0.0)

    arc fiveoclock_c24 sevenoclock_c24   (0.0 OFF3 R2)
    arc sevenoclock_c24 elevenoclock_c24 (-R2 OFF3 0.0)
    arc elevenoclock_c24 oneoclock_c24   (0.0 OFF3 -R2)
    arc oneoclock_c24 fiveoclock_c24     (R2  OFF3 0.0)

    arc fiveoclock_ct1 sevenoclock_ct1   (0.0 0.0 Rt)
    arc sevenoclock_ct1 elevenoclock_ct1 (-Rt 0.0 0.0)
    arc elevenoclock_ct1 oneoclock_ct1   (0.0 0.0 -Rt)
    arc oneoclock_ct1 fiveoclock_ct1     (Rt 0.0 0.0)

    arc fiveoclock_ct2 sevenoclock_ct2   (0.0 OFF1 Rt)
    arc sevenoclock_ct2 elevenoclock_ct2 (-Rt OFF1 0.0)
    arc elevenoclock_ct2 oneoclock_ct2   (0.0 OFF1 -Rt)
    arc oneoclock_ct2 fiveoclock_ct2     (Rt  OFF1 0.0)
    
    arc fiveoclock_ct3 sevenoclock_ct3   (0.0 OFF2 Rt)
    arc sevenoclock_ct3 elevenoclock_ct3 (-Rt OFF2 0.0)
    arc elevenoclock_ct3 oneoclock_ct3   (0.0 OFF2 -Rt)
    arc oneoclock_ct3 fiveoclock_ct3     (Rt  OFF2 0.0)

    arc fiveoclock_ct4 sevenoclock_ct4   (0.0 OFF3 Rt)
    arc sevenoclock_ct4 elevenoclock_ct4 (-Rt OFF3 0.0)
    arc elevenoclock_ct4 oneoclock_ct4   (0.0 OFF3 -Rt)
    arc oneoclock_ct4 fiveoclock_ct4     (Rt  OFF3 0.0)

   );

   patches
   (
    patch inlet
    (
     (fiveoclock_sq1 oneoclock_sq1 elevenoclock_sq1 sevenoclock_sq1)

     (fiveoclock_sq1 fiveoclock_c21 oneoclock_c21 oneoclock_sq1)
     (fiveoclock_c21 fiveoclock_sq1 sevenoclock_sq1 sevenoclock_c21)
     (sevenoclock_sq1 elevenoclock_sq1 elevenoclock_c21 sevenoclock_c21)
     (oneoclock_sq1 oneoclock_c21 elevenoclock_c21 elevenoclock_sq1)
    )

    patch outlet
    (
     (fiveoclock_sq4 oneoclock_sq4 elevenoclock_sq4 sevenoclock_sq4)

     (fiveoclock_sq4 fiveoclock_c24 oneoclock_c24 oneoclock_sq4)
     (fiveoclock_c24 fiveoclock_sq4 sevenoclock_sq4 sevenoclock_c24)
     (sevenoclock_sq4 elevenoclock_sq4 elevenoclock_c24 sevenoclock_c24)
     (oneoclock_sq4 oneoclock_c24 elevenoclock_c24 elevenoclock_sq4)

     (fiveoclock_ct4 oneoclock_ct4 oneoclock_c24 fiveoclock_c24)
     (oneoclock_ct4 elevenoclock_ct4 elevenoclock_c24 oneoclock_c24)
     (elevenoclock_ct4 sevenoclock_ct4 sevenoclock_c24 elevenoclock_c24)
     (fiveoclock_ct4 sevenoclock_ct4 sevenoclock_c24 fiveoclock_c24)

     (fiveoclock_ct4 fiveoclock_c14 oneoclock_c14 oneoclock_ct4)
     (fiveoclock_c14 fiveoclock_ct4 sevenoclock_ct4 sevenoclock_c14)
     (sevenoclock_c14 sevenoclock_ct4 elevenoclock_ct4 elevenoclock_c14)
     (elevenoclock_c14 elevenoclock_ct4 oneoclock_ct4 oneoclock_c14)
    )

    wall walls
    (

     (fiveoclock_c21 fiveoclock_ct1 oneoclock_ct1 oneoclock_c21)
     (fiveoclock_ct1 fiveoclock_c21  sevenoclock_c21 sevenoclock_ct1)
     (sevenoclock_ct1 sevenoclock_c21 elevenoclock_c21 elevenoclock_ct1)
     (elevenoclock_ct1 elevenoclock_c21 oneoclock_c21 oneoclock_ct1)

     (fiveoclock_ct1 fiveoclock_c11 oneoclock_c11 oneoclock_ct1)
     (fiveoclock_c11 fiveoclock_ct1  sevenoclock_ct1 sevenoclock_c11)
     (sevenoclock_c11 sevenoclock_ct1 elevenoclock_ct1 elevenoclock_c11)
     (elevenoclock_c11 elevenoclock_ct1 oneoclock_ct1 oneoclock_c11)

     (sevenoclock_c11 fiveoclock_c11 fiveoclock_c12 sevenoclock_c12)
     (sevenoclock_c11 sevenoclock_c12 elevenoclock_c12 elevenoclock_c11)
     (elevenoclock_c11 elevenoclock_c12 oneoclock_c12 oneoclock_c11)
     (oneoclock_c11 oneoclock_c12 fiveoclock_c12 fiveoclock_c11)

     (sevenoclock_c12 fiveoclock_c12 fiveoclock_c13 sevenoclock_c13)
     (sevenoclock_c12 sevenoclock_c13 elevenoclock_c13 elevenoclock_c12)
     (elevenoclock_c12 elevenoclock_c13 oneoclock_c13 oneoclock_c12)
     (oneoclock_c12 oneoclock_c13 fiveoclock_c13 fiveoclock_c12)

     (sevenoclock_c13 fiveoclock_c13 fiveoclock_c14 sevenoclock_c14)
     (sevenoclock_c13 sevenoclock_c14 elevenoclock_c14 elevenoclock_c13)
     (elevenoclock_c13 elevenoclock_c14 oneoclock_c14 oneoclock_c13)
     (oneoclock_c13 oneoclock_c14 fiveoclock_c14 fiveoclock_c13)

     (sevenoclock_c22 fiveoclock_c22 fiveoclock_c23 sevenoclock_c23)
     (sevenoclock_c22 sevenoclock_c23 elevenoclock_c23 elevenoclock_c22)
     (elevenoclock_c22 elevenoclock_c23 oneoclock_c23 oneoclock_c22)
     (oneoclock_c22 oneoclock_c23 fiveoclock_c23 fiveoclock_c22)

     (sevenoclock_ct2 fiveoclock_ct2 fiveoclock_ct3 sevenoclock_ct3)
     (sevenoclock_ct2 sevenoclock_ct3 elevenoclock_ct3 elevenoclock_ct2)
     (elevenoclock_ct2 elevenoclock_ct3 oneoclock_ct3 oneoclock_ct2)
     (sevenoclock_ct2 fiveoclock_ct2 fiveoclock_c22 sevenoclock_c22)

     (fiveoclock_ct2  oneoclock_ct2  oneoclock_c22  fiveoclock_c22)
     (oneoclock_ct2 elevenoclock_ct2 elevenoclock_c22 oneoclock_c22)
     (elevenoclock_ct2 sevenoclock_ct2 sevenoclock_c22 elevenoclock_c22)
     (oneoclock_ct2 oneoclock_ct3 fiveoclock_ct3 fiveoclock_ct2)

     (sevenoclock_ct3 fiveoclock_ct3 fiveoclock_c23 sevenoclock_c23)
     (fiveoclock_ct3  oneoclock_ct3  oneoclock_c23  fiveoclock_c23)
     (oneoclock_ct3 elevenoclock_ct3 elevenoclock_c23 oneoclock_c23)
     (elevenoclock_ct3 sevenoclock_ct3 sevenoclock_c23 elevenoclock_c23)

    )

);

mergePatchPairs
(
);
