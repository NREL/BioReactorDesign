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
define(VCOUNT, 0)
define(vlabel, [[// ]Vertex $1 = VCOUNT define($1, VCOUNT)define([VCOUNT], incr(VCOUNT))])

   convertToMeters 1;

   define(D, 5.0) // diameter
   define(L, 40.0) // length
   define(PI, 3.14159265)
   
   define(R, calc(D/2))
   define(CW, calc(D/4)) //Width of middle square section
   
   define(CX, calc(R*cos((PI/180)*45)))
   define(CZ, calc(R*sin((PI/180)*45)))
   
   define(NPS, 5) //how many cells in the square section
   define(NPD, 5) //how many cells from square section to perimeter
   define(NPY, 200) // how many cells from top to bottom

   vertices
   (
    ( CW 0.0  CW) vlabel(fiveoclocksqb)
    (-CW 0.0  CW) vlabel(sevenoclocksqb)
    (-CW 0.0 -CW) vlabel(elevenoclocksqb)
    ( CW 0.0 -CW) vlabel(oneoclocksqb)
   
    ( CX 0.0  CZ) vlabel(fiveoclockcb)
    (-CX 0.0  CZ) vlabel(sevenoclockcb)
    (-CX 0.0 -CZ) vlabel(elevenoclockcb)
    ( CX 0.0 -CZ) vlabel(oneoclockcb)

    ( CW L  CW) vlabel(fiveoclocksqt)
    (-CW L  CW) vlabel(sevenoclocksqt)
    (-CW L -CW) vlabel(elevenoclocksqt)
    ( CW L -CW) vlabel(oneoclocksqt)
   
    ( CX L  CZ) vlabel(fiveoclockct)
    (-CX L  CZ) vlabel(sevenoclockct)
    (-CX L -CZ) vlabel(elevenoclockct)
    ( CX L -CZ) vlabel(oneoclockct)
   );				

   blocks
   (
    //square block
    hex (
       sevenoclocksqb fiveoclocksqb oneoclocksqb elevenoclocksqb
       sevenoclocksqt fiveoclocksqt oneoclocksqt elevenoclocksqt
       )
    (NPS NPS NPY)
    simpleGrading (1 1 1)

    //slice1
    hex (
       sevenoclockcb fiveoclockcb fiveoclocksqb sevenoclocksqb
       sevenoclockct fiveoclockct fiveoclocksqt sevenoclocksqt
       )
    (NPS NPD NPY)
    simpleGrading (1 1 1)

    //slice2
    hex (
       sevenoclocksqb elevenoclocksqb elevenoclockcb sevenoclockcb 
       sevenoclocksqt elevenoclocksqt elevenoclockct sevenoclockct 
       )
   (NPS NPD NPY)
simpleGrading (1 1 1)

   //slice3
   hex (
         elevenoclocksqb oneoclocksqb oneoclockcb elevenoclockcb
         elevenoclocksqt oneoclocksqt oneoclockct elevenoclockct
       )
   (NPS NPD NPY)
simpleGrading (1 1 1)

   //slice4
   hex (
         oneoclocksqb fiveoclocksqb fiveoclockcb oneoclockcb
         oneoclocksqt fiveoclocksqt fiveoclockct oneoclockct
       )
   (NPS NPD  NPY)
simpleGrading (1 1 1)

   );


   //create the quarter circles
   edges
   (
    arc fiveoclockcb sevenoclockcb (0.0 0.0 R)
    arc sevenoclockcb elevenoclockcb (-R 0.0 0.0)
    arc elevenoclockcb oneoclockcb (0.0 0.0 -R)
    arc oneoclockcb fiveoclockcb (R 0.0 0.0)

    arc fiveoclockct sevenoclockct (0.0 L R)
    arc sevenoclockct elevenoclockct (-R L 0.0)
    arc elevenoclockct oneoclockct (0.0 L -R)
    arc oneoclockct fiveoclockct (R L 0.0)

   );

   patches
   (
    patch inlet
    (
     (fiveoclocksqb oneoclocksqb elevenoclocksqb sevenoclocksqb)
     (fiveoclocksqb fiveoclockcb oneoclockcb oneoclocksqb)
     (fiveoclockcb fiveoclocksqb sevenoclocksqb sevenoclockcb)
     (sevenoclocksqb elevenoclocksqb elevenoclockcb sevenoclockcb)
     (oneoclocksqb oneoclockcb elevenoclockcb elevenoclocksqb)
    )

    patch outlet
    (
     (fiveoclocksqt oneoclocksqt elevenoclocksqt sevenoclocksqt)
     (fiveoclocksqt fiveoclockct oneoclockct oneoclocksqt)
     (fiveoclockct fiveoclocksqt sevenoclocksqt sevenoclockct)
     (sevenoclocksqt elevenoclocksqt elevenoclockct sevenoclockct)
     (oneoclocksqt oneoclockct elevenoclockct elevenoclocksqt)
    )

    wall walls
    (
     (sevenoclockcb fiveoclockcb fiveoclockct sevenoclockct)
     (sevenoclockcb sevenoclockct elevenoclockct elevenoclockcb)
     (elevenoclockcb elevenoclockct oneoclockct oneoclockcb)
     (oneoclockcb oneoclockct fiveoclockct fiveoclockcb)
    )

);

mergePatchPairs
(
);
