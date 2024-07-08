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

   convertToMeters 0.025; //inches to m

   define(LX, 100.0)  
   define(LY, 60.0)  
   define(LZ, 60.0)  // outer cylinder length
   define(D, 6.0)   //  inner cylinder diameter
   define(DX,5)
   define(DY,5)
   define(DZ,5)
   
   define(NPL,4) //number of points per unit length

   vertices
   (
     //section 1
     //============
     (0.0  0.0  0.0)
     (DX   0.0  0.0)
     (calc(DX+D) 0.0  0.0)
     (calc(LX-DX-D) 0.0 0.0)
     (calc(LX-DX) 0.0 0.0)
     (LX 0.0 0.0)
     
     (0.0  D  0.0)
     (DX   D  0.0)
     (calc(DX+D) D  0.0)
     (calc(LX-DX-D) D 0.0)
     (calc(LX-DX) D 0.0)
     (LX D 0.0)
     
     (0.0  0.0  D)
     (DX   0.0  D)
     (calc(DX+D) 0.0  D)
     (calc(LX-DX-D) 0.0 D)
     (calc(LX-DX) 0.0 D)
     (LX 0.0 D)
     
     (0.0  D  D)
     (DX   D  D)
     (calc(DX+D) D  D)
     (calc(LX-DX-D) D D)
     (calc(LX-DX) D D)
     (LX D D)
     
     //section 2
     //============
     (0.0  0.0  calc(LZ-D-2.0*DZ))
     (DX   0.0  calc(LZ-D-2.0*DZ))
     (calc(DX+D) 0.0  calc(LZ-D-2.0*DZ))
     (calc(LX-DX-D) 0.0 calc(LZ-D-2.0*DZ))
     (calc(LX-DX) 0.0 calc(LZ-D-2.0*DZ))
     (LX 0.0 calc(LZ-D-2.0*DZ))
     
     (0.0  D  calc(LZ-D-2.0*DZ))
     (DX   D  calc(LZ-D-2.0*DZ))
     (calc(DX+D) D  calc(LZ-D-2.0*DZ))
     (calc(LX-DX-D) D calc(LZ-D-2.0*DZ))
     (calc(LX-DX) D calc(LZ-D-2.0*DZ))
     (LX D calc(LZ-D-2.0*DZ))
     
     (0.0  0.0  calc(LZ-2.0*DZ))
     (DX   0.0  calc(LZ-2.0*DZ))
     (calc(DX+D) 0.0  calc(LZ-2.0*DZ))
     (calc(LX-DX-D) 0.0 calc(LZ-2.0*DZ))
     (calc(LX-DX) 0.0 calc(LZ-2.0*DZ))
     (LX 0.0 calc(LZ-2.0*DZ))
     
     (0.0  D  calc(LZ-2.0*DZ))
     (DX   D  calc(LZ-2.0*DZ))
     (calc(DX+D) D  calc(LZ-2.0*DZ))
     (calc(LX-DX-D) D calc(LZ-2.0*DZ))
     (calc(LX-DX) D calc(LZ-2.0*DZ))
     (LX D calc(LZ-2.0*DZ))
     
     //section 3
     //============
     (calc(LX-D-DX)  0.0  calc(-DZ))
     (calc(LX-DX)    0.0  calc(-DZ))
     (calc(LX-D-DX)  D    calc(-DZ))
     (calc(LX-DX)    D    calc(-DZ))
     
     //section 4
     //============
     (calc(LX-D-DX)  0.0  calc(LZ-DZ))
     (calc(LX-DX)    0.0  calc(LZ-DZ))
     (calc(LX-D-DX)  D    calc(LZ-DZ))
     (calc(LX-DX)    D    calc(LZ-DZ))
     
     //section 5
     //============
     (DX    calc(-DY) 0.0)
     (calc(DX+D)  calc(-DY) 0.0)
     (DX    calc(-DY)   D)
     (calc(DX+D)  calc(-DY)   D)

     //section 6
     //===========
     (DX          calc(-DY)  calc(LZ-D-2.0*DZ))
     (calc(DX+D)  calc(-DY)  calc(LZ-D-2.0*DZ))
     (DX          calc(-DY)    calc(LZ-2.0*DZ))
     (calc(DX+D)  calc(-DY)    calc(LZ-2.0*DZ))
     
     //section 7
     //===========
     (DX    calc(LY-D-2.0*DY) 0.0)
     (calc(DX+D)    calc(LY-D-2.0*DY) 0.0)
     (DX calc(LY-D-2.0*DY)  D)
     (calc(DX+D)    calc(LY-D-2.0*DY) D)
     
     //section 8
     //===========
     (DX            calc(LY-D-2.0*DY)  calc(LZ-D-2.0*DZ))
     (calc(DX+D)    calc(LY-D-2.0*DY)  calc(LZ-D-2.0*DZ))
     (DX            calc(LY-D-2.0*DY)    calc(LZ-2.0*DZ))
     (calc(DX+D)    calc(LY-D-2.0*DY)    calc(LZ-2.0*DZ))
     
     //section 9
     //===========
     (DX    calc(LY-2.0*DY) 0.0)
     (calc(DX+D)    calc(LY-2.0*DY) 0.0)
     (DX calc(LY-2.0*DY)  D)
     (calc(DX+D)    calc(LY-2.0*DY) D)
     
     //section 10
     //===========
     (DX            calc(LY-2.0*DY)  calc(LZ-D-2.0*DZ))
     (calc(DX+D)    calc(LY-2.0*DY)  calc(LZ-D-2.0*DZ))
     (DX            calc(LY-2.0*DY)    calc(LZ-2.0*DZ))
     (calc(DX+D)    calc(LY-2.0*DY)    calc(LZ-2.0*DZ))
     
     //section 11
     //===========
     (DX            calc(LY-DY) 0.0)
     (calc(DX+D)    calc(LY-DY) 0.0)
     (DX            calc(LY-DY)   D)
     (calc(DX+D)    calc(LY-DY)   D)
     
     //section 12
     //===========
     (DX            calc(LY-DY) calc(LZ-D-2.0*DZ))
     (calc(DX+D)    calc(LY-DY) calc(LZ-D-2.0*DZ))
     (DX            calc(LY-DY)   calc(LZ-2.0*DZ))
     (calc(DX+D)    calc(LY-DY)   calc(LZ-2.0*DZ))


     //section 13
     //==========
     (DX         calc(LY-D-2.0*DY)  calc(-DZ))
     (calc(DX+D) calc(LY-D-2.0*DY)  calc(-DZ))
     (DX         calc(LY-2.0*DY)    calc(-DZ))
     (calc(DX+D) calc(LY-2.0*DY)    calc(-DZ))
     
     //section 14
     //==========
     (DX         calc(LY-D-2.0*DY)  calc(LZ-DZ))
     (calc(DX+D) calc(LY-D-2.0*DY)  calc(LZ-DZ))
     (DX         calc(LY-2.0*DY)    calc(LZ-DZ))
     (calc(DX+D) calc(LY-2.0*DY)    calc(LZ-DZ))

   );				

   blocks
   (
    //section 1
    hex (1 0 12 13 7 6 18 19) (calcint(DX*NPL) calcint(D*NPL) calcint(D*NPL)) simpleGrading (1 1 1)
    hex (2 1 13 14 8 7 19 20) (calcint(D*NPL) calcint(D*NPL) calcint(D*NPL)) simpleGrading (1 1 1)
    hex (3 2 14 15 9 8 20 21) (calcint((LX-2*D-2*DX)*NPL) calcint(D*NPL) calcint(D*NPL)) simpleGrading (1 1 1)
    hex (4 3 15 16 10 9 21 22) (calcint(D*NPL) calcint(D*NPL) calcint(D*NPL)) simpleGrading (1 1 1)
    hex (5 4 16 17 11 10 22 23) (calcint(DX*NPL) calcint(D*NPL) calcint(D*NPL)) simpleGrading (1 1 1)
    
    //section 2
    hex (25 24 36 37 31 30 42 43) (calcint(DX*NPL) calcint(D*NPL) calcint(D*NPL)) simpleGrading (1 1 1)
    hex (26 25 37 38 32 31 43 44) (calcint(D*NPL) calcint(D*NPL) calcint(D*NPL)) simpleGrading (1 1 1)
    hex (27 26 38 39 33 32 44 45) (calcint((LX-2*D-2*DX)*NPL) calcint(D*NPL) calcint(D*NPL)) simpleGrading (1 1 1)
    hex (28 27 39 40 34 33 45 46) (calcint(D*NPL) calcint(D*NPL) calcint(D*NPL)) simpleGrading (1 1 1)
    hex (29 28 40 41 35 34 46 47) (calcint(DX*NPL) calcint(D*NPL) calcint(D*NPL)) simpleGrading (1 1 1)

    //section 3
    hex (49 48  3  4 51 50  9 10) (calcint(D*NPL) calcint(DZ*NPL) calcint(D*NPL)) simpleGrading (1 1 1)

    //section 3-4
    hex (16 15 27 28 22 21 33 34) (calcint(D*NPL) calcint((LZ-2*D-2*DZ)*NPL) calcint(D*NPL)) simpleGrading (1 1 1)

    //section 4
    hex (40 39 52 53 46 45 54 55) (calcint(D*NPL) calcint(DZ*NPL) calcint(D*NPL)) simpleGrading (1 1 1)

    //section 5
    hex (57 56 58 59 2 1 13 14) (calcint(D*NPL) calcint(D*NPL) calcint(DY*NPL)) simpleGrading (1 1 1)

    //section 6
    hex (61 60 62 63 26 25 37 38) (calcint(D*NPL) calcint(D*NPL) calcint(DY*NPL)) simpleGrading (1 1 1)

    //section 7
    hex (8 7 19 20 65 64 66 67) (calcint(D*NPL) calcint(D*NPL) calcint((LY-2*D-2*DY)*NPL)) simpleGrading (1 1 1)

    //section 8
    hex (32 31 43 44 69 68 70 71) (calcint(D*NPL) calcint(D*NPL) calcint((LY-2*D-2*DY)*NPL)) simpleGrading (1 1 1)

    //section 9
    hex (65 64 66 67 73 72 74 75) (calcint(D*NPL) calcint(D*NPL) calcint(D*NPL)) simpleGrading (1 1 1)
    
    //section 10
    hex (69 68 70 71 77 76 78 79) (calcint(D*NPL) calcint(D*NPL) calcint(D*NPL)) simpleGrading (1 1 1)

    //section 11
    hex (73 72 74 75 81 80 82 83) (calcint(D*NPL) calcint(D*NPL) calcint(DY*NPL)) simpleGrading (1 1 1)

    //section 12
    hex (77 76 78 79 85 84 86 87) (calcint(D*NPL) calcint(D*NPL) calcint(DY*NPL)) simpleGrading (1 1 1)

    //section 9-10
    hex (67 66 68 69 75 74 76 77) (calcint(D*NPL) calcint((LZ-2*D-2*DZ)*NPL) calcint(D*NPL)) simpleGrading (1 1 1)

    //section 13
    hex (89 88 64 65 91 90 72 73) (calcint(D*NPL) calcint(DZ*NPL) calcint(D*NPL)) simpleGrading (1 1 1)

    //section 14
    hex (71 70 92 93 79 78 94 95) (calcint(D*NPL) calcint(DZ*NPL) calcint(D*NPL)) simpleGrading (1 1 1)
    
   );



   patches
   (
        wall wallinlets
        (
                //patch wallinlet1
                (0 12 18 6)

                //patch wallinlet2
                (56 57 59 58)
        
                //patch wallinlet3
                (48 49 51 50)
        
                //patch wallinlet4
                (5 17 23 11)
        
                //patch wallinlet5
                (29 41 47 35)

                //patch wallinlet6
                (52 53 55 54)

                //patch wallinlet7
                (60 61 63 62)
        
                //patch wallinlet8
                (24 36 42 30)
        )

        wall walloutlets
        (
               //patch walloutlet1
               //(84 85 87 86)

                //patch walloutlet2
                (80 81 83 82)
        )

        patch outlet1
        (
                (84 85 87 86)
        )
   );

   mergePatchPairs
   (
   );
