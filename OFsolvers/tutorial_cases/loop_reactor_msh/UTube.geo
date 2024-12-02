SetFactory("OpenCASCADE");

//+ Bounds of the domain
BX = DefineNumber[ 10., Name "Parameters/BX"];
BY = DefineNumber[ 5., Name "Parameters/BY" ];
BZ = DefineNumber[ 5., Name "Parameters/BZ" ];

//+ Pipe radius
rad = DefineNumber[ 0.5, Name "Parameters/rad" ];

//+ Radius of the bends
brad = DefineNumber[ 1.0, Name "Parameters/brad" ];

//+ Extension
lx = DefineNumber[ BX - 2*(rad+brad), Name "Parameters/lx"];
ly = DefineNumber[ BY - 2*(rad+brad), Name "Parameters/ly"];
lz = DefineNumber[ BZ - 2*(rad+brad), Name "Parameters/lz"];


// Create Points ---------------------------------------

// Circle (not used)
Point(1) = {brad*2 + lx, -rad, 0, 0.1};
Point(2) = {brad*2 + lx,  0, 0, 0.1};
Point(3) = {brad*2 + lx,  rad, 0, 0.1};

// Path

Point(4) = {brad*2 + lx, 0, lz/2, 0.1};
Point(5) = {brad + lx, 0, lz/2 + brad, 0.1};
Point(6) = {brad, 0, lz/2 + brad, 0.1};
Point(7) = {0, brad, lz/2 + brad, 0.1};
Point(8) = {0, ly + brad, lz/2 + brad, 0.1};
Point(9) = {0, ly + 2*brad, lz/2 , 0.1};
Point(10) = {0, ly + 2*brad, 0 , 0.1};

// Curvature Centers
Point(11) = {lx + brad , 0, lz/2, 0.1};
Point(12) = {brad, brad, lz/2 + brad, 0.1 };
Point(13) = {0, ly + brad, lz/2, 0.1};

// Create lines -----------------------------------------

//+
Line(3) = {2, 4};
//+
Circle(5) = {4, 11, 5};
//+
Line(6) = {5, 6};
//+
Circle(7) = {6, 12, 7};
//+
Line(8) = {7, 8};
//+
Circle(9) = {8, 13, 9};
//+
Line(10) = {9, 10};

// Create volume ------------------------------------------
Disk(1) = {brad*2 + lx, 0, 0, rad, rad};
//+
Wire(4) = {3, 5, 6, 7, 8, 9, 10};
Extrude { Surface{1}; } Using Wire {4}

//+
// Apply symmetry across the plane x = 0 to create a mirrored copy
Symmetry {0, 0, 1, 0} {
    Duplicata {
        Volume{1};
    }
}



//+ Inlets
Cylinder(10) = {lx/2.0, -rad - 0.2, lz  , 0, rad, 0, 0.2, 2*Pi};
Cylinder(11) = {lx/2.0, -rad - 0.2, -lz , 0, rad, 0, 0.2, 2*Pi};
Cylinder(12) = {lx + 2*brad, -rad - 0.2, 0 , 0, rad, 0, 0.2, 2*Pi};

//+ Outlets
Cylinder(100) = {0, ly + 2*brad,  lz/2.0, 0, 2*rad, 0, 0.2, 2*Pi};
Cylinder(101) = {0, ly + 2*brad,  -lz/2.0, 0, 2*rad, 0, 0.2, 2*Pi};

//+
mergedVolume() = BooleanUnion{ Volume{100}; Volume{101}; Volume{1}; Volume{10}; Volume{11}; Volume{12}; Delete; }{ Volume{2}; Delete; };

// Assign a physical group to the target volume
Physical Volume("fluid") = {mergedVolume()};
//+
Physical Surface("inlets", 102) = {18, 24, 25};
//+
Physical Surface("outlets", 103) = {5, 12};
//+
Physical Surface("walls", 104) = {6, 3, 4, 2, 7, 13, 9, 10, 16, 19, 23, 15, 8, 11, 14, 17, 20, 22, 21};


//Mesh.FromPhysicalGroups = 1;

// Set mesh size parameters for finer control over mesh refinement
Mesh.MeshSizeMin = 0.05;
Mesh.MeshSizeMax = 0.1;

// **Force first-order elements**
Mesh.ElementOrder = 1;
Mesh.HighOrderOptimize = 0;

// Generate the 3D mesh
Mesh 3;

// Refine the mesh globally
//RefineMesh;


// Save the mesh in MSH 2.2 format (if required)
Mesh.MshFileVersion = 2.2;

// Save the mesh
//Mesh.SaveAll = 0;
Save "UTube.msh";
//Save "UTube.stl"; */


