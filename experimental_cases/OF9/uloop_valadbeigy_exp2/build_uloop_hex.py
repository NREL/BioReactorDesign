"""
Reproduce the case in "Hydrodynamic optimization of a newly designed and fabricated U-Loop bioreactor using Taguchi–ANOVA analysis", Valadbeigy et al., Biochemical Engineering Journal, July 2026

Open-top U-loop reactor as 3 stitchable gmsh blocks

Structured hex mesh everywhere except the U-loop<->tank junction

Blocks (all interfaces are perimeter-matched -> OpenFOAM integral `stitchMesh`):
  A  hex  : the U pipe
  B  tet  : U-loop <-> tank junction
            Two down-stubs (filleted where they meet the tank floor)
  C  hex  : structured-hex tank, extruded up to the open top (Z_TOP).
            Flat top face is the `outlet` boundary; sides = wall.

This output block{A,B,C}.{msh,vtk}
"""

import math

import gmsh
import numpy as np

# Geometrical parameters
R = 0.020  # DN40 pipe [m]
R_BEND = 0.045  # elbow centerline bend radius [m]
X_LEG = 0.063  # leg half spacing [m]
Z_HORIZ = 0.000  # bottom height [m]
Z_TANK = (
    0.8  # tank axis height (sets the tank floor Z_BOT = Z_TANK-R_TANK) [m]
)
R_TANK = 0.100  # degassing-tank radius (sets the box cross-section) [m]
TANK_LEN = 2 * R_TANK
Z_OUTLET = 1.3  # open-top outlet height (tank roof) [m]
FILLET_R = 0.01  # junction fillet radius [m]


def loop_pipe_length(include_tank=False):
    """Compute pipe length which is reported in the paper"""
    r_bend = R_BEND
    z_bend_top = Z_HORIZ + r_bend
    leg_top = Z_TANK if include_tank else (Z_TANK - R_TANK)
    leg = leg_top - z_bend_top
    arc = 0.5 * math.pi * r_bend
    horiz = 2.0 * (X_LEG - r_bend)
    return 2.0 * leg + 2.0 * arc + horiz


def reactor_volume(tank_fraction=1.0):
    v_pipe = math.pi * R**2 * loop_pipe_length(include_tank=False)
    v_tank = math.pi * R_TANK**2 * TANK_LEN
    return v_pipe + tank_fraction * v_tank


# ---  derived helper dimensions (I need that later)
Z_BEND_TOP = Z_HORIZ + R_BEND  # where the bottom legs meet the elbows
Z_BOT = Z_TANK - R_TANK  # tank floor  (box bottom)  = 0.7
Z_TOP = Z_OUTLET  # tank roof / open outlet   = 1.3
HX = TANK_LEN / 2.0  # tank box half-width along x
HY = R_TANK  # tank box half-width along y (cross-section)

STUB = 0.03  # how far do we stop before the legs at the filletted junction [m]
B_SLAB = (
    0.03  # how far do we extend the filletted junction into the hex tank [m]
)
Z_AB = Z_BOT - STUB  # A<->B interface (leg tops) [m]
Z_BC = Z_BOT + B_SLAB  # B<->C interface (tank square) [m]

# --- resolution
RI_FRAC = 0.5
N_SIDE = 6  # even -> circle/Pillow rims share nodes
N_RAD = max(1, round(N_SIDE * (1 - RI_FRAC) / (RI_FRAC * math.sqrt(2))))
H_AX = 0.004  # target axial cell size for the pipe sweeps
N_TANK = 30  # structured cells per tank-square edge
# FINER mesh at the junction is obtained with SMALLER JUNCTION_RES
JUNCTION_RES = 1.8


# ---- iterative mesh cleanup
N_LEG = max(1, round((Z_AB - Z_BEND_TOP) / H_AX))
N_ARC = max(1, round((R_BEND * math.pi / 2) / H_AX))
N_HOR = max(1, round(2 * (X_LEG - R_BEND) / H_AX))
N_HC = max(1, round((Z_TOP - Z_BC) / (2 * HX / N_TANK)))  # uniform tank cells


def _pillow(geo, cx, cy, cz, r, n_side, n_rad):
    """Pillow shape cylindrical mesh cross-section
    Normal direction is z (consistently with the block cylindrical meshing)"""
    ang = [math.pi / 4 + k * math.pi / 2 for k in range(4)]
    ri = RI_FRAC * r
    c = geo.addPoint(cx, cy, cz)
    Q = [
        geo.addPoint(cx + ri * math.cos(a), cy + ri * math.sin(a), cz)
        for a in ang
    ]
    A = [
        geo.addPoint(cx + r * math.cos(a), cy + r * math.sin(a), cz)
        for a in ang
    ]
    Qe = [geo.addLine(Q[i], Q[(i + 1) % 4]) for i in range(4)]
    Rad = [geo.addLine(Q[i], A[i]) for i in range(4)]
    Arc = [geo.addCircleArc(A[i], c, A[(i + 1) % 4]) for i in range(4)]
    surfs = [geo.addPlaneSurface([geo.addCurveLoop(Qe)])]
    for i in range(4):
        surfs.append(
            geo.addSurfaceFilling(
                [geo.addCurveLoop([Rad[i], Arc[i], -Rad[(i + 1) % 4], -Qe[i]])]
            )
        )
    for e in Qe + Arc:
        geo.mesh.setTransfiniteCurve(e, n_side + 1)
    for e in Rad:
        geo.mesh.setTransfiniteCurve(e, n_rad + 1)
    for s in surfs:
        geo.mesh.setTransfiniteSurface(s)
        geo.mesh.setRecombine(2, s)
    return surfs


def _isflat(s, idx, val, tol=1e-6):
    """
    True if surface *s* lies entirely on the plane coord[idx] == val.
    I.e. is a constant coordinate plane
    useful to check if what we extruded gives us a flat surface
    """
    bb = gmsh.model.getBoundingBox(2, s)
    return abs(bb[idx] - val) < tol and abs(bb[idx + 3] - val) < tol


def _tip(idx, val):
    """
    Find flat boundary surface after gmesh extrusion
    """
    vols = [t for _, t in gmsh.model.getEntities(3)]
    bnd = {
        t
        for _, t in gmsh.model.getBoundary(
            [(3, v) for v in vols], combined=True, oriented=False
        )
    }
    return [(2, s) for s in bnd if _isflat(s, idx, val)]


def _by(surfs, idx, val):
    """Filter surface to those lying on the plane coord[idx] == val."""
    return [s for s in surfs if _isflat(s, idx, val)]


def _cx(s):
    """Bounding box used to distinguish the left and right leg"""
    bb = gmsh.model.getBoundingBox(2, s)
    return 0.5 * (bb[0] + bb[3])


def _rims(surfs):
    """find the rims at the junction between the legs and the filleted tets
    and for the junction between tet block and hex tank block"""
    rim = set()
    for s in surfs:
        for _, cc in gmsh.model.getBoundary([(2, s)], oriented=False):
            rim.add(cc)
    return rim


def _boundary_surfs():
    """Returns volume IDs and their boundary surface IDs."""
    vols = [t for _, t in gmsh.model.getEntities(3)]
    return vols, [
        t
        for _, t in gmsh.model.getBoundary(
            [(3, v) for v in vols], combined=True, oriented=False
        )
    ]


def _cell_volume():
    """Sum of all 3-D cell volumes"""
    tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = coords.reshape(-1, 3)
    idx = {int(t): i for i, t in enumerate(tags)}
    npe = {4: 4, 5: 8, 6: 6, 7: 5}
    fans = {
        4: [(0, 1, 2, 3)],
        5: [
            (0, 1, 2, 6),
            (0, 2, 3, 6),
            (0, 3, 7, 6),
            (0, 7, 4, 6),
            (0, 4, 5, 6),
            (0, 5, 1, 6),
        ],
        6: [(0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5)],
        7: [(0, 1, 2, 4), (0, 2, 3, 4)],
    }
    total = 0.0
    ets, _, enodes = gmsh.model.mesh.getElements(3)
    for et, en in zip(ets, enodes):
        conn = np.array([idx[int(t)] for t in en]).reshape(-1, npe[et])
        P = coords[conn]
        for a, b, c, d in fans[et]:
            v = P[:, a], P[:, b], P[:, c], P[:, d]
            total += np.abs(
                np.einsum(
                    "ij,ij->i", np.cross(v[1] - v[0], v[2] - v[0]), v[3] - v[0]
                )
            ).sum()
    return total / 6.0


def _write(path, tag):
    """Write Gmesh object to .msh and print summary"""
    TYPE = {4: "tet", 5: "hex", 6: "prism", 7: "pyramid"}
    ets, etags, _ = gmsh.model.mesh.getElements(3)
    counts = {TYPE.get(e, e): len(t) for e, t in zip(ets, etags)}
    vol = _cell_volume()
    print(f"[block {tag}] cells={counts}  volume={vol * 1e3:.3f} L")
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(path)
    gmsh.write(path.rsplit(".", 1)[0] + ".vtk")
    return vol


# --- U pipe (structured hex)
def build_block_A(path):
    """Extrude the pillow cross-section down in sequence
    1) left leg
    2) 90 deg elbow
    3) bottom leg
    4) second 90 degree elbow
    5) up the right leg

    the two leg-top meet the filleted mesh as z=Z_AB"""
    gmsh.initialize()
    gmsh.model.add("A")
    gmsh.option.setNumber("General.Terminal", 0)
    geo = gmsh.model.geo

    disk = _pillow(geo, -X_LEG, 0, Z_AB, R, N_SIDE, N_RAD)
    geo.extrude(
        [(2, s) for s in disk],
        0,
        0,
        -(Z_AB - Z_BEND_TOP),
        numElements=[N_LEG],
        recombine=True,
    )
    geo.synchronize()

    # left elbow: revolve the leg-bottom disk about y through the bend centre
    geo.revolve(
        _tip(2, Z_BEND_TOP),
        -X_LEG + R_BEND,
        0,
        Z_BEND_TOP,
        0,
        -1,
        0,
        math.pi / 2,
        numElements=[N_ARC],
        recombine=True,
    )
    geo.synchronize()

    # bottom horizontal run: extrude +x
    geo.extrude(
        _tip(0, -X_LEG + R_BEND),
        2 * (X_LEG - R_BEND),
        0,
        0,
        numElements=[N_HOR],
        recombine=True,
    )
    geo.synchronize()

    # right elbow
    geo.revolve(
        _tip(0, X_LEG - R_BEND),
        X_LEG - R_BEND,
        0,
        Z_BEND_TOP,
        0,
        -1,
        0,
        math.pi / 2,
        numElements=[N_ARC],
        recombine=True,
    )
    geo.synchronize()

    # right leg: extrude +z up to Z_AB
    geo.extrude(
        _tip(2, Z_BEND_TOP),
        0,
        0,
        Z_AB - Z_BEND_TOP,
        numElements=[N_LEG],
        recombine=True,
    )
    geo.synchronize()

    vols, bnd = _boundary_surfs()
    iface = _by(bnd, 2, Z_AB)
    legL = [s for s in iface if _cx(s) < 0]
    legR = [s for s in iface if _cx(s) > 0]
    walls = [s for s in bnd if s not in iface]
    gmsh.model.addPhysicalGroup(3, vols, name="pipeU")
    gmsh.model.addPhysicalGroup(2, legL, name="int_A_legL")
    gmsh.model.addPhysicalGroup(2, legR, name="int_A_legR")
    gmsh.model.addPhysicalGroup(2, walls, name="wall_A")

    gmsh.model.mesh.generate(3)
    vol = _write(path, "A")
    gmsh.finalize()
    return vol


# ---- block B: U-loop <-> tank junction
def build_block_B(path):
    """Tet-meshed junction connecting the U-pipe (A) to the hex tank (C).
      1. Rectangular from Z_BOT to Z_BC (the tank-floor transition layer).
      2. Two cylindrical partial leds
      3. Fillet

    Interface matching (that was the hard part!)
      - Bottom circles (int_B_legL/R at Z_AB): rim nodes match A's pillow perimeter.
      - Top rectangle (int_B_top at Z_BC): rim nodes match C's structured grid edges.
    """

    gmsh.initialize()
    gmsh.model.add("B")
    gmsh.option.setNumber("General.Terminal", 0)
    occ = gmsh.model.occ

    pen = 0.4 * (Z_BC - Z_BOT)
    slab = occ.addBox(-HX, -HY, Z_BOT, 2 * HX, 2 * HY, Z_BC - Z_BOT)
    stubs = [
        occ.addCylinder(sx, 0, Z_AB, 0, 0, (Z_BOT - Z_AB) + pen, R)
        for sx in (-X_LEG, X_LEG)
    ]
    S, _ = occ.fuse([(3, slab)], [(3, s) for s in stubs])
    occ.synchronize()
    vol = S[0][1]

    ring = []
    for _, e in gmsh.model.getEntities(1):
        ex, _ey, ez = occ.getCenterOfMass(1, e)
        x0, _, _, x1, _, _ = gmsh.model.getBoundingBox(1, e)
        if (
            abs(ez - Z_BOT) < 1e-3
            and abs(abs(ex) - X_LEG) < 0.02
            and (x1 - x0) < 3 * R
        ):
            ring.append(e)
    occ.fillet([vol], ring, [FILLET_R])
    occ.synchronize()

    vols, bnd = _boundary_surfs()
    bot = _by(bnd, 2, Z_AB)  # two pipe circles -> A
    top = _by(bnd, 2, Z_BC)  # tank square -> C
    walls = [s for s in bnd if s not in bot and s not in top]
    legL = [s for s in bot if occ.getCenterOfMass(2, s)[0] < 0]
    legR = [s for s in bot if occ.getCenterOfMass(2, s)[0] > 0]

    for s in bot:  # match each stub rim to A (4*N_SIDE)
        rc = _rims([s])
        per = max(1, round(4 * N_SIDE / len(rc)))
        for cc in rc:
            gmsh.model.mesh.setTransfiniteCurve(cc, per + 1)
    for cc in _rims(top):  # match tank square rim to C (N_TANK/edge)
        gmsh.model.mesh.setTransfiniteCurve(cc, N_TANK + 1)

    gmsh.model.addPhysicalGroup(3, vols, name="juncB")
    gmsh.model.addPhysicalGroup(2, legL, name="int_B_legL")
    gmsh.model.addPhysicalGroup(2, legR, name="int_B_legR")
    gmsh.model.addPhysicalGroup(2, top, name="int_B_top")
    gmsh.model.addPhysicalGroup(2, walls, name="wall_B")
    gmsh.option.setNumber("Mesh.MeshSizeMax", R / N_SIDE * JUNCTION_RES)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Netgen")
    vol = _write(path, "B")
    gmsh.finalize()
    return vol


# --- block C: hex tank
def build_block_C(path):
    """Structured-hex tank extruded from Z_BC to Z_TOP.

    1) N_TANK nodes per edge, matching block B's top
    2) Extrude +z to Z_TOP with N_HC uniform layers.

    Top face is the open outlet boundary;
    Sides are wall_C;
    Bottom is for stitching to block B.
    """
    gmsh.initialize()
    gmsh.model.add("C")
    gmsh.option.setNumber("General.Terminal", 0)
    geo = gmsh.model.geo

    p = [
        geo.addPoint(-HX, -HY, Z_BC),
        geo.addPoint(HX, -HY, Z_BC),
        geo.addPoint(HX, HY, Z_BC),
        geo.addPoint(-HX, HY, Z_BC),
    ]
    l = [geo.addLine(p[i], p[(i + 1) % 4]) for i in range(4)]
    sq = geo.addPlaneSurface([geo.addCurveLoop(l)])
    for e in l:
        geo.mesh.setTransfiniteCurve(e, N_TANK + 1)
    geo.mesh.setTransfiniteSurface(sq)
    geo.mesh.setRecombine(2, sq)
    geo.extrude(
        [(2, sq)], 0, 0, Z_TOP - Z_BC, numElements=[N_HC], recombine=True
    )
    geo.synchronize()

    vols, bnd = _boundary_surfs()
    bot = _by(bnd, 2, Z_BC)
    top = _by(bnd, 2, Z_TOP)  # open top -> outlet boundary (no stitch)
    walls = [s for s in bnd if s not in bot and s not in top]
    gmsh.model.addPhysicalGroup(3, vols, name="tank")
    gmsh.model.addPhysicalGroup(2, bot, name="int_C_bot")
    gmsh.model.addPhysicalGroup(2, top, name="outlet")
    gmsh.model.addPhysicalGroup(2, walls, name="wall_C")
    gmsh.model.mesh.generate(3)
    vol = _write(path, "C")
    gmsh.finalize()
    return vol


# ---- verify that the junction that stitch mesh will operate on has consistent
# face perimeter
def _plane_nodes(path, zval):
    """All mesh nodes at z == zval from a .msh file, returned as (x, y) pairs."""
    gmsh.initialize()
    gmsh.open(path)
    _, coords, _ = gmsh.model.mesh.getNodes()
    coords = coords.reshape(-1, 3)
    pts = [(x, y) for x, y, z in coords if abs(z - zval) < 1e-6]
    gmsh.finalize()
    return pts


def _circle_rim(pts, cx, r):
    """Subset of (x, y) points lying on a circle centred at (cx, 0) with radius r."""
    return sorted(
        (round(x, 9), round(y, 9))
        for x, y in pts
        if abs(math.hypot(x - cx, y) - r) < 1e-4
    )


def _square_rim(pts, hx, hy):
    """Subset of (x, y) points lying on the perimeter of a [-hx,hx] x [-hy,hy] rectangle."""
    return sorted(
        (round(x, 9), round(y, 9))
        for x, y in pts
        if abs(abs(x) - hx) < 1e-4 or abs(abs(y) - hy) < 1e-4
    )


def _verify(name, a, b, tol=1e-9):
    """Assert two rim point sets have the same count and are coincident within tol."""
    assert len(a) == len(
        b
    ), f"{name}: rim node COUNT differs (A={len(a)}, B={len(b)}) -> areas differ."
    worst = max(
        min(math.hypot(px - qx, py - qy) for qx, qy in b) for px, py in a
    )
    ok = worst < tol
    print(
        f"[verify {name}] n={len(a)}  max rim gap={worst:.2e} m  {'OK' if ok else 'FAIL'}"
    )
    assert ok, f"{name}: rims not coincident (gap {worst:.1e} > {tol})."


def verify_interfaces():
    """Check that A-B circle rims and B-C square rim match node-for-node across blocks."""
    A_ab = _plane_nodes("blockA.msh", Z_AB)
    B_ab = _plane_nodes("blockB.msh", Z_AB)
    _verify(
        "A-B legL", _circle_rim(A_ab, -X_LEG, R), _circle_rim(B_ab, -X_LEG, R)
    )
    _verify(
        "A-B legR", _circle_rim(A_ab, X_LEG, R), _circle_rim(B_ab, X_LEG, R)
    )
    B_bc = _plane_nodes("blockB.msh", Z_BC)
    C_bc = _plane_nodes("blockC.msh", Z_BC)
    _verify("B-C square", _square_rim(B_bc, HX, HY), _square_rim(C_bc, HX, HY))


# ---- main
if __name__ == "__main__":
    print(
        f"[resolution] N_SIDE={N_SIDE} N_RAD={N_RAD} N_LEG={N_LEG} N_ARC={N_ARC} "
        f"N_HOR={N_HOR} N_TANK={N_TANK} N_HC={N_HC}"
    )
    vols = {
        "A": build_block_A("blockA.msh"),
        "B": build_block_B("blockB.msh"),
        "C": build_block_C("blockC.msh"),
    }
    verify_interfaces()

    box_tank = (2 * HX) * (2 * HY) * (Z_TOP - Z_BOT)
    print("=" * 70)
    print(
        f"[pipe length]  incl. tank = {loop_pipe_length(True):.4f} m   "
        f"excl. tank = {loop_pipe_length(False):.4f} m"
    )
    print(
        f"[ieactor volume] this mesh (open-top box tank, blocks A-C) "
        f"= {sum(vols.values()) * 1e3:.3f} L"
    )
    print(
        f"                 of which the box tank alone = {box_tank * 1e3:.3f} L"
    )
    print(
        f"[open top] outlet = full tank roof at z={Z_TOP:.3f} m "
        f"({2 * HX:.3f} x {2 * HY:.3f} m)"
    )
    print("[write] block{A,B,C}.{msh,vtk} -> will stitch next")
