import os
import tempfile

from bird.postprocess.post_quantities import compute_turbulent_diffusivity


def write_scalar_case(root, fields):
    """Write each {name: values} as a nonuniform scalar field in the 0 folder."""
    os.makedirs(os.path.join(root, "0"), exist_ok=True)
    for name, values in fields.items():
        entries = "\n".join(f"{v:.10g}" for v in values)
        with open(os.path.join(root, "0", name), "w") as f:
            f.write("FoamFile\n{\n    format      ascii;\n")
            f.write("    class       volScalarField;\n")
            f.write(f"    object      {name};\n}}\n\n")
            f.write("dimensions      [0 0 0 0 0 0 0];\n\n")
            f.write(
                "internalField   nonuniform List<scalar> \n"
                f"{len(values)}\n(\n{entries}\n)\n;\n"
            )


def test_compute_turbulent_diffusivity():
    # alpha.liquid selects cells 0,1 (cell 2 is gas). D = alphat/rho = [1,2,3];
    # volume-weighted over the liquid = (1*1 + 2*1)/(1+1) = 1.5
    with tempfile.TemporaryDirectory() as tmp:
        write_scalar_case(
            tmp,
            {
                "alpha.liquid": [1.0, 1.0, 0.0],
                "alphat.liquid": [2.0, 4.0, 6.0],
                "rho.liquid": [2.0, 2.0, 2.0],
                "V": [1.0, 1.0, 1.0],
            },
        )
        d_turb, _ = compute_turbulent_diffusivity(tmp, "0")
    assert abs(d_turb - 1.5) < 1e-10

    # no alphat.liquid -> nut.liquid / Prt (0.85). nut = [0.85, 1.70] -> [1, 2]
    with tempfile.TemporaryDirectory() as tmp:
        write_scalar_case(
            tmp,
            {
                "alpha.liquid": [1.0, 1.0],
                "nut.liquid": [0.85, 1.70],
                "rho.liquid": [1.0, 1.0],
                "V": [1.0, 3.0],
            },
        )
        d_fallback, _ = compute_turbulent_diffusivity(tmp, "0")
    # volume-weighted (1*1 + 2*3)/(1+3) = 1.75
    assert abs(d_fallback - 1.75) < 1e-10
