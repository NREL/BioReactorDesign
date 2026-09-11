import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from prettyPlot.plotting import plt, pretty_labels

from bird.postprocess.post_quantities import (
    _get_ind_gas,
    _get_ind_liq,
    compute_ave_bubble_diam,
    compute_ave_conc_liq,
    compute_ave_liquid_density,
    compute_ave_liquid_velocity,
    compute_ave_y_liq,
    compute_fitted_kl,
    compute_fitted_kla,
    compute_gas_holdup,
    compute_instantaneous_kl,
    compute_instantaneous_kla,
    compute_superficial_gas_velocity,
    interfacial_area,
)


def write_uniform_alpha_case(root, alpha_liq, cell_volumes, y_liq):
    """
    Write a minimal case whose alpha.liquid is a uniform field

    Reading a uniform field gives a single value rather than one value per
    cell, so this exercises the branch where every cell is selected at once.
    """

    def write_field(name, body):
        with open(os.path.join(root, "0", name), "w") as f:
            f.write("FoamFile\n{\n    format      ascii;\n")
            f.write("    class       volScalarField;\n")
            f.write(f"    object      {name};\n}}\n\n")
            f.write("dimensions      [0 0 0 0 0 0 0];\n\n")
            f.write(body)

    def nonuniform(values):
        entries = "\n".join(f"{value:.10g}" for value in values)
        return (
            "internalField   nonuniform List<scalar> \n"
            f"{len(values)}\n(\n{entries}\n)\n;\n"
        )

    os.makedirs(os.path.join(root, "0"), exist_ok=True)
    write_field("V", nonuniform(cell_volumes))
    write_field("alpha.liquid", f"internalField   uniform {alpha_liq};\n")
    write_field("O2.liquid", nonuniform(y_liq))


def test_compute_gh():
    """
    Test for gas holdup calculation
    """
    case_folder = os.path.join(
        Path(__file__).parent,
        " .. ".strip(),
        " .. ".strip(),
        "bird",
        "postprocess",
        "data_conditional_mean",
    )
    kwargs = {"case_folder": case_folder, "n_cells": None, "volume_time": "1"}
    field_dict = {}
    gh, field_dict = compute_gas_holdup(
        time_folder="1", field_dict=field_dict, **kwargs
    )
    field_dict = {}
    gh, field_dict = compute_gas_holdup(
        time_folder="79", field_dict=field_dict, **kwargs
    )

    # Make sure None arguments are correctly handled
    n_cells = len(field_dict["alpha.liquid"])
    time_folder = "79"
    gh1, _ = compute_gas_holdup(
        case_folder=case_folder, time_folder=time_folder
    )
    gh2, _ = compute_gas_holdup(
        case_folder=case_folder, n_cells=n_cells, time_folder=time_folder
    )

    # Results need to be exactly the same
    assert abs(gh1 - gh) < 1e-12
    assert abs(gh2 - gh) < 1e-12

    # A uniform alpha.liquid selects every cell, so the holdup is 1 - alpha
    cell_volumes = [1.0, 2.0, 3.0, 4.0]
    with tempfile.TemporaryDirectory() as tmpdirname:
        write_uniform_alpha_case(
            tmpdirname,
            alpha_liq=0.9,
            cell_volumes=cell_volumes,
            y_liq=[0.1, 0.2, 0.3, 0.4],
        )
        gh_unif, _ = compute_gas_holdup(
            case_folder=tmpdirname, time_folder="0", volume_time="0"
        )
    assert abs(gh_unif - 0.1) < 1e-12


def test_compute_diam():
    """
    Test for bubble diameter calculation
    """
    case_folder = os.path.join(
        Path(__file__).parent,
        " .. ".strip(),
        " .. ".strip(),
        "bird",
        "postprocess",
        "data_conditional_mean",
    )
    kwargs = {"case_folder": case_folder, "n_cells": None, "volume_time": "1"}
    field_dict = {}
    diam, field_dict = compute_ave_bubble_diam(
        time_folder="1", field_dict=field_dict, **kwargs
    )
    field_dict = {}
    diam, field_dict = compute_ave_bubble_diam(
        time_folder="79", field_dict=field_dict, **kwargs
    )

    # Make sure None arguments are correctly handled
    n_cells = len(field_dict["d.gas"])
    time_folder = "79"
    diam1, _ = compute_ave_bubble_diam(
        case_folder=case_folder, time_folder=time_folder
    )
    diam2, _ = compute_ave_bubble_diam(
        case_folder=case_folder, n_cells=n_cells, time_folder=time_folder
    )

    # Results need to be exactly the same
    assert abs(diam1 - diam) < 1e-12
    assert abs(diam2 - diam) < 1e-12


def test_compute_superficial_gas_velocity():
    """
    Test for superficial gas velocity calculation
    """
    case_folder = os.path.join(
        Path(__file__).parent,
        " .. ".strip(),
        " .. ".strip(),
        "bird",
        "postprocess",
        "data_conditional_mean/",
    )
    kwargs = {
        "case_folder": case_folder,
        "n_cells": None,
        "volume_time": "1",
        "direction": 1,
        "cell_centers_file": "meshCellCentres_1.obj",
    }
    field_dict = {}
    sup_vel, field_dict = compute_superficial_gas_velocity(
        time_folder="79", field_dict=field_dict, **kwargs
    )

    # Make sure None arguments are correctly handled
    n_cells = len(field_dict["V"])
    time_folder = "79"
    sup_vel1, _ = compute_superficial_gas_velocity(
        case_folder=case_folder, time_folder=time_folder, direction=1
    )
    sup_vel2, _ = compute_superficial_gas_velocity(
        case_folder=case_folder,
        n_cells=n_cells,
        time_folder=time_folder,
        direction=1,
    )

    # Results need to be exactly the same
    assert abs(sup_vel1 - sup_vel) < 1e-12
    assert abs(sup_vel2 - sup_vel) < 1e-12

    # Make sure that we don't use paraview if not possible
    polyMesh_dir = os.path.join(case_folder, "constant", "polyMesh")
    shutil.move(os.path.join(polyMesh_dir, "faces"), ".")
    sup_vel4, _ = compute_superficial_gas_velocity(
        case_folder=case_folder,
        time_folder=time_folder,
        direction=1,
        use_pv=True,
    )
    # Results need to be exactly the same
    assert abs(sup_vel4 - sup_vel) < 1e-12
    shutil.move("faces", polyMesh_dir)


def test__superficial_velocity_pv():
    """
    Test the paraview branch of the superficial gas velocity

    Skipped when the installed paraview cannot read the case (it needs a
    complete polyMesh and the case path to end with a separator)
    """
    case_folder = os.path.join(
        Path(__file__).parent,
        " .. ".strip(),
        " .. ".strip(),
        "bird",
        "postprocess",
        "data_conditional_mean/",
    )
    sup_vel_np, _ = compute_superficial_gas_velocity(
        case_folder=case_folder,
        time_folder="79",
        direction=1,
        volume_time="1",
    )
    try:
        sup_vel_pv, _ = compute_superficial_gas_velocity(
            case_folder=case_folder,
            time_folder="79",
            direction=1,
            use_pv=True,
        )
    except (AssertionError, ImportError):
        pytest.skip("paraview cannot read the case in this environment")
    # The paraview and numpy methods should agree to within 1%
    assert abs((sup_vel_pv - sup_vel_np) / sup_vel_np) < 0.01


def test_ave_y_liq():
    """
    Test for liquid volume averaged species mass fraction
    """
    case_folder = os.path.join(
        Path(__file__).parent,
        " .. ".strip(),
        " .. ".strip(),
        "bird",
        "postprocess",
        "data_conditional_mean",
    )
    kwargs = {
        "time_folder": "79",
        "case_folder": case_folder,
        "n_cells": None,
        "volume_time": "1",
    }
    field_dict = {}
    ave_y_co2, field_dict = compute_ave_y_liq(
        species_name="CO2", field_dict=field_dict, **kwargs
    )
    ave_y_co, field_dict = compute_ave_y_liq(
        species_name="CO", field_dict=field_dict, **kwargs
    )
    ave_y_h2, field_dict = compute_ave_y_liq(
        species_name="H2", field_dict=field_dict, **kwargs
    )

    # Make sure None arguments are correctly handled
    n_cells = len(field_dict["H2.liquid"])
    time_folder = kwargs["time_folder"]
    ave_y_h21, _ = compute_ave_y_liq(
        case_folder=case_folder, time_folder=time_folder, species_name="H2"
    )
    ave_y_h22, _ = compute_ave_y_liq(
        case_folder=case_folder,
        n_cells=n_cells,
        time_folder=time_folder,
        species_name="H2",
    )

    # Results need to be exactly the same
    assert abs(ave_y_h21 - ave_y_h2) < 1e-12

    # With a uniform alpha.liquid the average must run over every cell, so it
    # is the volume weighted mean of the mass fraction. Selecting only the
    # first cell would give 0.1 instead
    cell_volumes = np.array([1.0, 2.0, 3.0, 4.0])
    y_liq = np.array([0.1, 0.2, 0.3, 0.4])
    with tempfile.TemporaryDirectory() as tmpdirname:
        write_uniform_alpha_case(
            tmpdirname,
            alpha_liq=0.9,
            cell_volumes=cell_volumes,
            y_liq=y_liq,
        )
        ave_y_unif, _ = compute_ave_y_liq(
            case_folder=tmpdirname,
            time_folder="0",
            species_name="O2",
            volume_time="0",
        )
    expected = np.sum(y_liq * cell_volumes) / np.sum(cell_volumes)
    assert abs(ave_y_unif - expected) < 1e-12
    assert abs(ave_y_h22 - ave_y_h2) < 1e-12


def test_ave_conc_liq():
    """
    Test for liquid volume averaged species concentration
    """
    case_folder = os.path.join(
        Path(__file__).parent,
        " .. ".strip(),
        " .. ".strip(),
        "bird",
        "postprocess",
        "data_conditional_mean",
    )
    kwargs = {
        "time_folder": "79",
        "case_folder": case_folder,
        "n_cells": None,
        "volume_time": "1",
    }
    field_dict = {}
    ave_conc_co2, field_dict = compute_ave_conc_liq(
        species_name="CO2",
        field_dict=field_dict,
        **kwargs,
    )
    ave_conc_co, field_dict = compute_ave_conc_liq(
        species_name="CO",
        field_dict=field_dict,
        **kwargs,
    )
    ave_conc_h2, field_dict = compute_ave_conc_liq(
        species_name="H2",
        field_dict=field_dict,
        **kwargs,
    )
    # Make sure None arguments are correctly handled
    n_cells = len(field_dict["H2.liquid"])
    time_folder = kwargs["time_folder"]
    ave_conc_h21, _ = compute_ave_conc_liq(
        case_folder=case_folder,
        time_folder=time_folder,
        species_name="H2",
    )
    ave_conc_h22, _ = compute_ave_conc_liq(
        case_folder=case_folder,
        time_folder=time_folder,
        species_name="H2",
        n_cells=n_cells,
    )

    # Results need to be exactly the same
    assert abs(ave_conc_h21 - ave_conc_h2) < 1e-12
    assert abs(ave_conc_h22 - ave_conc_h2) < 1e-12


def test_instantaneous_kla():
    """
    Test for instantaneous kla calculation
    """
    case_folder = os.path.join(
        Path(__file__).parent,
        " .. ".strip(),
        " .. ".strip(),
        "bird",
        "postprocess",
        "data_conditional_mean",
    )
    field_dict = {}
    kla_spec1, cstar_spec1, _ = compute_instantaneous_kla(
        species_names=["CO2", "CO", "H2"],
        case_folder=case_folder,
        time_folder="79",
        n_cells=None,
        volume_time="1",
    )
    kla_spec2, cstar_spec2, _ = compute_instantaneous_kla(
        species_names=["CO2"],
        case_folder=case_folder,
        time_folder="79",
        n_cells=None,
        volume_time="1",
    )
    # Make sure list of species allows to compute what we expect
    assert abs(kla_spec2["CO2"] - kla_spec1["CO2"]) / kla_spec1["CO2"] < 1e-6
    assert (
        abs(cstar_spec2["CO2"] - cstar_spec1["CO2"]) / cstar_spec1["CO2"]
        < 1e-6
    )
    kla_spec3, cstar_spec3, _ = compute_instantaneous_kla(
        species_names=["CO2"],
        case_folder=case_folder,
        time_folder="79",
    )
    # Make sure None arguments are correctly handled
    assert abs(kla_spec3["CO2"] - kla_spec1["CO2"]) / kla_spec1["CO2"] < 1e-6
    assert (
        abs(cstar_spec3["CO2"] - cstar_spec1["CO2"]) / cstar_spec1["CO2"]
        < 1e-6
    )
    kla_spec4, cstar_spec4, _ = compute_instantaneous_kla(
        species_names=["CO2"],
        case_folder=case_folder,
        time_folder="80",
    )
    # Make sure values change over time
    assert abs(kla_spec4["CO2"] - kla_spec1["CO2"]) / kla_spec1["CO2"] > 1e-6
    assert (
        abs(cstar_spec4["CO2"] - cstar_spec1["CO2"]) / cstar_spec1["CO2"]
        > 1e-6
    )
    kla_spec5, cstar_spec5, _ = compute_instantaneous_kla(
        species_names="CO2",
        case_folder=case_folder,
        time_folder="80",
    )
    # Make sure values change over time
    assert abs(kla_spec5["CO2"] - kla_spec1["CO2"]) / kla_spec1["CO2"] > 1e-6
    assert (
        abs(cstar_spec5["CO2"] - cstar_spec1["CO2"]) / cstar_spec1["CO2"]
        > 1e-6
    )


def test_fitted_kla():
    """
    Test for fitted kla calculation
    """
    case_folder = os.path.join(
        Path(__file__).parent,
        " .. ".strip(),
        " .. ".strip(),
        "bird",
        "postprocess",
        "data_conditional_mean",
    )
    field_dict = {}
    # Check that assertion error is sent if too few snapshots
    try:
        kla_spec1, cstar_spec1, _ = compute_fitted_kla(
            species_names=["CO2"],
            case_folder=case_folder,
        )
    except AssertionError:
        pass

    # Create dummy time folders
    for time_folder in [str(entry) for entry in range(81, 89)]:
        shutil.copytree(
            os.path.join(case_folder, "80"),
            os.path.join(case_folder, time_folder),
        )
    kla_spec1, cstar_spec1, _ = compute_fitted_kla(
        species_names=["CO2"],
        case_folder=case_folder,
        num_warmup=100,
        num_samples=100,
    )
    for time_folder in [str(entry) for entry in range(81, 89)]:
        shutil.rmtree(os.path.join(case_folder, time_folder))

    # Create dummy time folders to trigger bootstrapping
    for time_folder in [str(entry) for entry in range(81, 91)]:
        shutil.copytree(
            os.path.join(case_folder, "80"),
            os.path.join(case_folder, time_folder),
        )
    kla_spec1, cstar_spec1, _ = compute_fitted_kla(
        species_names=["CO2"],
        case_folder=case_folder,
        num_warmup=100,
        num_samples=100,
    )
    for time_folder in [str(entry) for entry in range(81, 91)]:
        shutil.rmtree(os.path.join(case_folder, time_folder))


def test_get_ind_gas():
    """
    Test for the selection of pure gas cells
    """
    case_folder = os.path.join(
        Path(__file__).parent,
        " .. ".strip(),
        " .. ".strip(),
        "bird",
        "postprocess",
        "data_conditional_mean",
    )
    n_cells = 137980
    ind_gas, field_dict = _get_ind_gas(
        case_folder=case_folder, time_folder="80"
    )
    ind_liq, _ = _get_ind_liq(case_folder=case_folder, time_folder="80")

    # Same flat shape as the liquid counterpart
    assert ind_gas.ndim == 1
    assert ind_liq.ndim == 1
    # Gas and liquid cells partition the domain
    assert len(ind_gas) + len(ind_liq) == n_cells
    assert len(np.intersect1d(ind_gas, ind_liq)) == 0
    # The selection agrees with the field it was built from
    alpha_liq = field_dict["alpha.liquid"]
    assert np.all(alpha_liq[ind_gas] <= 0.5)
    assert np.all(alpha_liq[ind_liq] > 0.5)
    # Reading again reuses the cached indices
    ind_gas_again, _ = _get_ind_gas(
        case_folder=case_folder, time_folder="80", field_dict=field_dict
    )
    assert np.array_equal(ind_gas_again, ind_gas)

    # A uniform liquid field leaves no gas cell to select
    cell_volumes = [1.0, 2.0, 3.0, 4.0]
    with tempfile.TemporaryDirectory() as tmpdirname:
        write_uniform_alpha_case(
            tmpdirname,
            alpha_liq=0.9,
            cell_volumes=cell_volumes,
            y_liq=[0.1, 0.2, 0.3, 0.4],
        )
        ind_gas_unif, _ = _get_ind_gas(case_folder=tmpdirname, time_folder="0")
        ind_liq_unif, _ = _get_ind_liq(case_folder=tmpdirname, time_folder="0")
    assert len(ind_gas_unif) == 0
    # A uniform liquid field needs no filtering at all, rather than selecting
    # a single cell
    assert ind_liq_unif is None


def _write_liquid_fields(root, u_vector, rho, cell_volumes):
    """Minimal case: all-liquid, uniform U.liquid and rho, nonuniform V."""
    os.makedirs(os.path.join(root, "0"), exist_ok=True)

    def write_field(name, foam_class, body):
        with open(os.path.join(root, "0", name), "w") as f:
            f.write("FoamFile\n{\n    format      ascii;\n")
            f.write(f"    class       {foam_class};\n")
            f.write(f"    object      {name};\n}}\n\n")
            f.write("dimensions      [0 0 0 0 0 0 0];\n\n")
            f.write(body)

    entries = "\n".join(f"{v:.10g}" for v in cell_volumes)
    write_field(
        "V",
        "volScalarField",
        "internalField   nonuniform List<scalar> \n"
        f"{len(cell_volumes)}\n(\n{entries}\n)\n;\n",
    )
    write_field(
        "alpha.liquid", "volScalarField", "internalField   uniform 1;\n"
    )
    write_field(
        "U.liquid",
        "volVectorField",
        f"internalField   uniform ({u_vector[0]} {u_vector[1]} "
        f"{u_vector[2]});\n",
    )
    write_field(
        "thermo:rho.liquid",
        "volScalarField",
        f"internalField   uniform {rho};\n",
    )


def test_compute_ave_liquid_density():
    with tempfile.TemporaryDirectory() as tmp:
        _write_liquid_fields(tmp, (1.0, 0.0, 0.0), 1050.0, [1.0, 2.0, 3.0])
        rho, _ = compute_ave_liquid_density(tmp, "0", volume_time="0")
    assert rho == pytest.approx(1050.0)


def test_compute_ave_liquid_velocity():
    # |U.liquid| = |(3,4,0)| = 5
    with tempfile.TemporaryDirectory() as tmp:
        _write_liquid_fields(tmp, (3.0, 4.0, 0.0), 1000.0, [1.0, 2.0, 3.0])
        velocity, _ = compute_ave_liquid_velocity(tmp, "0", volume_time="0")
    assert velocity == pytest.approx(5.0)


def test_interfacial_area():
    """
    Test the interfacial area a = 6 * holdup / diameter
    """
    assert interfacial_area(0.2, 0.005) == pytest.approx(240.0)


def test_instantaneous_kl():
    """
    Test for instantaneous kl calculation (volume-averaged Higbie coefficient)
    """
    case_folder = os.path.join(
        Path(__file__).parent,
        " .. ".strip(),
        " .. ".strip(),
        "bird",
        "postprocess",
        "data_conditional_mean",
    )
    kl_list, cstar_list, _ = compute_instantaneous_kl(
        species_names=["CO2"],
        case_folder=case_folder,
        time_folder="79",
        volume_time="1",
    )
    kl_str, _, _ = compute_instantaneous_kl(
        species_names="CO2",
        case_folder=case_folder,
        time_folder="79",
        volume_time="1",
    )
    # positive, finite, and the single/list species forms agree
    assert kl_list["CO2"] > 0 and np.isfinite(kl_list["CO2"])
    assert kl_str["CO2"] == pytest.approx(kl_list["CO2"])

    kla_spec, cstar_kla, _ = compute_instantaneous_kla(
        species_names=["CO2"],
        case_folder=case_folder,
        time_folder="79",
        volume_time="1",
    )
    # cstar is shared with compute_instantaneous_kla
    assert cstar_list["CO2"] == pytest.approx(cstar_kla["CO2"])
    # kL (= <coef>) and kLa (= <coef * a>) are distinct quantities
    assert kl_list["CO2"] != pytest.approx(kla_spec["CO2"])


def test_fitted_kl():
    """
    Test for fitted kl calculation (fitted kLa / interfacial area)
    """
    case_folder = os.path.join(
        Path(__file__).parent,
        " .. ".strip(),
        " .. ".strip(),
        "bird",
        "postprocess",
        "data_conditional_mean",
    )
    # dummy time folders so the fit has enough snapshots
    for time_folder in [str(entry) for entry in range(81, 89)]:
        shutil.copytree(
            os.path.join(case_folder, "80"),
            os.path.join(case_folder, time_folder),
        )
    kl_spec, _, _ = compute_fitted_kl(
        species_names=["CO2"],
        case_folder=case_folder,
        num_warmup=100,
        num_samples=100,
    )
    for time_folder in [str(entry) for entry in range(81, 89)]:
        shutil.rmtree(os.path.join(case_folder, time_folder))

    assert "mean" in kl_spec["CO2"] and "std" in kl_spec["CO2"]
    assert np.isfinite(kl_spec["CO2"]["mean"])
