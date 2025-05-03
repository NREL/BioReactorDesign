import os
import re
from collections import defaultdict

from ruamel.yaml import YAML

from bird import BIRD_CONST_DIR
from bird.utilities.ofio import read_properties


def parse_yaml(filename: str):
    yaml = YAML()
    with open(filename, "r+") as f:
        spec = yaml.load(f)

    return spec


def global_vars_to_dict(filename: str):
    with open(filename, "r+") as f:
        lines = f.readlines()
    global_vars_dict = {}
    for line in lines:
        if not line.startswith("//") and not line.startswith("\n"):
            line_l = line.split()
            name = line_l[0]
            val = line[len(name) + 1 :]
            ind_end = val.index(";")
            val = val[:ind_end]
            global_vars_dict[name] = val

    return global_vars_dict


def add_liquid_properties(case_dir, global_vars_dict):
    print("WARNING: assuming liquid phase is water")
    h2o_prop = parse_yaml(os.path.join(BIRD_CONST_DIR, "h2o.yaml"))
    CpMixLiq = h2o_prop["liquid"]["CpMixLiq"]
    muMixLiq = h2o_prop["liquid"]["muMixLiq"]
    kThermLiq = h2o_prop["liquid"]["kThermLiq"]
    rho0MixLiq = h2o_prop["liquid"]["rho0MixLiq"]
    sigmaLiq = h2o_prop["liquid"]["sigmaLiq"]
    WC_psi = h2o_prop["liquid"]["WC_psi"]
    WC_M = h2o_prop["specie"]["molWeight"]

    if "CpMixLiq" not in global_vars_dict:
        global_vars_dict["CpMixLiq"] = CpMixLiq
    else:
        print(
            f"WARNING: CpMixLiq stays at {global_vars_dict['CpMixLiq']} instead of {CpMixLiq}"
        )
    if "muMixLiq" not in global_vars_dict:
        global_vars_dict["muMixLiq"] = muMixLiq
    else:
        print(
            f"WARNING: muMixLiq stays at {global_vars_dict['muMixLiq']} instead of {muMixLiq}"
        )
    if "kThermLiq" not in global_vars_dict:
        global_vars_dict["kThermLiq"] = kThermLiq
    else:
        print(
            f"WARNING: kThermLiq stays at {global_vars_dict['kThermLiq']} instead of {kThermLiq}"
        )
    if "rho0MixLiq" not in global_vars_dict:
        global_vars_dict["rho0MixLiq"] = rho0MixLiq
    else:
        print(
            f"WARNING: rho0MixLiq stays at {global_vars_dict['rho0MixLiq']} instead of {rho0MixLiq}"
        )
    if "sigmaLiq" not in global_vars_dict:
        global_vars_dict["sigmaLiq"] = sigmaLiq
    else:
        print(
            f"WARNING: sigmaLiq stays at {global_vars_dict['sigmaLiq']} instead of {sigmaLiq}"
        )
    if "WC_psi" not in global_vars_dict:
        global_vars_dict["WC_psi"] = WC_psi
    else:
        print(
            f"WARNING: WC_psi stays at {global_vars_dict['WC_psi']} instead of {WC_psi}"
        )
    if "WC_M" not in global_vars_dict:
        global_vars_dict["WC_M"] = WC_M
    else:
        print(
            f"WARNING: WC_M stays at {global_vars_dict['WC_M']} instead of {WC_M}"
        )

    return global_vars_dict


def add_gas_properties(case_dir, global_vars_dict):
    gas_properties = read_properties(
        os.path.join(case_dir, "constant", "thermophysicalProperties.gas")
    )
    list_gas_spec = list(
        set(gas_properties["species"] + [gas_properties["defaultSpecie"]])
    )
    try:
        list_gas_spec.remove("water")
    except ValueError:
        pass
    try:
        list_gas_spec.remove("H2O")
    except ValueError:
        pass

    print(f"INFO: gas species = {list_gas_spec}")
    for spec in list_gas_spec:
        if spec == "water":
            spec_prop = parse_yaml(os.path.join(BIRD_CONST_DIR, "h2o.yaml"))
        else:
            spec_prop = parse_yaml(
                os.path.join(BIRD_CONST_DIR, f"{spec.lower()}.yaml")
            )
        name = spec_prop["name"]
        WC_V = spec_prop["gas"]["in-H2O"]["WC_V"]
        D = spec_prop["gas"]["in-H2O"]["D"]
        H_298 = spec_prop["gas"]["H_298"]
        DH = spec_prop["gas"]["DH"]
        He = spec_prop["gas"]["He"]
        LeLiq = spec_prop["liquid"]["LeLiq"]
        k = spec_prop["gas"]["k"]
        Pr = spec_prop["gas"]["Pr"]
        if not f"WC_V_{name}" in global_vars_dict:
            global_vars_dict[f"WC_V_{name}"] = WC_V
        else:
            print(
                f"WARNING: WC_V_{name} stays at {global_vars_dict[f'WC_V_{name}']} instead of {WC_V}"
            )
        if not f"D_{name}" in global_vars_dict:
            global_vars_dict[f"D_{name}"] = D
        else:
            print(
                f"WARNING: D_{name} stays at {global_vars_dict[f'D_{name}']} instead of {D}"
            )
        if not f"H_{name}_298" in global_vars_dict:
            global_vars_dict[f"H_{name}_298"] = H_298
        else:
            print(
                f"WARNING: H_{name}_298 stays at {global_vars_dict[f'H_{name}_298']} instead of {H_298}"
            )
        if not f"DH_{name}" in global_vars_dict:
            global_vars_dict[f"DH_{name}"] = DH
        else:
            print(
                f"WARNING: DH_{name} stays at {global_vars_dict[f'DH_{name}']} instead of {DH}"
            )
        if not f"He_{name}" in global_vars_dict:
            global_vars_dict[f"He_{name}"] = He
        else:
            print(
                f"WARNING: He_{name} stays at {global_vars_dict[f'He_{name}']} instead of {He}"
            )
        if not f"LeLiq{name}" in global_vars_dict:
            global_vars_dict[f"LeLiq{name}"] = LeLiq
        else:
            print(
                f"WARNING: LeLiq{name} stays at {global_vars_dict[f'LeLiq{name}']} instead of {LeLiq}"
            )
        if not f"k{name}" in global_vars_dict:
            global_vars_dict[f"k{name}"] = k
        else:
            print(
                f"WARNING: k{name} stays at {global_vars_dict[f'k{name}']} instead of {k}"
            )
        if not f"Pr{name}" in global_vars_dict:
            global_vars_dict[f"Pr{name}"] = Pr
        else:
            print(
                f"WARNING: Pr{name} stays at {global_vars_dict[f'Pr{name}']} instead of {Pr}"
            )

    return global_vars_dict, list_gas_spec


def check_globalVars(case_dir, global_vars_dict, list_gas_spec):
    globalVar_filename = os.path.join(case_dir, "constant", "globalVars")
    list_supported_key = ["T0", "VVM"]
    list_supported_key += [
        "CpMixLiq",
        "muMixLiq",
        "kThermLiq",
        "rho0MixLiq",
        "sigmaLiq",
    ]
    list_supported_key += ["WC_psi", "WC_M"]
    for spec in list_gas_spec:
        list_supported_key += [f"WC_V_{spec}"]
    for spec in list_gas_spec:
        list_supported_key += [f"D_{spec}"]
    for spec in list_gas_spec:
        list_supported_key += [f"H_{spec}_298"]
    for spec in list_gas_spec:
        list_supported_key += [f"DH_{spec}"]
    for spec in list_gas_spec:
        list_supported_key += [f"He_{spec}"]
    for spec in list_gas_spec:
        list_supported_key += [f"f_{spec}"]
        list_supported_key += [f"x_{spec}"]
    list_supported_key += ["f_H2O", "f_water"]
    list_supported_key += [
        "inletA",
        "liqVol",
        "alphaGas",
        "alphaLiq",
        "uGasPhase",
        "P0",
        "Pbot",
        "Pmid",
        "ArbyAs",
        "uSupVel",
        "uGas",
        "rho_gas",
    ]
    list_supported_key += [
        "uGasPhase_Sup",
        "uLiqPhase_Sup",
        "uLiqPhase",
        "mflowRateLiq",
        "mflowRateGas",
        "mflowRate",
    ]
    for spec in list_gas_spec:
        list_supported_key += [f"LeLiq{spec}"]
    list_supported_key += ["PrMixLiq", "LeLiqMix"]
    for spec in list_gas_spec:
        list_supported_key += [f"k{spec}", f"Pr{spec}"]
    list_supported_key += [
        "l_scale",
        "intensity",
        "k_inlet_gas",
        "k_inlet_liq",
        "eps_inlet_gas",
        "eps_inlet_liq",
        "omega_inlet_gas",
        "omega_inlet_liq",
    ]
    list_supported_key += [
        "HtBcol",
        "DiaBcol",
        "LiqHt",
        "NPS",
        "NPD",
        "NPY",
        "A_Bcol",
        "V_flowRate",
    ]

    keys_unknown = []
    for key in global_vars_dict:
        if key not in list_supported_key:
            print(
                f"WARNING: no specific treatment for key {key}, will be written at the beginning"
            )
            keys_unknown.append(key)
    return keys_unknown


def write_macro_prop(case_dir, global_vars_dict, list_gas_spec):
    globalVar_filename = os.path.join(case_dir, "constant", "globalVars")
    try:
        os.remove(globalVar_filename)
    except FileNotFoundError:
        pass

    with open(globalVar_filename, "w+") as f:
        if "T0" in global_vars_dict:
            f.write(
                f"T0      {global_vars_dict['T0']};//initial T(K) which stays constant\n"
            )
            print(f"WARNING: Assuming 300K for T0")
            f.write(f"T0      300;//initial T(K) which stays constant\n")
        if "VVM" in global_vars_dict:
            f.write(f"VVM      {global_vars_dict['VVM']};\n")


def write_liq_prop(case_dir, global_vars_dict, list_gas_spec):
    globalVar_filename = os.path.join(case_dir, "constant", "globalVars")
    with open(globalVar_filename, "a+") as f:
        f.write("//****water Liquid properties**************\n")
        if "CpMixLiq" in global_vars_dict:
            f.write(f"CpMixLiq      {global_vars_dict['CpMixLiq']};\n")
        if "muMixLiq" in global_vars_dict:
            f.write(
                f"muMixLiq      {global_vars_dict['muMixLiq']};//viscosity (Pa.s) of water as a function of T(K) \n"
            )
        if "kThermLiq" in global_vars_dict:
            f.write(
                f"kThermLiq     {global_vars_dict['kThermLiq']};// W/m-K\n"
            )
        if "rho0MixLiq" in global_vars_dict:
            f.write(
                f"rho0MixLiq    {global_vars_dict['rho0MixLiq']};// kg/m^3\n"
            )
        if "sigmaLiq" in global_vars_dict:
            f.write(
                f"sigmaLiq      {global_vars_dict['sigmaLiq']};//surface tension N/m\n"
            )


def write_wc_prop(case_dir, global_vars_dict, list_gas_spec):
    globalVar_filename = os.path.join(case_dir, "constant", "globalVars")
    with open(globalVar_filename, "a+") as f:
        f.write(
            "//****Wilke-Chang params for diffusion coefficient of a given solute in water (solvent)\n"
        )
        if "WC_psi" in global_vars_dict:
            f.write(f"WC_psi      {global_vars_dict['WC_psi']};\n")
        if "WC_M" in global_vars_dict:
            f.write(f"WC_M      {global_vars_dict['WC_M']};// kg/kmol\n")
        for spec in list_gas_spec:
            key = f"WC_V_{spec}"
            if key in global_vars_dict:
                f.write(f"{key}     {global_vars_dict[key]};// m3/kmol\n")


def write_diff_prop(case_dir, global_vars_dict, list_gas_spec):
    globalVar_filename = os.path.join(case_dir, "constant", "globalVars")
    with open(globalVar_filename, "a+") as f:
        f.write("//****** diffusion coeff ***********\n")
        for spec in list_gas_spec:
            key = f"D_{spec}"
            if key in global_vars_dict:
                f.write(f"{key}     {global_vars_dict[key]};// m3/kmol\n")


def write_henry_prop(case_dir, global_vars_dict, list_gas_spec):
    globalVar_filename = os.path.join(case_dir, "constant", "globalVars")
    with open(globalVar_filename, "a+") as f:
        f.write("//****** Henry coeff ***********\n")
        for spec in list_gas_spec:
            key = f"H_{spec}_298"
            if key in global_vars_dict:
                f.write(f"{key}     {global_vars_dict[key]};\n")
            key = f"DH_{spec}"
            if key in global_vars_dict:
                f.write(f"{key}     {global_vars_dict[key]};\n")
        for spec in list_gas_spec:
            key = f"He_{spec}"
            if key in global_vars_dict:
                f.write(f"{key}     {global_vars_dict[key]};\n")


def write_inlet_prop(case_dir, global_vars_dict, list_gas_spec):
    globalVar_filename = os.path.join(case_dir, "constant", "globalVars")
    with open(globalVar_filename, "a+") as f:
        f.write("//****** inlet gas frac ***********\n")
        for spec in list_gas_spec:
            key = f"x_{spec}"
            if key in global_vars_dict:
                f.write(f"{key}     {global_vars_dict[key]};\n")
        for spec in list_gas_spec:
            key = f"f_{spec}"
            if key in global_vars_dict:
                f.write(f"{key}     {global_vars_dict[key]};\n")
        key = f"f_H2O"
        if key in global_vars_dict:
            f.write(f"{key}     {global_vars_dict[key]};\n")
        key = f"f_water"
        if key in global_vars_dict:
            f.write(f"{key}     {global_vars_dict[key]};\n")
        key_list = [
            "uGasPhase_Sup",
            "uLiqPhase_Sup",
            "alphaGas",
            "alphaLiq",
            "uGasPhase",
            "uLiqPhase",
            "mflowRateLiq",
            "mflowRateGas",
            "mflowRate",
        ]
        key_list += ["inletA", "liqVol", "ArbyAs", "uSupVel", "uGas"]
        for key in key_list:
            if key in global_vars_dict:
                val = global_vars_dict[key]
                f.write(f"{key}     {val};\n")


def write_cond_prop(case_dir, global_vars_dict, list_gas_spec):
    globalVar_filename = os.path.join(case_dir, "constant", "globalVars")
    with open(globalVar_filename, "a+") as f:
        f.write("//*****************\n")
        for ispec, spec in enumerate(list_gas_spec):
            key = f"k{spec}"
            if key in global_vars_dict:
                f.write(f"{key}     {global_vars_dict[key]};\n")
            key = f"Pr{spec}"
            if key in global_vars_dict:
                f.write(f"{key}     {global_vars_dict[key]};\n")
            if ispec < len(list_gas_spec) - 1:
                f.write("\n")


def write_Le_prop(case_dir, global_vars_dict, list_gas_spec):
    globalVar_filename = os.path.join(case_dir, "constant", "globalVars")
    with open(globalVar_filename, "a+") as f:
        f.write("//*****************\n")
        for ispec, spec in enumerate(list_gas_spec):
            key = f"LeLiq{spec}"
            if key in global_vars_dict:
                f.write(f"{key}     {global_vars_dict[key]};\n")
        if "LeLiqMix" in global_vars_dict:
            f.write(f"LeLiqMix     {global_vars_dict['LeLiqMix']};\n")
        else:
            LeLiqMix_val = "#calc "
            for ispec, spec in enumerate(list_gas_spec):
                if ispec == 0:
                    LeLiqMix_val += f'"$f_{spec}*$LeLiq{spec}'
                else:
                    LeLiqMix_val += f"+$f_{spec}*$LeLiq{spec}"
                if ispec == len(list_gas_spec) - 1:
                    LeLiqMix_val += '"'
            f.write(f"LeLiqMix     {LeLiqMix_val};\n")
        if "PrMixLiq" in global_vars_dict:
            f.write(f"PrMixLiq     {global_vars_dict['PrMixLiq']};\n")
        else:
            f.write(
                f'PrMixLiq     #calc "$CpMixLiq * $muMixLiq / $kThermLiq";\n'
            )


def write_turb_prop(case_dir, global_vars_dict, list_gas_spec):
    globalVar_filename = os.path.join(case_dir, "constant", "globalVars")
    with open(globalVar_filename, "a+") as f:
        f.write("//*****************\n")
        if "l_scale" in global_vars_dict:
            f.write(f"l_scale     {global_vars_dict['l_scale']};\n")
        if "intensity" in global_vars_dict:
            f.write(f"intensity   {global_vars_dict['intensity']};\n")
        if "k_inlet_gas" in global_vars_dict:
            f.write(f"k_inlet_gas {global_vars_dict['k_inlet_gas']};\n")
        else:
            if (
                "uGasPhase" in global_vars_dict
                and "intensity" in global_vars_dict
            ):
                f.write(
                    f'k_inlet_gas #calc "1.5 * Foam::pow(($uGasPhase), 2) * Foam::pow($intensity, 2)";\n'
                )
                global_vars_dict["k_inlet_gas"] = (
                    '#calc "1.5 * Foam::pow(($uGasPhase), 2) * Foam::pow($intensity, 2)"'
                )
            else:
                print("WARNING: no k_inlet_gas could be printed")
        if "k_inlet_liq" in global_vars_dict:
            f.write(f"k_inlet_liq   {global_vars_dict['k_inlet_liq']};\n")
        else:
            if (
                "uGasPhase" in global_vars_dict
                and "intensity" in global_vars_dict
            ):
                f.write(
                    f'k_inlet_liq #calc "1.5 * Foam::pow(($uGasPhase), 2) * Foam::pow($intensity, 2)";\n'
                )
                global_vars_dict["k_inlet_liq"] = (
                    '#calc "1.5 * Foam::pow(($uGasPhase), 2) * Foam::pow($intensity, 2)"'
                )
            else:
                print("WARNING: no k_inlet_liq could be printed")
        if "eps_inlet_gas" in global_vars_dict:
            f.write(f"eps_inlet_gas {global_vars_dict['eps_inlet_gas']};\n")
        else:
            if (
                "k_inlet_gas" in global_vars_dict
                and "l_scale" in global_vars_dict
            ):
                f.write(
                    f'eps_inlet_gas #calc "pow(0.09,0.75) * Foam::pow($k_inlet_gas, 1.5) / ($l_scale * 0.07)";\n'
                )
                global_vars_dict[f"eps_inlet_gas"] = (
                    '#calc "pow(0.09,0.75) * Foam::pow($k_inlet_gas, 1.5) / ($l_scale * 0.07)"'
                )
            else:
                print("WARNING: no eps_inlet_gas could be printed")
        if "eps_inlet_liq" in global_vars_dict:
            f.write(f"eps_inlet_liq   {global_vars_dict['eps_inlet_liq']};\n")
        else:
            if (
                "k_inlet_liq" in global_vars_dict
                and "l_scale" in global_vars_dict
            ):
                f.write(
                    f'eps_inlet_liq #calc "pow(0.09,0.75) * Foam::pow($k_inlet_liq, 1.5) / ($l_scale * 0.07)";\n'
                )
                global_vars_dict[f"eps_inlet_liq"] = (
                    '#calc "pow(0.09,0.75) * Foam::pow($k_inlet_liq, 1.5) / ($l_scale * 0.07)"'
                )
            else:
                print("WARNING: no eps_inlet_liq could be printed")
        if "omega_inlet_gas" in global_vars_dict:
            f.write(
                f"omega_inlet_gas {global_vars_dict['omega_inlet_gas']};\n"
            )
        else:
            if (
                "k_inlet_gas" in global_vars_dict
                and "l_scale" in global_vars_dict
            ):
                f.write(
                    f'omega_inlet_gas #calc "pow(0.09,-0.25) * pow($k_inlet_gas,0.5) / ($l_scale * 0.07)";\n'
                )
                global_vars_dict[f"omega_inlet_gas"] = (
                    '#calc "pow(0.09,-0.25) * pow($k_inlet_gas,0.5) / ($l_scale * 0.07)"'
                )
            else:
                print("WARNING: no omega_inlet_gas could be printed")

        if "omega_inlet_liq" in global_vars_dict:
            f.write(
                f"omega_inlet_liq   {global_vars_dict['omega_inlet_liq']};\n"
            )
        else:
            if (
                "k_inlet_liq" in global_vars_dict
                and "l_scale" in global_vars_dict
            ):
                f.write(
                    f'omega_inlet_liq #calc "pow(0.09,-0.25) * pow($k_inlet_liq,0.5) / ($l_scale * 0.07)";\n'
                )
                global_vars_dict[f"omega_inlet_liq"] = (
                    '#calc "pow(0.09,-0.25) * pow($k_inlet_liq,0.5) / ($l_scale * 0.07)"'
                )
            else:
                print("WARNING: no omega_inlet_liq could be printed")


def write_unknown_prop(
    case_dir, global_vars_dict, list_gas_spec, unknown_keys
):
    globalVar_filename = os.path.join(case_dir, "constant", "globalVars")
    with open(globalVar_filename, "a+") as f:
        f.write("//*****************\n")
        for key in unknown_keys:
            if key in global_vars_dict:
                val = global_vars_dict[key]
                f.write(f"{key}     {val};\n")


def write_geom_prop(case_dir, global_vars_dict, list_gas_spec):
    globalVar_filename = os.path.join(case_dir, "constant", "globalVars")
    with open(globalVar_filename, "a+") as f:
        f.write("//*****************\n")
        for key in [
            "HtBcol",
            "DiaBcol",
            "LiqHt",
            "NPS",
            "NPD",
            "NPY",
            "P0",
            "Pbot",
            "Pmid",
            "A_Bcol",
            "V_flowRate",
            "rho_gas",
        ]:
            if key in global_vars_dict:
                val = global_vars_dict[key]
                f.write(f"{key}     {val};\n")


def write_globalVars(case_dir, global_vars_dict, list_gas_spec):
    unknown_keys = check_globalVars(case_dir, global_vars_dict, list_gas_spec)
    write_unknown_prop(case_dir, global_vars_dict, list_gas_spec, unknown_keys)
    write_macro_prop(case_dir, global_vars_dict, list_gas_spec)
    write_liq_prop(case_dir, global_vars_dict, list_gas_spec)
    write_wc_prop(case_dir, global_vars_dict, list_gas_spec)
    write_diff_prop(case_dir, global_vars_dict, list_gas_spec)
    write_henry_prop(case_dir, global_vars_dict, list_gas_spec)
    write_geom_prop(case_dir, global_vars_dict, list_gas_spec)
    write_inlet_prop(case_dir, global_vars_dict, list_gas_spec)
    write_Le_prop(case_dir, global_vars_dict, list_gas_spec)
    write_cond_prop(case_dir, global_vars_dict, list_gas_spec)
    write_turb_prop(case_dir, global_vars_dict, list_gas_spec)


def fill_global_prop(case_dir: str):
    global_vars_dict = global_vars_to_dict(
        os.path.join(case_dir, "constant", "globalVars_temp")
    )
    global_vars_dict = add_liquid_properties(case_dir, global_vars_dict)
    global_vars_dict, list_gas_spec = add_gas_properties(
        case_dir, global_vars_dict
    )
    write_globalVars(case_dir, global_vars_dict, list_gas_spec)


if __name__ == "__main__":
    # fill_global_prop("../../../experimental_cases_new/disengagement/bubble_column_pbe_20L/")
    fill_global_prop("../../../experimental_cases_new/deckwer17")
