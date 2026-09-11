import os


def write_preamble(output_folder):
    with open(os.path.join(output_folder, "fvModels"), "w+") as f:
        f.write("FoamFile\n")
        f.write("{\n")
        f.write("\tversion  2.0;\n")
        f.write("\tformat   ascii;\n")
        f.write("\tclass    dictionary;\n")
        f.write('\tlocation "constant";\n')
        f.write("\tobject   fvModels;\n")
        f.write("}\n\n")
        f.write("codedSource\n")
        f.write("{\n")
        f.write("\ttype\tcoded;\n")
        f.write("\tselectionMode\tall;\n")
        f.write("\tfield\tU.liquid;\n")
        f.write("\tname\tsourceTime;\n\n")
        f.write("\tcodeInclude\n")
        f.write("\t#{\n")
        f.write('\t\t#include "dynamicMix_util.H"\n')
        f.write("\t#};\n")
        f.write("\n")
        f.write("\tcodeOptions\n")
        f.write("\t#{\n")
        f.write("\t\t-I${FOAM_CASE}/constant\n")
        f.write("\t#};\n")

        f.write("\tcodeAddAlphaRhoSup\n")
        f.write("\t#{\n")
        f.write("\t\tconst Time& time = mesh().time();\n")
        f.write("\t\tconst scalarField& V = mesh().V();\n")
        f.write("\t\tvectorField& Usource = eqn.source();\n")
        f.write("\t\tconst vectorField& C = mesh().C();\n")
        f.write("\t\tconst volScalarField& rhoL =\n")
        f.write(
            '\t\t\tmesh().lookupObject<volScalarField>("thermo:rho.liquid");\n'
        )
        f.write("\t\tconst volScalarField& alphaL =\n")
        f.write('\t\t\tmesh().lookupObject<volScalarField>("alpha.liquid");\n')
        f.write("\t\tconst volVectorField& UL =\n")
        f.write('\t\t\tmesh().lookupObject<volVectorField>("U.liquid");\n')
        f.write("\t\tdouble pi=3.141592654;\n")
        f.write(f"\t\tdouble source_pt_x;\n")
        f.write(f"\t\tdouble source_pt_y;\n")
        f.write(f"\t\tdouble source_pt_z;\n")
        f.write(f"\t\tdouble disk_rad;\n")
        f.write("\t\tdouble disk_area;\n")
        f.write(f"\t\tdouble power;\n")
        f.write(f"\t\tdouble smear_factor;\n")
        f.write(f"\t\tdouble startTime;\n")


def write_mixer(mixer, output_folder):
    with open(os.path.join(output_folder, "fvModels"), "a+") as f:
        f.write(f"\t\tsource_pt_x={mixer.x};\n")
        f.write(f"\t\tsource_pt_y={mixer.y};\n")
        f.write(f"\t\tsource_pt_z={mixer.z};\n")
        f.write(f"\t\tdisk_rad={mixer.rad};\n")
        f.write("\t\tdisk_area=pi*disk_rad*disk_rad;\n")
        f.write(f"\t\tpower={mixer.power};\n")
        f.write(f"\t\tsmear_factor={float(mixer.smear)};\n")
        f.write(f"\t\tstartTime = {mixer.start_time};\n")
        f.write("\t\tif (time.value() > startTime)\n")
        f.write("\t\t{\n")
        f.write("\t\t\t// Get V1\n")
        f.write("\t\t\tdouble source_sign_factor = 1.0;\n")
        f.write("\t\t\tdouble V1 = 0;\n")
        f.write("\t\t\tdouble V2 = 0;\n")
        f.write("\t\t\tdouble rhoV;\n")
        f.write("\t\t\tdouble dist_tol = disk_rad*3;\n")
        f.write("\n")
        f.write("\t\t\tdouble dist_n;\n")
        f.write("\t\t\tdouble upV = 0;\n")
        f.write("\t\t\tdouble uprhoV = 0;\n")
        f.write("\t\t\tdouble upVvol = 0;\n")
        f.write("\t\t\tdouble downV = 0;\n")
        f.write("\t\t\tdouble downrhoV = 0;\n")
        f.write("\t\t\tdouble downVvol = 0;\n")
        f.write("\t\t\tdouble dist2;\n")

        f.write("\t\t\tforAll(C,i)\n")
        f.write("\t\t\t{\n")
        f.write(
            "\t\t\t\tdist2 = (C[i].x()-source_pt_x)*(C[i].x()-source_pt_x);\n"
        )
        f.write(
            "\t\t\t\tdist2 += (C[i].y()-source_pt_y)*(C[i].y()-source_pt_y);\n"
        )
        f.write(
            "\t\t\t\tdist2 += (C[i].z()-source_pt_z)*(C[i].z()-source_pt_z);\n"
        )
        f.write("\n")
        if mixer.normal_dir == 0:
            f.write("\t\t\t\tdist_n = (C[i].x()-source_pt_x);\n")
        elif mixer.normal_dir == 1:
            f.write("\t\t\t\tdist_n = (C[i].y()-source_pt_y);\n")
        elif mixer.normal_dir == 2:
            f.write("\t\t\t\tdist_n = (C[i].z()-source_pt_z);\n")
        f.write("\n")

        f.write(
            "\t\t\t\tif (dist2 < dist_tol*dist_tol && dist_n < -dist_tol/2) {\n"
        )
        f.write("\t\t\t\t\tupVvol += V[i] * alphaL[i];\n")
        f.write(
            f"\t\t\t\t\tupV += V[i] * alphaL[i] * UL[i][{int(mixer.normal_dir)}];\n"
        )
        f.write("\t\t\t\t\tuprhoV += V[i] * alphaL[i] * rhoL[i];\n")
        f.write("\t\t\t\t}\n")
        f.write(
            "\t\t\t\tif (dist2 < dist_tol*dist_tol && dist_n > dist_tol/2) {\n"
        )
        f.write("\t\t\t\t\tdownVvol += V[i] * alphaL[i];\n")
        f.write(
            f"\t\t\t\t\tdownV += V[i] * alphaL[i] * UL[i][{int(mixer.normal_dir)}];\n"
        )
        f.write("\t\t\t\t\tdownrhoV += V[i] * alphaL[i] * rhoL[i];\n")
        f.write("\t\t\t\t}\n")
        f.write("\t\t\t}\n")
        f.write("\n")
        f.write("\t\t\treduce(uprhoV, sumOp<scalar>());\n")
        f.write("\t\t\treduce(downrhoV, sumOp<scalar>());\n")
        f.write("\t\t\treduce(upV, sumOp<scalar>());\n")
        f.write("\t\t\treduce(downV, sumOp<scalar>());\n")
        f.write("\t\t\treduce(downVvol, sumOp<scalar>());\n")
        f.write("\t\t\treduce(upVvol, sumOp<scalar>());\n")
        f.write("\n")
        f.write("\t\t\tdownV /= downVvol;\n")
        f.write("\t\t\tupV /= upVvol;\n")
        f.write("\t\t\tdownrhoV /= downVvol;\n")
        f.write("\t\t\tuprhoV /= upVvol;\n")
        f.write("\n")
        f.write("\t\t\tif (upV <= 0 && downV <= 0) {\n")
        f.write("\t\t\t\tsource_sign_factor = -1.0;\n")
        f.write("\t\t\t\tV1 = std::abs(upV);\n")
        f.write("\t\t\t\trhoV = uprhoV;\n")
        f.write("\t\t\t} else if (upV >= 0 && downV >= 0) {\n")
        f.write("\t\t\t\tsource_sign_factor = 1.0;\n")
        f.write("\t\t\t\tV1 = std::abs(downV);\n")
        f.write("\t\t\t\trhoV = downrhoV;\n")
        f.write("\t\t\t} else {\n")
        f.write("\t\t\t\tV1 = 0.0;\n")
        if mixer.sign == "+":
            f.write("\t\t\t\tsource_sign_factor = -1.0;\n")
            f.write("\t\t\t\trhoV = uprhoV;\n")
        elif mixer.sign == "-":
            f.write("\t\t\t\tsource_sign_factor = 1.0;\n")
            f.write("\t\t\t\trhoV = downrhoV;\n")
        else:
            error_message = (
                f"mixer.sign = {mixer.sign} but should be '+' or '-'"
            )
            raise ValueError(error_message)
        f.write(
            '\t\t\t\tFoam::Info << "[BIRD:DYNMIX WARN] " << "upV = " << upV << " downV = " << downV << " for source at " << source_pt_x << ", " << source_pt_y << ", " << source_pt_z <<  endl;\n'
        )
        f.write("\t\t\t}\n")
        f.write(
            '\t\t\tFoam::Info << "[BIRD:DYNMIX INFO] V1 = " << V1 << endl;\n'
        )
        f.write("\t\t\t\n")
        f.write("\t\t\t// Get V2\n")
        f.write("\t\t\tV2 = findV2(power, rhoV, disk_area, V1);\n")
        f.write("\n")
        f.write("\t\t\tforAll(C,i)\n")
        f.write("\t\t\t{\n")
        f.write(
            "\t\t\t\tdouble Thrust=0.5*rhoL[i]*(V2*V2 - V1*V1)*disk_area;\n"
        )
        f.write(
            "\t\t\t\tdouble dist2=(C[i].x()-source_pt_x)*(C[i].x()-source_pt_x);\n"
        )
        f.write(
            "\t\t\t\tdist2 += (C[i].y()-source_pt_y)*(C[i].y()-source_pt_y);\n"
        )
        f.write(
            "\t\t\t\tdist2 += (C[i].z()-source_pt_z)*(C[i].z()-source_pt_z);\n"
        )

        f.write("\t\t\t\tdouble epsilon=pow(V[i],0.33333)*smear_factor;\n")
        f.write(
            "\t\t\t\tdouble sourceterm=alphaL[i]*(Thrust/pow(pi,1.5)/pow(epsilon,3.0))*\n"
        )
        f.write("\t\t\t\t\texp(-dist2/(epsilon*epsilon));\n")

        f.write(
            f"\t\t\t\tUsource[i][{int(mixer.normal_dir)}] -=  source_sign_factor*sourceterm*V[i];\n"
        )

        f.write("\t\t\t}\n")
        f.write("\t\t}\n")


def write_mixer_force_sign(mixer, output_folder):
    with open(os.path.join(output_folder, "fvModels"), "a+") as f:
        f.write(f"\t\tsource_pt_x={mixer.x};\n")
        f.write(f"\t\tsource_pt_y={mixer.y};\n")
        f.write(f"\t\tsource_pt_z={mixer.z};\n")
        f.write(f"\t\tdisk_rad={mixer.rad};\n")
        f.write("\t\tdisk_area=pi*disk_rad*disk_rad;\n")
        f.write(f"\t\tpower={mixer.power};\n")
        f.write(f"\t\tsmear_factor={float(mixer.smear)};\n")
        f.write(f"\t\tstartTime = {mixer.start_time};\n")
        f.write("\t\tif (time.value() > startTime)\n")
        f.write("\t\t{\n")
        f.write("\t\t\t// Get V1\n")
        f.write("\t\t\tdouble source_sign_factor = 1.0;\n")
        f.write("\t\t\tdouble V1 = 0;\n")
        f.write("\t\t\tdouble V2 = 0;\n")
        f.write("\t\t\tdouble rhoV;\n")
        f.write("\t\t\tdouble dist_tol = disk_rad*3;\n")
        f.write("\n")
        f.write("\t\t\tdouble dist_n;\n")
        f.write("\t\t\tdouble upV = 0;\n")
        f.write("\t\t\tdouble uprhoV = 0;\n")
        f.write("\t\t\tdouble upVvol = 0;\n")
        f.write("\t\t\tdouble downV = 0;\n")
        f.write("\t\t\tdouble downrhoV = 0;\n")
        f.write("\t\t\tdouble downVvol = 0;\n")
        f.write("\t\t\tdouble dist2;\n")

        f.write("\t\t\tforAll(C,i)\n")
        f.write("\t\t\t{\n")
        f.write(
            "\t\t\t\tdist2 = (C[i].x()-source_pt_x)*(C[i].x()-source_pt_x);\n"
        )
        f.write(
            "\t\t\t\tdist2 += (C[i].y()-source_pt_y)*(C[i].y()-source_pt_y);\n"
        )
        f.write(
            "\t\t\t\tdist2 += (C[i].z()-source_pt_z)*(C[i].z()-source_pt_z);\n"
        )
        f.write("\n")
        if mixer.normal_dir == 0:
            f.write("\t\t\t\tdist_n = (C[i].x()-source_pt_x);\n")
        elif mixer.normal_dir == 1:
            f.write("\t\t\t\tdist_n = (C[i].y()-source_pt_y);\n")
        elif mixer.normal_dir == 2:
            f.write("\t\t\t\tdist_n = (C[i].z()-source_pt_z);\n")
        f.write("\n")

        f.write(
            "\t\t\t\tif (dist2 < dist_tol*dist_tol && dist_n < -dist_tol/2) {\n"
        )
        f.write("\t\t\t\t\tupVvol += V[i] * alphaL[i];\n")
        f.write(
            f"\t\t\t\t\tupV += V[i] * alphaL[i] * UL[i][{int(mixer.normal_dir)}];\n"
        )
        f.write("\t\t\t\t\tuprhoV += V[i] * alphaL[i] * rhoL[i];\n")
        f.write("\t\t\t\t}\n")
        f.write(
            "\t\t\t\tif (dist2 < dist_tol*dist_tol && dist_n > dist_tol/2) {\n"
        )
        f.write("\t\t\t\t\tdownVvol += V[i] * alphaL[i];\n")
        f.write(
            f"\t\t\t\t\tdownV += V[i] * alphaL[i] * UL[i][{int(mixer.normal_dir)}];\n"
        )
        f.write("\t\t\t\t\tdownrhoV += V[i] * alphaL[i] * rhoL[i];\n")
        f.write("\t\t\t\t}\n")
        f.write("\t\t\t}\n")
        f.write("\n")
        f.write("\t\t\treduce(uprhoV, sumOp<scalar>());\n")
        f.write("\t\t\treduce(downrhoV, sumOp<scalar>());\n")
        f.write("\t\t\treduce(upV, sumOp<scalar>());\n")
        f.write("\t\t\treduce(downV, sumOp<scalar>());\n")
        f.write("\t\t\treduce(downVvol, sumOp<scalar>());\n")
        f.write("\t\t\treduce(upVvol, sumOp<scalar>());\n")
        f.write("\n")
        f.write("\t\t\tdownV /= downVvol;\n")
        f.write("\t\t\tupV /= upVvol;\n")
        f.write("\t\t\tdownrhoV /= downVvol;\n")
        f.write("\t\t\tuprhoV /= upVvol;\n")
        f.write("\n")
        if mixer.sign == "+":
            f.write("\t\t\tsource_sign_factor = -1.0;\n")
            f.write("\t\t\tif (upV >= 0){\n")
            f.write("\t\t\t\tV1 = 0.0;\n")
            f.write("\t\t\t} else {\n")
            f.write("\t\t\t\tV1 = std::abs(upV);\n")
            f.write("\t\t\t}\n")
            f.write("\t\t\trhoV = uprhoV;\n")
        elif mixer.sign == "-":
            f.write("\t\t\tsource_sign_factor = 1.0;\n")
            f.write("\t\t\tif (downV <= 0){\n")
            f.write("\t\t\t\tV1 = 0.0;\n")
            f.write("\t\t\t} else {\n")
            f.write("\t\t\t\tV1 = std::abs(downV);\n")
            f.write("\t\t\t}\n")
            f.write("\t\t\trhoV = downrhoV;\n")
        # f.write("\t\t\t}\n")
        f.write(
            '\t\t\tFoam::Info << "[BIRD:DYNMIX INFO] V1 = " << V1 << endl;\n'
        )
        f.write("\t\t\t\n")
        f.write("\t\t\t// Get V2\n")
        f.write("\t\t\tV2 = findV2(power, rhoV, disk_area, V1);\n")
        f.write("\n")
        f.write("\t\t\tforAll(C,i)\n")
        f.write("\t\t\t{\n")
        f.write(
            "\t\t\t\tdouble Thrust=0.5*rhoL[i]*(V2*V2 - V1*V1)*disk_area;\n"
        )
        f.write(
            "\t\t\t\tdouble dist2=(C[i].x()-source_pt_x)*(C[i].x()-source_pt_x);\n"
        )
        f.write(
            "\t\t\t\tdist2 += (C[i].y()-source_pt_y)*(C[i].y()-source_pt_y);\n"
        )
        f.write(
            "\t\t\t\tdist2 += (C[i].z()-source_pt_z)*(C[i].z()-source_pt_z);\n"
        )

        f.write("\t\t\t\tdouble epsilon=pow(V[i],0.33333)*smear_factor;\n")
        f.write(
            "\t\t\t\tdouble sourceterm=alphaL[i]*(Thrust/pow(pi,1.5)/pow(epsilon,3.0))*\n"
        )
        f.write("\t\t\t\t\texp(-dist2/(epsilon*epsilon));\n")

        f.write(
            f"\t\t\t\tUsource[i][{int(mixer.normal_dir)}] -=  source_sign_factor*sourceterm*V[i];\n"
        )

        f.write("\t\t\t}\n")
        f.write("\t\t}\n")


def write_preamble_ball(output_folder):
    """Write the FoamFile header + codedSource preamble for the ``ball`` source.

    The Newton solve is inlined in each mixer block (see ``write_mixer_ball``),
    so no external ``dynamicMix_util.H`` is needed.
    """
    with open(os.path.join(output_folder, "fvModels"), "w+") as f:
        f.write("FoamFile\n")
        f.write("{\n")
        f.write("\tversion  2.0;\n")
        f.write("\tformat   ascii;\n")
        f.write("\tclass    dictionary;\n")
        f.write('\tlocation "constant";\n')
        f.write("\tobject   fvModels;\n")
        f.write("}\n\n")
        f.write("codedSource\n")
        f.write("{\n")
        f.write("\ttype\tcoded;\n")
        f.write("\tselectionMode\tall;\n")
        f.write("\tfield\tU.liquid;\n")
        f.write("\tname\tsourceTime;\n\n")
        f.write("\tcodeInclude\n")
        f.write("\t#{\n")
        f.write("\t\t#include <cmath>\n")
        f.write("\t\t#include <algorithm>\n")
        f.write("\t#};\n\n")
        f.write("\tcodeAddAlphaRhoSup\n")
        f.write("\t#{\n")
        f.write("\t\tconst Time& time = mesh().time();\n")
        f.write("\t\tconst scalarField& V = mesh().V();\n")
        f.write("\t\tvectorField& Usource = eqn.source();\n")
        f.write("\t\tconst vectorField& C = mesh().C();\n")
        f.write("\t\tconst volScalarField& rhoL =\n")
        f.write(
            '\t\t\tmesh().lookupObject<volScalarField>("thermo:rho.liquid");\n'
        )
        f.write("\t\tconst volScalarField& alphaL =\n")
        f.write('\t\t\tmesh().lookupObject<volScalarField>("alpha.liquid");\n')
        f.write("\t\tconst volVectorField& UL =\n")
        f.write('\t\t\tmesh().lookupObject<volVectorField>("U.liquid");\n')
        f.write("\t\tconst double pi = 3.14159265358979;\n")


def write_mixer_ball(
    mixer, output_folder, power_mode="from_Np_Vtip", momentum_mode="axial"
):
    """Append one ``ball`` mixer block to ``fvModels``.

    Parameters
    ----------
    mixer: ActuatorMixer
        A ready :class:`~bird.preprocess.dynamic_mixer.mixer.ActuatorMixer`.
    power_mode: str
        ``"from_P"`` (drive by ``mixer.power``) or ``"from_Np_Vtip"`` (drive by
        ``mixer.Np`` and ``mixer.Vtip``).
    momentum_mode: str
        ``"axial"`` (thrust only) or ``"axial_and_swirl"`` (thrust + tangential
        source using ``mixer.sigma``).
    """
    if power_mode not in ("from_P", "from_Np_Vtip"):
        raise ValueError(f"unknown power_mode {power_mode!r}")
    if momentum_mode not in ("axial", "axial_and_swirl"):
        raise ValueError(f"unknown momentum_mode {momentum_mode!r}")
    if power_mode == "from_P" and mixer.power is None:
        raise ValueError("power_mode 'from_P' requires 'power' in the mixer")

    nd = int(mixer.normal_dir)
    dn = ["dx", "dy", "dz"][nd]
    # theta_hat = n_hat x r_hat, per axis: (component index, numerator expr)
    tan = {
        0: [(1, "-dz"), (2, "dy")],
        1: [(0, "dz"), (2, "-dx")],
        2: [(0, "-dy"), (1, "dx")],
    }[nd]
    push_ax = "1.0" if mixer.sign == "+" else "-1.0"
    push_th = "1.0" if mixer.swirl_sign == "+" else "-1.0"
    swirl = momentum_mode == "axial_and_swirl"

    if power_mode == "from_P":
        rhs = f"4.0*{mixer.power}/(rhoM*area)"
    else:
        rhs = f"16.0*{mixer.Np}*Vtip*Vtip*Vtip/pow(pi,4.0)"
    swirl_F = f" + {mixer.sigma}*(V1+V2)*Vtip*Vtip" if swirl else ""
    swirl_dF = f" + {mixer.sigma}*Vtip*Vtip" if swirl else ""

    with open(os.path.join(output_folder, "fvModels"), "a+") as f:
        f.write("\t\t// ===== ball mixer =====\n")
        f.write("\t\t{\n")
        f.write(f"\t\t\tconst double Rmix = {mixer.R};\n")
        f.write("\t\t\tconst double area = pi*Rmix*Rmix;\n")
        f.write(f"\t\t\tconst double Vtip = {mixer.Vtip};\n")
        if swirl:
            f.write(f"\t\t\tconst double sigma = {mixer.sigma};\n")
        f.write(f"\t\t\tconst double startT = {mixer.start_time};\n")
        f.write(
            f"\t\t\tconst double px = {mixer.x}, py = {mixer.y}, pz = {mixer.z};\n"
        )
        f.write("\t\t\tif (time.value() > startT)\n")
        f.write("\t\t\t{\n")
        # --- sense V1 and rho over the upstream half-ball ---
        f.write("\t\t\t\tscalar sV = 0.0, sVU = 0.0, sVrho = 0.0;\n")
        f.write("\t\t\t\tforAll(C, i)\n")
        f.write("\t\t\t\t{\n")
        f.write(
            "\t\t\t\t\tconst double dx=C[i].x()-px, dy=C[i].y()-py, dz=C[i].z()-pz;\n"
        )
        f.write("\t\t\t\t\tconst double d2 = dx*dx + dy*dy + dz*dz;\n")
        f.write(f"\t\t\t\t\tif (d2 <= Rmix*Rmix && {push_ax}*{dn} < 0.0)\n")
        f.write("\t\t\t\t\t{\n")
        f.write("\t\t\t\t\t\tconst double w = V[i]*alphaL[i];\n")
        f.write(
            f"\t\t\t\t\t\tsV += w; sVU += w*UL[i][{nd}]; sVrho += w*rhoL[i];\n"
        )
        f.write("\t\t\t\t\t}\n")
        f.write("\t\t\t\t}\n")
        f.write("\t\t\t\treduce(sV, sumOp<scalar>());\n")
        f.write("\t\t\t\treduce(sVU, sumOp<scalar>());\n")
        f.write("\t\t\t\treduce(sVrho, sumOp<scalar>());\n")
        f.write(
            f"\t\t\t\tdouble V1 = (sV>1e-30) ? {push_ax}*(sVU/sV) : 0.0;\n"
        )
        f.write("\t\t\t\tif (V1 < 0.0) V1 = 0.0;\n")
        f.write(
            "\t\t\t\tconst double rhoM = (sV>1e-30) ? sVrho/sV : 1000.0;\n"
        )
        # --- Newton solve for V2 ---
        f.write(f"\t\t\t\tconst double rhs = {rhs};\n")
        f.write(
            "\t\t\t\tdouble V2 = (V1>1e-6) ? 2.0*V1 : std::cbrt(std::abs(rhs));\n"
        )
        f.write("\t\t\t\tfor (int it = 0; it < 100; ++it)\n")
        f.write("\t\t\t\t{\n")
        f.write(
            f"\t\t\t\t\tconst double F = (V2-V1)*(V2+V1)*(V2+V1){swirl_F} - rhs;\n"
        )
        f.write(
            f"\t\t\t\t\tconst double dF = 3.0*V2*V2 + 2.0*V1*V2 - V1*V1{swirl_dF};\n"
        )
        f.write("\t\t\t\t\tconst double dV = F/dF;\n")
        f.write("\t\t\t\t\tV2 -= dV;\n")
        f.write("\t\t\t\t\tif (std::abs(dV) < 1e-10) break;\n")
        f.write("\t\t\t\t}\n")
        f.write("\t\t\t\tconst double Tax = 0.5*rhoM*area*(V2*V2 - V1*V1);\n")
        if swirl:
            f.write(
                "\t\t\t\tconst double Qsw = 0.25*rhoM*(V1+V2)*sigma*Rmix*area*Vtip;\n"
            )
        # --- pass 1: normalisation sums over the ball ---
        if swirl:
            f.write("\t\t\t\tscalar Sax = 0.0, Sth = 0.0;\n")
        else:
            f.write("\t\t\t\tscalar Sax = 0.0;\n")
        f.write("\t\t\t\tforAll(C, i)\n")
        f.write("\t\t\t\t{\n")
        f.write(
            "\t\t\t\t\tconst double dx=C[i].x()-px, dy=C[i].y()-py, dz=C[i].z()-pz;\n"
        )
        f.write("\t\t\t\t\tconst double d2 = dx*dx + dy*dy + dz*dz;\n")
        f.write("\t\t\t\t\tif (d2 <= Rmix*Rmix)\n")
        f.write("\t\t\t\t\t{\n")
        f.write(
            "\t\t\t\t\t\tconst double epsi = std::max(0.6123724356957945*Rmix, 2.0*std::cbrt(V[i]));\n"
        )
        f.write("\t\t\t\t\t\tconst double g = std::exp(-d2/(epsi*epsi));\n")
        f.write("\t\t\t\t\t\tSax += alphaL[i]*g*V[i];\n")
        if swirl:
            f.write(
                f"\t\t\t\t\t\tconst double rr = std::sqrt(d2-({dn})*({dn}));\n"
            )
            f.write("\t\t\t\t\t\tSth += alphaL[i]*g*rr*V[i];\n")
        f.write("\t\t\t\t\t}\n")
        f.write("\t\t\t\t}\n")
        f.write("\t\t\t\treduce(Sax, sumOp<scalar>());\n")
        if swirl:
            f.write("\t\t\t\treduce(Sth, sumOp<scalar>());\n")
        # --- pass 2: apply ---
        f.write("\t\t\t\tforAll(C, i)\n")
        f.write("\t\t\t\t{\n")
        f.write(
            "\t\t\t\t\tconst double dx=C[i].x()-px, dy=C[i].y()-py, dz=C[i].z()-pz;\n"
        )
        f.write("\t\t\t\t\tconst double d2 = dx*dx + dy*dy + dz*dz;\n")
        f.write("\t\t\t\t\tif (d2 <= Rmix*Rmix)\n")
        f.write("\t\t\t\t\t{\n")
        f.write(
            "\t\t\t\t\t\tconst double epsi = std::max(0.6123724356957945*Rmix, 2.0*std::cbrt(V[i]));\n"
        )
        f.write("\t\t\t\t\t\tconst double g = std::exp(-d2/(epsi*epsi));\n")
        f.write("\t\t\t\t\t\tif (Sax > 1e-30)\n")
        f.write("\t\t\t\t\t\t{\n")
        f.write("\t\t\t\t\t\t\tconst double fax = Tax/Sax*alphaL[i]*g;\n")
        f.write(f"\t\t\t\t\t\t\tUsource[i][{nd}] -= {push_ax}*fax*V[i];\n")
        f.write("\t\t\t\t\t\t}\n")
        if swirl:
            f.write(
                f"\t\t\t\t\t\tconst double rr = std::sqrt(d2-({dn})*({dn}));\n"
            )
            f.write("\t\t\t\t\t\tif (rr > 1e-3*Rmix && Sth > 1e-30)\n")
            f.write("\t\t\t\t\t\t{\n")
            f.write("\t\t\t\t\t\t\tconst double fth = Qsw/Sth*alphaL[i]*g;\n")
            f.write(
                f"\t\t\t\t\t\t\tUsource[i][{tan[0][0]}] -= {push_th}*fth*V[i]*(({tan[0][1]})/rr);\n"
            )
            f.write(
                f"\t\t\t\t\t\t\tUsource[i][{tan[1][0]}] -= {push_th}*fth*V[i]*(({tan[1][1]})/rr);\n"
            )
            f.write("\t\t\t\t\t\t}\n")
        f.write("\t\t\t\t\t}\n")
        f.write("\t\t\t\t}\n")
        f.write("\t\t\t}\n")
        f.write("\t\t}\n")


def write_static_mixer_ball(mixer, output_folder):
    """Append one passive ``ball`` static-mixer block to ``fvModels``.

    The swirl is imposed as an azimuthal body force proportional to the local
    axial dynamic pressure (Kiesewetter), balanced cell-by-cell by an
    energy-neutral axial reaction; a lumped axial drag adds the viscous loss.
    The source is inactive when the inflow opposes the mixer orientation.

    Parameters
    ----------
    mixer: StaticMixer
        A ready :class:`~bird.preprocess.dynamic_mixer.mixer.StaticMixer`.
    """
    nd = int(mixer.normal_dir)
    dn = ["dx", "dy", "dz"][nd]
    # theta_hat = n_hat x r_hat, per axis: (component index, numerator expr)
    tan = {
        0: [(1, "-dz"), (2, "dy")],
        1: [(0, "dz"), (2, "-dx")],
        2: [(0, "-dy"), (1, "dx")],
    }[nd]
    push_ax = "1.0" if mixer.sign == "+" else "-1.0"
    push_th = "1.0" if mixer.swirl_sign == "+" else "-1.0"
    # pre-combined signs (= -push_ax / -push_th) as single literals, so the
    # emitted C++ never contains "--1.0" (a decrement on a literal, a compile
    # error). The drag opposes the oriented inflow; the reaction is -push_th.
    drag_ax = "-1.0" if mixer.sign == "+" else "1.0"
    cp_th = "-1.0" if mixer.swirl_sign == "+" else "1.0"

    with open(os.path.join(output_folder, "fvModels"), "a+") as f:
        f.write("\t\t// ===== static mixer =====\n")
        f.write("\t\t{\n")
        f.write(f"\t\t\tconst double Rmix = {mixer.R};\n")
        f.write("\t\t\tconst double area = pi*Rmix*Rmix;\n")
        f.write(f"\t\t\tconst double Snum = {mixer.S};\n")
        f.write(f"\t\t\tconst double Kloss = {mixer.K};\n")
        f.write(f"\t\t\tconst double startT = {mixer.start_time};\n")
        f.write(
            f"\t\t\tconst double px = {mixer.x}, py = {mixer.y}, pz = {mixer.z};\n"
        )
        f.write("\t\t\tif (time.value() > startT)\n")
        f.write("\t\t\t{\n")
        # --- sense V1 and rho over the upstream half-ball ---
        # (duplicated from write_mixer_ball; the dynamic path is kept untouched)
        f.write("\t\t\t\tscalar sV = 0.0, sVU = 0.0, sVrho = 0.0;\n")
        f.write("\t\t\t\tforAll(C, i)\n")
        f.write("\t\t\t\t{\n")
        f.write(
            "\t\t\t\t\tconst double dx=C[i].x()-px, dy=C[i].y()-py, dz=C[i].z()-pz;\n"
        )
        f.write("\t\t\t\t\tconst double d2 = dx*dx + dy*dy + dz*dz;\n")
        f.write(f"\t\t\t\t\tif (d2 <= Rmix*Rmix && {push_ax}*{dn} < 0.0)\n")
        f.write("\t\t\t\t\t{\n")
        f.write("\t\t\t\t\t\tconst double w = V[i]*alphaL[i];\n")
        f.write(
            f"\t\t\t\t\t\tsV += w; sVU += w*UL[i][{nd}]; sVrho += w*rhoL[i];\n"
        )
        f.write("\t\t\t\t\t}\n")
        f.write("\t\t\t\t}\n")
        f.write("\t\t\t\treduce(sV, sumOp<scalar>());\n")
        f.write("\t\t\t\treduce(sVU, sumOp<scalar>());\n")
        f.write("\t\t\t\treduce(sVrho, sumOp<scalar>());\n")
        f.write(
            f"\t\t\t\tdouble V1 = (sV>1e-30) ? {push_ax}*(sVU/sV) : 0.0;\n"
        )
        f.write("\t\t\t\tif (V1 < 0.0) V1 = 0.0;\n")
        f.write(
            "\t\t\t\tconst double rhoM = (sV>1e-30) ? sVrho/sV : 1000.0;\n"
        )
        # --- passive loads (no Newton solve) ---
        f.write("\t\t\t\tconst double Qsw = Snum*Rmix*rhoM*area*V1*V1;\n")
        f.write("\t\t\t\tconst double Tls = 0.5*Kloss*rhoM*area*V1*V1;\n")
        # --- pass 1: normalisation sums over the ball ---
        f.write("\t\t\t\tscalar Sax = 0.0, Ssw = 0.0;\n")
        f.write("\t\t\t\tforAll(C, i)\n")
        f.write("\t\t\t\t{\n")
        f.write(
            "\t\t\t\t\tconst double dx=C[i].x()-px, dy=C[i].y()-py, dz=C[i].z()-pz;\n"
        )
        f.write("\t\t\t\t\tconst double d2 = dx*dx + dy*dy + dz*dz;\n")
        f.write("\t\t\t\t\tif (d2 <= Rmix*Rmix)\n")
        f.write("\t\t\t\t\t{\n")
        f.write(
            "\t\t\t\t\t\tconst double epsi = std::max(0.6123724356957945*Rmix, 2.0*std::cbrt(V[i]));\n"
        )
        f.write("\t\t\t\t\t\tconst double g = std::exp(-d2/(epsi*epsi));\n")
        f.write(
            f"\t\t\t\t\t\tconst double rr = std::sqrt(d2-({dn})*({dn}));\n"
        )
        f.write(f"\t\t\t\t\t\tconst double ux = UL[i][{nd}];\n")
        f.write("\t\t\t\t\t\tSax += alphaL[i]*g*V[i];\n")
        f.write("\t\t\t\t\t\tSsw += alphaL[i]*g*rhoL[i]*ux*ux*rr*V[i];\n")
        f.write("\t\t\t\t\t}\n")
        f.write("\t\t\t\t}\n")
        f.write("\t\t\t\treduce(Sax, sumOp<scalar>());\n")
        f.write("\t\t\t\treduce(Ssw, sumOp<scalar>());\n")
        # --- pass 2: apply ---
        f.write("\t\t\t\tforAll(C, i)\n")
        f.write("\t\t\t\t{\n")
        f.write(
            "\t\t\t\t\tconst double dx=C[i].x()-px, dy=C[i].y()-py, dz=C[i].z()-pz;\n"
        )
        f.write("\t\t\t\t\tconst double d2 = dx*dx + dy*dy + dz*dz;\n")
        f.write("\t\t\t\t\tif (d2 <= Rmix*Rmix)\n")
        f.write("\t\t\t\t\t{\n")
        f.write(
            "\t\t\t\t\t\tconst double epsi = std::max(0.6123724356957945*Rmix, 2.0*std::cbrt(V[i]));\n"
        )
        f.write("\t\t\t\t\t\tconst double g = std::exp(-d2/(epsi*epsi));\n")
        # viscous drag (lumped): opposes the oriented inflow
        f.write("\t\t\t\t\t\tif (Sax > 1e-30)\n")
        f.write("\t\t\t\t\t\t{\n")
        f.write("\t\t\t\t\t\t\tconst double fvisc = Tls/Sax*alphaL[i]*g;\n")
        f.write(f"\t\t\t\t\t\t\tUsource[i][{nd}] -= {drag_ax}*fvisc*V[i];\n")
        f.write("\t\t\t\t\t\t}\n")
        # swirl + energy-neutral axial reaction (local, velocity-weighted)
        f.write(
            f"\t\t\t\t\t\tconst double rr = std::sqrt(d2-({dn})*({dn}));\n"
        )
        f.write("\t\t\t\t\t\tif (rr > 1e-3*Rmix && Ssw > 1e-30)\n")
        f.write("\t\t\t\t\t\t{\n")
        f.write(f"\t\t\t\t\t\t\tconst double ux = UL[i][{nd}];\n")
        f.write(
            f"\t\t\t\t\t\t\tconst double uth = UL[i][{tan[0][0]}]*(({tan[0][1]})/rr) + UL[i][{tan[1][0]}]*(({tan[1][1]})/rr);\n"
        )
        f.write("\t\t\t\t\t\t\tconst double A0 = Qsw/Ssw;\n")
        f.write(
            "\t\t\t\t\t\t\tconst double fsw = A0*rhoL[i]*ux*ux*alphaL[i]*g;\n"
        )
        f.write(
            f"\t\t\t\t\t\t\tUsource[i][{tan[0][0]}] -= {push_th}*fsw*V[i]*(({tan[0][1]})/rr);\n"
        )
        f.write(
            f"\t\t\t\t\t\t\tUsource[i][{tan[1][0]}] -= {push_th}*fsw*V[i]*(({tan[1][1]})/rr);\n"
        )
        f.write(
            "\t\t\t\t\t\t\tconst double fcp = A0*rhoL[i]*ux*uth*alphaL[i]*g;\n"
        )
        f.write(f"\t\t\t\t\t\t\tUsource[i][{nd}] -= {cp_th}*fcp*V[i];\n")
        f.write("\t\t\t\t\t\t}\n")
        f.write("\t\t\t\t\t}\n")
        f.write("\t\t\t\t}\n")
        f.write("\t\t\t}\n")
        f.write("\t\t}\n")


def write_end(output_folder):
    with open(os.path.join(output_folder, "fvModels"), "a+") as f:
        f.write("\t#};\n")
        f.write("};\n")
