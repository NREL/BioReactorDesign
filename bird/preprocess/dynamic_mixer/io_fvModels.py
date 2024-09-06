import os


def write_preamble(output_folder):
    with open(os.path.join(output_folder, "fvModels"), "w+") as f:
        f.write("FoamFile\n")
        f.write("{\n")
        f.write("\tversion  9.0;\n")
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
            sys.exit(
                f"ERROR: mixer.sign = {mixer.sign} but should be '+' or '-'"
            )
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


def write_end(output_folder):
    with open(os.path.join(output_folder, "fvModels"), "a+") as f:
        f.write("\t#};\n")
        f.write("};\n")
