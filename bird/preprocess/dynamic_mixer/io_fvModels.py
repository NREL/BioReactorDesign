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
        f.write("\tcodeAddAlphaRhoSup\n")
        f.write("\t#{\n")
        f.write("\t\tconst Time& time = mesh().time();\n")
        f.write("\t\tconst scalarField& V = mesh().V();\n")
        f.write("\t\tvectorField& Usource = eqn.source();\n")
        f.write("\t\tconst vectorField& C = mesh().C();\n")
        f.write("\t\tconst volScalarField& rhoL =\n")
        f.write(
            '\t\tmesh().lookupObject<volScalarField>("thermo:rho.liquid");\n'
        )
        f.write("\t\tconst volScalarField& alphaL =\n")
        f.write('\t\tmesh().lookupObject<volScalarField>("alpha.liquid");\n')
        f.write("\t\tdouble pi=3.141592654;\n")


def write_mixer(mixer, output_folder):
    with open(os.path.join(output_folder, "fvModels"), "a+") as f:
        f.write(f"\t\tdouble source_pt_x={mixer.x};\n")
        f.write(f"\t\tdouble source_pt_y={mixer.y};\n")
        f.write(f"\t\tdouble source_pt_z={mixer.z};\n")
        f.write(f"\t\tdouble disk_rad={mixer.rad};\n")
        f.write("\t\tdouble disk_area=pi*disk_rad*disk_rad;\n")
        f.write(f"\t\tdouble power={mixer.power};\n")
        f.write("\t\tdouble smear_factor=3.0;\n")
        f.write(f"\t\tconst scalar startTime = {mixer.start_time};\n")
        f.write("\t\tif (time.value() > startTime)\n")
        f.write("\t\t{\n")
        f.write("\t\t\tforAll(C,i)\n")
        f.write("\t\t\t{\n")
        f.write(
            "\t\t\t\tdouble v2=pow((4.0*power/rhoL[i]/disk_area),0.333333);\n"
        )
        f.write("\t\t\t\tdouble Thrust=0.5*rhoL[i]*v2*v2*disk_area;\n")

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
            f"\t\t\t\tUsource[i][{int(mixer.normal_dir)}] {mixer.sign}=  sourceterm*V[i];\n"
        )

        f.write("\t\t\t}\n")
        f.write("\t\t}\n")


def write_end(output_folder):
    with open(os.path.join(output_folder, "fvModels"), "a+") as f:
        f.write("\t#};\n")
        f.write("};\n")
