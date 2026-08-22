import numpy as np

# Conversion factor: Bohr → Ångström
BOHR_TO_ANGSTROM = 0.529177

input_file1 = "data/n2o/N2O_aug-cc-pVQZ_CCSDT_Cs_3D_training_set_energies_all.tab"
input_file2 = "data/n2o/N2O_aug-cc-pVQZ_CCSDT_Cs_3D_verifying_set_energies_all.tab"
input_file_dim1="data/n2o/dims/N2O_aug-cc-pVQZ_CCSDT_Cs_1D_cut_rNO_training_set_energies_all.tab"
output_file_dim1="data/n2o/angstrom/dims/N2O_aug-cc-pVQZ_CCSDT_Cs_1D_cut_rNO_training_set_energies_all.tab"
input_file_dim2="data/n2o/dims/N2O_aug-cc-pVQZ_CCSDT_Cs_1D_cut_rNO_verifying_set_energies_all.tab"
output_file_dim2="data/n2o/angstrom/dims/N2O_aug-cc-pVQZ_CCSDT_Cs_1D_cut_rNO_verifying_set_energies_all.tab"
input_file_dim3="data/n2o/dims/N2O_aug-cc-pVQZ_CCSDT_Cs_2D_cut_rNO_rNN_training_set_energies_all.tab"
output_file_dim3="data/n2o/angstrom/dims/N2O_aug-cc-pVQZ_CCSDT_Cs_2D_cut_rNO_rNN_training_set_energies_all.tab"
input_file_dim4="data/n2o/dims/N2O_aug-cc-pVQZ_CCSDT_Cs_2D_cut_rNO_rNN_verifying_set_energies_all.tab"
output_file_dim4="data/n2o/angstrom/dims/N2O_aug-cc-pVQZ_CCSDT_Cs_2D_cut_rNO_rNN_verifying_set_energies_all.tab"
output_file1 = "data/n2o/angstrom/N2O_aug-cc-pVQZ_CCSDT_Cs_3D_training_set_energies_all.tab"
output_file2 = "data/n2o/angstrom/N2O_aug-cc-pVQZ_CCSDT_Cs_3D_verifying_set_energies_all.tab"

for input_file, output_file in zip([input_file1,input_file2,input_file_dim1],[output_file1,output_file2]):
    with open(input_file, "r") as f:
        lines = f.readlines()

    # Separate header and data
    header = lines[0]
    data_lines = lines[1:]

    # Load numeric data
    data = np.loadtxt(data_lines)

    # Columns:
    # 0 = RNN, 1 = RNO, 2 = ANNO, 3 = EHF, 4 = ECCSDT

    # Convert distances (RNN, RNO) from Bohr to Å
    data[:, 0] *= BOHR_TO_ANGSTROM
    data[:, 1] *= BOHR_TO_ANGSTROM

    # Save to new file
    with open(output_file, "w") as f:
        f.write(header)  # keep original header
        np.savetxt(f, data, fmt="%15.8f")

    print(f"Converted data saved to {output_file}")