import numpy as np

# Conversion factor: Bohr → Ångström
BOHR_TO_ANGSTROM = 0.529177

input_file1 = "data/n2o/random_dimension/test_random_directions_energies_all.tab"
input_file2 = (
    "data/n2o/random_dimension/test_random_directions_shifted_energies_all.tab"
)
output_file1 = (
    "data/n2o/random_dimension/angstrom/test_random_directions_energies_all.tab"
)
output_file2 = (
    "data/n2o/random_dimension/angstrom/test_random_directions_shifted_energies_all.tab"
)

for input_file, output_file in zip(
    [input_file1, input_file2], [output_file1, output_file2]
):
    with open(input_file, "r") as f:
        lines = f.readlines()

    # Separate header and data
    header = lines[0]
    data_lines = lines[1:]

    # Load numeric data
    data = np.loadtxt(data_lines)

    # Columns:
    # 0 = RNN, 1 = RN1O, 2 = RN2O, 3 = ANNO, 4 = EHF, 5 = ECCSDT

    # Convert distances (RNN, RNO, RN2O) from Bohr to Å
    data[:, 0] *= BOHR_TO_ANGSTROM
    data[:, 1] *= BOHR_TO_ANGSTROM
    data[:, 2] *= BOHR_TO_ANGSTROM

    # Save to new file
    with open(output_file, "w") as f:
        f.write(header)  # keep original header
        np.savetxt(f, data, fmt="%15.8f")

    print(f"Converted data saved to {output_file}")
