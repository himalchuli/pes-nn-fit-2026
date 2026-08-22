import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_tab_file(filename):
    """Loads the file by correctly handling the '#' in headers and whitespace separation."""
    with open(filename, "r") as f:
        header_line = f.readline()
        # Extract column names by stripping '#' and splitting
        header = header_line.strip().lstrip("#").split()

    # Read data using whitespace as separator
    df = pd.read_csv(filename, sep=r"\s+", skiprows=1, names=header)
    return df, header_line


def save_formatted_file(df, output_filename, original_header):
    """Saves the DataFrame using fixed-width formatting to match the original style."""
    with open(output_filename, "w") as f:
        # Ensure header starts with #
        if not original_header.startswith("#"):
            original_header = "# " + original_header.lstrip()
        f.write(original_header)

        # Write rows with fixed widths (mimicking original 18.8f / 16.8f padding)
        for _, row in df.iterrows():
            formatted_row = (
                f"{row['RNN']:18.8f}"
                f"{row['RNO']:16.8f}"
                f"{row['ANNO']:16.8f}"
                f"{row['EHF']:16.8f}"
                f"{row['ECCSDT']:16.8f}\n"
            )
            f.write(formatted_row)


# File configuration
input_files = [
    "data/n2o/N2O_aug-cc-pVQZ_CCSDT_Cs_energies_all.tab",
    "data/n2o/N2O_aug-cc-pVQZ_CCSDT_Cs_shifted_in_all_coordinates_energies_all.tab",
]
output_files = [
    "data/n2o/cut/N2O_aug-cc-pVQZ_CCSDT_Cs_energies_cut.tab",
    "data/n2o/cut/N2O_aug-cc-pVQZ_CCSDT_Cs_shifted_in_all_coordinates_energies_cut.tab",
]

for input_file, output_file in zip(input_files, output_files):
    # 1. Load data
    df, original_header = load_tab_file(input_file)

    ## 1-5. 3D Plotting
    # x = df["RNO"]
    # y = df["ANNO"]
    # z = df["ECCSDT"]

    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection="3d")
    # sc = ax.scatter(x, y, z, c=z, cmap="viridis", s=10)
    # plt.colorbar(sc, label="Energy (ECCSDT)")

    # ax.set_xlabel("RNO ($\AA$)")
    # ax.set_ylabel("ANNO (deg)")
    # ax.set_zlabel("Energy (Hartree)")
    # ax.set_title(f"3D Energy Surface: {input_file}")
    # plt.show()

    # 2. Filter step
    # Removes rows matching specific coordinates
    filtered_df = df[~((df["RNO"] == 1.5))]
    filtered_df = filtered_df[~((filtered_df["RNO"] == 1.6))]

    # 3. Save in original format
    save_formatted_file(filtered_df, output_file, original_header)

    # 4. 3D Plotting
    x = filtered_df["RNO"]
    y = filtered_df["ANNO"]
    z = filtered_df["ECCSDT"]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x, y, z, c=z, cmap="viridis", s=10)
    plt.colorbar(sc, label="Energy (ECCSDT)")

    ax.set_xlabel("RNO ($\AA$)")
    ax.set_ylabel("ANNO (deg)")
    ax.set_zlabel("Energy (Hartree)")
    ax.set_title(f"3D Energy Surface: {input_file}")
    plt.show()

print("Processing complete. Files and plots have been generated.")
