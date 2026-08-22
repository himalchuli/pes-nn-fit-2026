#!/usr/bin/env python3

import sys
import os
import numpy as np

BOHR_TO_ANGSTROM = 0.529177
# to run in command line: for f in *.tab; do     ./convert2angstrom.py "$f"; done
def convert_file(input_path):
    # Resolve absolute paths
    input_path = os.path.abspath(input_path)
    parent_dir = os.path.dirname(input_path)

    # Create output directory: ./angstrom (inside same parent)
    output_dir = os.path.join(parent_dir, "angstrom")
    os.makedirs(output_dir, exist_ok=True)

    # Output file keeps same name
    filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, filename)

    # Read file
    with open(input_path, "r") as f:
        lines = f.readlines()

    header = lines[0]
    data_lines = lines[1:]

    # Load numeric data
    data = np.loadtxt(data_lines)

    # Convert columns 0 and 1 (RNN, RNO)
    data[:, 0] *= BOHR_TO_ANGSTROM
    data[:, 1] *= BOHR_TO_ANGSTROM

    # Write output
    with open(output_path, "w") as f:
        f.write(header)
        np.savetxt(f, data, fmt="%15.8f")

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: convert_to_angstrom.py <input_file>")
        sys.exit(1)

    convert_file(sys.argv[1])