import pandas as pd
import numpy as np
from src.utils.visualization import plot_points_3d
from src.utils.helpers import au2angstrom

file_path = "data/h2o/mrci/H2O_ground_state_HF_MRCI.tab"
cols = ["ZH1", "XH2", "ZH2", "HF", "MRCI"]
df = pd.read_csv(file_path, sep=r"\s+", skiprows=1, names=cols)
rOH1 = df["ZH1"].abs()
rOH2 = np.sqrt(df["XH2"] ** 2 + df["ZH2"] ** 2)
rOH2 = np.sqrt(df["XH2"] ** 2 + df["ZH2"] ** 2)
rHH = np.sqrt(df["XH2"] ** 2 + (df["ZH1"] - df["ZH2"]) ** 2)
cos = np.clip(
    (rHH**2 - rOH1**2 - rOH2**2) / (-2 * rOH1 * rOH2),
    -1,
    1,
)
aHOH = np.rad2deg(np.arccos(cos))
# transform distances to angstrom
rOH1 = au2angstrom(rOH1)
rOH2= au2angstrom(rOH2)
# save new dataframe
df_zmat = pd.DataFrame({"rOH1": rOH1, "rOH2": rOH2, "aHOH": aHOH, "MRCI": df["MRCI"]})
print(df_zmat.head())

plot_points_3d(df_zmat.to_numpy()[:, :-1])

df_zmat.to_csv("data/h2o/mrci/H2O_mrci_zmat.csv", index=False)
dataset = pd.read_csv("data/h2o/mrci/H2O_mrci_zmat.csv").to_numpy()
print(dataset)
