import pandas as pd
from glob import glob

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------

single_file_acc = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
)

single_file_gyr = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
)

# --------------------------------------------------------------
# List all data in data/raw/MetaMotion; glob is a library that lists all the files (can also specify certain extension - *.csv below) within a directory
# --------------------------------------------------------------

files = glob("../../data/raw/MetaMotion/*.csv")
len(files)

# --------------------------------------------------------------
# Extract features from filename and adding them to the dataframe
# --------------------------------------------------------------

# experimenting with a variable before applying it to the whole list
data_path = "../../data/raw/MetaMotion/"
f = files[0]
# spliting the whole string to get the desired information we want
participant = f.split("-")[0].replace(data_path, "")
exercise_label = f.split("-")[1]
# removing any characters from the right that are 123 (since participants did 3 sets)
intensity_category = f.split(
    "-"
)[
    2
].rsplit(
    "123"
)[
    0
]  # for some reason this variable was being returned as a 1 item list, so I added the [0] at the end to fix it


df = pd.read_csv(f)
# creating columns and adding them to the df
df["participant"] = participant
df["exercise_label"] = exercise_label
df["intensity_category"] = intensity_category

# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------


# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------


# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------


# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------


# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
