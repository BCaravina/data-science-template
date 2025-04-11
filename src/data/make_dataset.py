import pandas as pd
from glob import glob
import re
# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------

# single_file_acc = pd.read_csv(
#     "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
# )

# single_file_gyr = pd.read_csv(
#     "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
# )

# # --------------------------------------------------------------
# # List all data in data/raw/MetaMotion; glob is a library that lists all the files (can also specify certain extension - *.csv below) within a directory
# # --------------------------------------------------------------

# files = glob("../../data/raw/MetaMotion/*.csv")
# len(files)

# # --------------------------------------------------------------
# # Extract features from filename and adding them to the dataframe
# # --------------------------------------------------------------

# # experimenting with a variable before applying it to the whole list
# data_path = "../../data/raw/MetaMotion/"
# f = files[1]
# # spliting the whole string to get the desired information we want
# participant = f.split("-")[0].replace(data_path, "")
# exercise_label = f.split("-")[1]
# # removing any characters from the right that are 123 (since participants did 3 sets)
# intensity_category = f.split(
#     "-"
# )[
#     2
# ].rsplit(
#     "123"
# )[
#     0
# ]  # for some reason this variable was being returned as a 1 item list, so I added the [0] at the end to fix it

# df = pd.read_csv(f)
# # creating columns and adding them to the df
# df["participant"] = participant
# df["exercise_label"] = exercise_label
# df["intensity_category"] = intensity_category

# # --------------------------------------------------------------
# # Read all files
# # --------------------------------------------------------------

# acc_df = pd.DataFrame()
# gyr_df = pd.DataFrame()
# # the sets are identifyers we will be using later instead of using groupby many times
# acc_set = 1
# gyr_set = 1

# for f in files:
#     participant = f.split("-")[0].replace(data_path, "")
#     exercise_label = f.split("-")[1]
#     # Match one of the valid intensities even if followed by digits or dashes; had to use regex because splitting and replacing was not working
#     intensity_match = re.search(r"(heavy|medium|standing|sitting)", f)
#     intensity_category = intensity_match.group(1) if intensity_match else "unknown"

#     # creating the df and the new columns it should contain
#     df = pd.read_csv(f)
#     df["participant"] = participant
#     df["exercise_label"] = exercise_label
#     df["intensity_category"] = intensity_category

#     # checking if accelerometer or gyroscope data and concatenating it to their respective dfs [constantly overridding the df until they have added all csv files]
#     if "Accelerometer" in f:
#         df["set"] = acc_set
#         acc_set += 1
#         acc_df = pd.concat([acc_df, df])
#     if "Gyroscope" in f:
#         df["set"] = gyr_set
#         gyr_set += 1
#         gyr_df = pd.concat([gyr_df, df])


# # --------------------------------------------------------------
# # Working with datetimes
# # --------------------------------------------------------------

# # pandas recognizes epoch and time (01:00) columns as int and object, respectively, so we gotta convert one of them to datetime object so we can apply methods to them
# # acc_df.info()
# pd.to_datetime(df["epoch (ms)"], unit="ms")

# # setting the epoch columns as the df index for acc and gyr and cleaning the dfs by deleting unnecessary columns
# acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
# gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")
# del acc_df["epoch (ms)"]
# del acc_df["time (01:00)"]
# del acc_df["elapsed (s)"]
# del gyr_df["epoch (ms)"]
# del gyr_df["time (01:00)"]
# del gyr_df["elapsed (s)"]

# --------------------------------------------------------------
# Turn everything into a function
# --------------------------------------------------------------
files = glob("../../data/raw/MetaMotion/*.csv")


def read_data_from_files(files):
    """
    Creates two dataframes from .csv files using the file names.
    Returned dataframes are edited to have a datetime object as the index and four extra columns: participant, exercise_label, intensity_category, and set.
    """
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    acc_set = 1
    gyr_set = 1

    data_path = "../../data/raw/MetaMotion/"
    for f in files:
        participant = f.split("-")[0].replace(data_path, "")
        exercise_label = f.split("-")[1]
        intensity_match = re.search(r"(heavy|medium|standing|sitting)", f)
        intensity_category = intensity_match.group(1) if intensity_match else "unknown"

        df = pd.read_csv(f)
        df["participant"] = participant
        df["exercise_label"] = exercise_label
        df["intensity_category"] = intensity_category

        if "Accelerometer" in f:
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df])
        if "Gyroscope" in f:
            df["set"] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, df])

    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")
    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]
    del gyr_df["epoch (ms)"]
    del gyr_df["time (01:00)"]
    del gyr_df["elapsed (s)"]

    return acc_df, gyr_df


acc_df, gyr_df = read_data_from_files(files)

# --------------------------------------------------------------
# Merging datasets -> creating a single df with both acc and gyr data COLUMN-wise (axis=1)
# --------------------------------------------------------------

# getting the first 3 columns of acc_df and merging it with the full gyr_df so we don't get duplicate columns for participants, exercise, intensity and set
merged_data = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)

# renaming the columns so we know which ones are accelerometer or gyroscope data
merged_data.columns = [
    "acc_x (g)",
    "acc_y (g)",
    "acc_z (g)",
    "gyr_x (deg/s)",
    "gyr_y (deg/s)",
    "gyr_z (deg/s)",
    "participant",
    "exercise_label",
    "intensity_category",
    "set",
]

# --------------------------------------------------------------
# Resample data (frequency conversion)
# Accelerometer: 12.500HZ // Gyroscope: 25.000Hz
# --------------------------------------------------------------
# higher gyroscope frequency = greater number of measurements; so we need to resample to find the best conversion rate to keep as many measurements as possible but ensuring there is data for every row/column
sampling = {
    "acc_x (g)": "mean",
    "acc_y (g)": "mean",
    "acc_z (g)": "mean",
    "gyr_x (deg/s)": "mean",
    "gyr_y (deg/s)": "mean",
    "gyr_z (deg/s)": "mean",
    "participant": "last",
    "exercise_label": "last",
    "intensity_category": "last",
    "set": "last",
}

merged_data[:1000].resample(rule="200ms").apply(sampling)

# spliting the data by day as so to prevent crashing and using too much computing power, since the data acquisition was done within a period of an entire week, and then merging all days into a final, resampled dataframe
days = [g for n, g in merged_data.groupby(pd.Grouper(freq="D"))]
resampled_data = pd.concat(
    [df.resample(rule="200ms").apply(sampling).dropna() for df in days]
)

resampled_data.info()  # checking for nulls and data type

# fixing "set" column data type to int instead of float
resampled_data["set"] = resampled_data["set"].astype("int")

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
