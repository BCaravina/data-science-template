import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")
# all ACC and GYR columns to make it easier for using below
predictor_columns = list(df.columns[:6])

# defining plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Dealing with missing values (imputation -> filling in the gaps in the data by using specific methods; we will be using interpolation)
# --------------------------------------------------------------

# checking for the empty values -> from the removed outliers and visualizing a subset to see any empty values
# df.info()
# subset = df[df["set"] == 42]["gyr_y (deg/s)"].plot()

# using pandas function interpolate to add the missing values
for col in predictor_columns:
    df[col] = df[col].interpolate()

df.info()  # no more NaN values

# --------------------------------------------------------------
# Calculating the average duration of the exercise set
# --------------------------------------------------------------

# choosing random subsets to visualize any patterns on the plot -> medium sets are 10 reps and heavy sets are 5 reps
df[df["set"] == 25]["acc_y (g)"].plot()
df[df["set"] == 50]["acc_y (g)"].plot()

# subtracting the first from last timestamp (index) of a set to calculate the timedelta (duration)
duration = df[df["set"] == 1].index[-1] - df[df["set"] == 1].index[0]
duration.seconds

# looping through all sets to calculate their durations
for s in df["set"].unique():
    start = df[df["set"] == s].index[0]
    end = df[df["set"] == s].index[-1]
    duration = end - start
    # adding the duration to the df in a new column while also taking the set into account
    df.loc[(df["set"] == s), "duration"] = duration.seconds

# grouping the df by intensity_category and looking at the duration column and getting the average duration of heavy, medium, sitting and standing sets
duration_df = df.groupby(["intensity_category"])["duration"].mean()
# diving the duration of heavy/medium sets by the number of reps performed to see the avg duration of ONE rep to use in the butterworth lowpass filter below
duration_df.iloc[0] / 5
duration_df.iloc[1] / 10

# --------------------------------------------------------------
# Butterworth lowpass filter -> used to remove high frequency noise from a dataset; most commonly used in ML projects to improve the accuracy of the model
# --------------------------------------------------------------

df_lowpass = df.copy()
LowPass = LowPassFilter()  # creating an instance of the LowPassFilter class we created
sf = 1000 / 200
cutoff = 1.3  # the higher the cutoff, the less smoothing of the data; aka more similar to the original data
df_lowpass = LowPass.low_pass_filter(
    data_table=df_lowpass,
    col="acc_y (g)",
    sampling_frequency=sf,
    cutoff_frequency=cutoff,
    order=5,
)
subset = df_lowpass[df_lowpass["set"] == 45]
print(subset["exercise_label"][0])
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(subset["acc_y (g)"].reset_index(drop=True), label="raw data")
ax[1].plot(
    subset["acc_y (g)_lowpass"].reset_index(drop=True), label="butterworth filter"
)
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)

# applying to filter to all columns
for col in predictor_columns:
    # applying filter to all ACC and GYR data
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, sf, cutoff, order=5)
    # overridding the originals with the filtered columns
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    # deleting the filtered columns to avoid duplicates
    del df_lowpass[col + "_lowpass"]


# --------------------------------------------------------------
# Principal component analysis PCA -> technique used in ML to reduce the complexity of data by transforming the data into a new set of variables called principal components; the transformation is a way that the new set of variables captures the most amount of info from the OG data set, while reducing the number of variables necessary.
# --------------------------------------------------------------

df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()

# determine the optimal amount of principal component (using the functions from the imported PCA class) for the ACC and GYR columns (6 total)
pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

# VISUALIZING THE ELBOW TECHNIQUE FOR SELECTING THE PC VALUES TO KEEP
plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictor_columns) + 1), pc_values)
plt.xlabel("number of principal components")
plt.ylabel("explained variance")
plt.show()

# applying the number of components to the pca function (from the imported class) -> basically summarized the 6 predictor columns into 3 pca columns while explaining as much as the variance as possible
df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)

# using a random subset just to visualize the PCA data is applied
subset = df_pca[df_pca["set"] == 35]
subset[["pca_1", "pca_2", "pca_3"]].plot()

# --------------------------------------------------------------
# Sum of squares attributes -> the advantage of using r versus any particupar data direction is that it is IMPARTIAL TO DEVICE ORIENTATION and can handle dynamic re-orientations.
# --------------------------------------------------------------

df_squared = df_pca.copy()
acc_r = (
    df_squared["acc_x (g)"] ** 2
    + df_squared["acc_y (g)"] ** 2
    + df_squared["acc_z (g)"] ** 2
)

gyr_r = (
    df_squared["gyr_x (deg/s)"] ** 2
    + df_squared["gyr_y (deg/s)"] ** 2
    + df_squared["gyr_z (deg/s)"] ** 2
)

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)

subset = df_squared[df_squared["set"] == 14]
subset[["acc_r", "gyr_r"]].plot(subplots=True)
# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
