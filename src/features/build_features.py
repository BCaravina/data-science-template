import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans

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
# Temporal abstraction -> using rolling averages w/ pandas
# --------------------------------------------------------------

df_temporal = df_squared.copy()
NumAbs = NumericalAbstraction()

# adding the temporal features to the acc and gyr columns as well as the r squared columns (gyr_r and acc_r)
predictor_columns = predictor_columns + ["acc_r", "gyr_r"]

# window size determines how many values we want to use to predict the mean/standard deviation
ws = int(
    1000 / 200
)  # step size is of 200ms, so to get a ws of 1 second we need 5 step sizes

# looping through all the columns and compute the mean and stand dev
for col in predictor_columns:
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "mean")
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "std")

# since the ws uses the previous 4 values to compute the mean and std, this brings up two problems: first 4 rows won't have values computed, AND bench press data will be used to compute the means for squats or deadlifts or rows...so we need to fix that by making SUBSETS for each label
df_temporal_list = []
for s in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"] == s].copy()

    for col in predictor_columns:
        subset = NumAbs.abstract_numerical(subset, [col], ws, "mean")
        subset = NumAbs.abstract_numerical(subset, [col], ws, "std")
    df_temporal_list.append(subset)

# turning the list with the subsets into a single dataframe again (overriding the OG dataframe)
df_temporal = pd.concat(df_temporal_list)

df_temporal.info()  # we can see we lost some of the data on the new columns, but that is good because we are avoiding the "spillage" of mismatched label data into another label type (ex: bench into squat OR squat into row, etc)

# using a random subset to visualize the data
subset[["acc_r", "acc_y (g)_temp_mean_ws_5", "acc_y (g)_temp_std_ws_5"]].plot()
subset[["gyr_r", "gyr_y (deg/s)_temp_mean_ws_5", "gyr_y (deg/s)_temp_std_ws_5"]].plot()

# --------------------------------------------------------------
# Frequency features -> applying the Discrete Fourier Transformation (DFT)
# --------------------------------------------------------------

df_frequency = df_temporal.copy().reset_index()
FreqAbs = FourierTransformation()

# frequency sampling rate and window size
fs = int(1000 / 200)
ws = int(2800 / 200)

df_frequency = FreqAbs.abstract_frequency(df_frequency, ["acc_y (g)"], ws, fs)

# repeating the steps of the step above, creating empty list and populating it with the subset lists then concatenating them all back into the dataframe
df_frequency_list = []
for s in df_frequency["set"].unique():
    print(f"Applying Fourier transformations to set {s}")
    subset = df_frequency[df_frequency["set"] == s].reset_index(drop=True).copy()
    subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, fs)
    df_frequency_list.append(subset)

df_frequency = pd.concat(df_frequency_list).set_index("epoch (ms)", drop=True)

# --------------------------------------------------------------
# Dealing with overlapping windows -> avoiding possible overfitting
# --------------------------------------------------------------

dr_frequency = df_frequency.dropna()
df_frequency = df_frequency.iloc[
    ::2
]  # removing EVERY OTHER row of the dataframe -> reducing data in 50%


# --------------------------------------------------------------
# Clustering -> K-Means clustering
# --------------------------------------------------------------

df_cluster = df_frequency.copy()
cluster_columns = ["acc_x (g)", "acc_y (g)", "acc_z (g)"]
k_values = range(2, 10)
inertias = []

for k in k_values:
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 10))
plt.plot(k_values, inertias)
plt.xlabel("k")
plt.ylabel("Sum of squared distances")
plt.show()

# selecting the optimal n_cluster number after visualizing the plot above
kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
subset = df_cluster[cluster_columns]
df_cluster["cluster"] = kmeans.fit_predict(subset)

# visually checking if the selected clusters make sense
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for c in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"] == c]
    ax.scatter(subset["acc_x (g)"], subset["acc_y (g)"], subset["acc_z (g)"], label=c)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

# visually comparing the splitting of the data by clusters vs EXERCISE LABEL clusters
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for l in df_cluster["exercise_label"].unique():
    subset = df_cluster[df_cluster["exercise_label"] == l]
    ax.scatter(subset["acc_x (g)"], subset["acc_y (g)"], subset["acc_z (g)"], label=l)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

# comparing the 3D plots above we can see which cluster number corresponds to which exercises

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_cluster.to_pickle("../../data/interim/03_data_features.pkl")
