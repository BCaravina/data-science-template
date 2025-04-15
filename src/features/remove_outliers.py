import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor

# --------------------------------------------------------------
# Load processed data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

# all numerical columns should be analyzed when looking for outliers, so we created a variable to store all numerical data (ACC ang GYR -> the first 6 columns)
outlier_columns = list(df.columns[:6])

# --------------------------------------------------------------
# Plotting outliers -> extreme values that much lower or higher than the majority of the values within the data set; can skew the results and lead to incorrect conclusions about patterns in the data.
# --------------------------------------------------------------

# styling all the subsequent plots at once using rcParams; style + size + export image resolution
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100

df[["acc_y (g)", "exercise_label"]].boxplot(by="exercise_label", figsize=(20, 10))

# making a selection within the dataframe and taking the ACC (:3) and GYR (4:) data columns separately and adding the label column as well

df[outlier_columns[:3] + ["exercise_label"]].boxplot(
    by="exercise_label", figsize=(20, 10), layout=(1, 3)
)
df[outlier_columns[3:] + ["exercise_label"]].boxplot(
    by="exercise_label", figsize=(20, 10), layout=(1, 3)
)
plt.show()


def plot_binary_outliers(dataset, col, outlier_col, reset_index):
    """Plot outliers in case of a binary outlier score. Here, the col specifies the real data
    column and outlier_col the columns with a binary value (outlier or not).

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): Column that you want to plot
        outlier_col (string): Outlier column marked with true/false
        reset_index (bool): whether to reset the index for plotting
    """

    # Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/util/VisualizeDataset.py

    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots()

    plt.xlabel("samples")
    plt.ylabel("value")

    # Plot non outliers in default color
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )
    # Plot data points that are outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )

    plt.legend(
        ["outlier " + col, "no outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.show()


# --------------------------------------------------------------
# Boxplot and Interquartile Range (distribution based)
# --------------------------------------------------------------


# Insert IQR function
def mark_outliers_iqr(dataset, col):
    """Function to mark values as outliers using the IQR method.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
        dataset[col] > upper_bound
    )

    return dataset


# Plot a single column to visualize it
col = "acc_x (g)"
dataset = mark_outliers_iqr(df, col)
plot_binary_outliers(
    dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True
)

# Loop over all columns to observe ACC and GYR data behavior
for col in outlier_columns:
    dataset = mark_outliers_iqr(df, col)
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True
    )

# --------------------------------------------------------------
# Chauvenets criteron (distribution based) - it ASSUMES normal distribution of the data
# --------------------------------------------------------------

# Check for normal distribution (histogram or boxplot -> is the box symmetrical?)
# looka the histograms below -> do we see bell shaped curves? MOSTLY YES except for rest
df[outlier_columns[:3] + ["exercise_label"]].plot.hist(
    by="exercise_label", figsize=(20, 20), layout=(3, 3)
)
df[outlier_columns[3:] + ["exercise_label"]].plot.hist(
    by="exercise_label", figsize=(20, 20), layout=(3, 3)
)
plt.show()


# Insert Chauvenet's function
def mark_outliers_chauvenet(dataset, col, C=2):
    """Finds outliers in the specified column of datatable and adds a binary column with
    the same name extended with '_outlier' that expresses the result per data point.

    Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/Chapter3/OutlierDetection.py

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        C (int, optional): Degree of certainty for the identification of outliers given the assumption
                           of a normal distribution, typicaly between 1 - 10. Defaults to 2.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()
    # Compute the mean and standard deviation.
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []

    # Pass all rows in the dataset.
    for i in range(0, len(dataset.index)):
        # Determine the probability of observing the point
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
        )
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    dataset[col + "_outlier"] = mask
    return dataset


# Loop over all columns
for col in outlier_columns:
    dataset = mark_outliers_chauvenet(df, col)
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True
    )

# --------------------------------------------------------------
# Local outlier factor (distance based)
# --------------------------------------------------------------


# Insert LOF function
def mark_outliers_lof(dataset, columns, n=20):
    """Mark values as outliers using LOF

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        n (int, optional): n_neighbors. Defaults to 20.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    lof = LocalOutlierFactor(n_neighbors=n)
    data = dataset[columns]
    outliers = lof.fit_predict(data)
    X_scores = lof.negative_outlier_factor_

    dataset["outlier_lof"] = outliers == -1
    return dataset, outliers, X_scores


# Loop over all columns
dataset, outliers, X_scores = mark_outliers_lof(df, outlier_columns)
for col in outlier_columns:
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col="outlier_lof", reset_index=True
    )

# --------------------------------------------------------------
# Check outliers grouped by exercise label -> until now we were looking at the WHOLE data set, all exercises at once.
# --------------------------------------------------------------

# comparing the different methods to spot outliers
label = "bench"
for col in outlier_columns:
    dataset = mark_outliers_iqr(df[df["exercise_label"] == label], col)
    plot_binary_outliers(dataset, col, col + "_outlier", reset_index=True)

for col in outlier_columns:
    dataset = mark_outliers_chauvenet(df[df["exercise_label"] == label], col)
    plot_binary_outliers(dataset, col, col + "_outlier", reset_index=True)

dataset, outliers, X_scores = mark_outliers_lof(
    df[df["exercise_label"] == label], outlier_columns
)
for col in outlier_columns:
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col="outlier_lof", reset_index=True
    )

# --------------------------------------------------------------
# Choose method and deal with outliers -> which approach do we want to use + what do we want to do with outlier values?
# --------------------------------------------------------------

# Test on single column
col = "gyr_y (deg/s)"
dataset = mark_outliers_chauvenet(df, col=col)
dataset[dataset["gyr_y (deg/s)_outlier"]]
dataset.loc[dataset["gyr_y (deg/s)_outlier"], "gyr_z (deg/s)"] = np.nan

# Create a loop
outliers_removed_df = df.copy()
for col in outlier_columns:
    for label in df["exercise_label"].unique():
        # creating a subset of the original df based on the value of the for loop and then mark the outliers using chauvenet's criteria
        dataset = mark_outliers_chauvenet(df[df["exercise_label"] == label], col)
        # replace values marked as outliers with NaN
        dataset.loc[dataset[col + "_outlier"], col] = np.nan

        # update the column in the original dataframe
        outliers_removed_df.loc[
            (outliers_removed_df["exercise_label"] == label), col
        ] = dataset[col]
        n_outliers = len(dataset) - len(dataset[col].dropna())
        print(f"Removed {n_outliers} from {col} for {label}")


# we can see the removed values (non-null) from the process above
outliers_removed_df.info()

# --------------------------------------------------------------
# Export new dataframe
# --------------------------------------------------------------

outliers_removed_df.to_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")
