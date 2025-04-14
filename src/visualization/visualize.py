import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------

# taking a subset of the original data frame -> only data where the set column equals 1
set_df = df[df["set"] == 1]
plt.plot(set_df["acc_y (g)"])
# droping the timestamp index to regular integers so that we can see the amount of entries (~102) on the X-axis, whereas above we can only see the duration (~15-20seconds)
plt.plot(set_df["acc_y (g)"].reset_index(drop=True))


# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------

# using the unique() method to return a list of all the unique values within the specified column
df["exercise_label"].unique()
# using the list above to loop through all the individual labels to create the subsets and plot the data
for label in df["exercise_label"].unique():
    subset = df[df["exercise_label"] == label]
    fig, ax = plt.subplots()
    plt.plot(subset["acc_y (g)"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()

# getting only the first 100 values for each subset to better visualize the data
for label in df["exercise_label"].unique():
    subset = df[df["exercise_label"] == label]
    fig, ax = plt.subplots()
    plt.plot(subset[:100]["acc_y (g)"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()

# --------------------------------------------------------------
# Adjust plot settings -> using matplotlib's rcParams to ensure all plotted figures have the same styling/formatting
# --------------------------------------------------------------

mpl.style.use("seaborn-v0_8-deep")  # styling
mpl.rcParams["figure.figsize"] = (20, 5)  # more spread out data
mpl.rcParams["figure.dpi"] = 100  # exporting figures w/ correct resolution

# --------------------------------------------------------------
# Compare figures of medium vs. heavy sets
# --------------------------------------------------------------

# another way to create subsets using pandas; we're using a string match, so the query has to be in "" and the string match has to be in ''; we can also stack multiple queries together
category_df = (
    df.query("exercise_label == 'squat'").query("participant == 'A'").reset_index()
)

# creating a group plot (grouping by intensity) and defining the column we want to plot (acc_y in this case)
fig, ax = plt.subplots()
category_df.groupby(["intensity_category"])["acc_y (g)"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()

# --------------------------------------------------------------
# Compare participants -> do the plots for each exercise look the same for different participants? a.k.a. can it be
# --------------------------------------------------------------

participant_df = (
    df.query("exercise_label == 'bench'").sort_values("participant").reset_index()
)

fig, ax = plt.subplots()
participant_df.groupby(["participant"])["acc_y (g)"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()

# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------

exercise_label = "squat"
participant = "A"
all_axis_df = (
    df.query(f"exercise_label == '{exercise_label}'")
    .query(f"participant == '{participant}'")
    .reset_index()
)

fig, ax = plt.subplots()
# have to be in double curly brackets so it is a PANDAS DATAFRAME, not a SERIES
all_axis_df[["acc_x (g)", "acc_y (g)", "acc_z (g)"]].plot(ax=ax)
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()

# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------

# getting all the values for the labels and participants
exercise_labels = df["exercise_label"].unique()
participants = df["participant"].unique()

# looping through each label AND each paricipant within each label and creating a df for each for the ACC DATA
for label in exercise_labels:
    for participant in participants:
        all_axis_df = (
            df.query(f"exercise_label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
        )

        if len(all_axis_df) > 0:
            fig, ax = plt.subplots()
            all_axis_df[["acc_x (g)", "acc_y (g)", "acc_z (g)"]].plot(ax=ax)
            ax.set_ylabel("acc_y")
            ax.set_xlabel("samples")
            plt.title(f"{label} - Participant {participant}".title())
            plt.legend()

# looping through each label AND each paricipant within each label and creating a df for each for the GYR DATA
for label in exercise_labels:
    for participant in participants:
        all_axis_df = (
            df.query(f"exercise_label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
        )

        if len(all_axis_df) > 0:
            fig, ax = plt.subplots()
            all_axis_df[["gyr_x (deg/s)", "gyr_y (deg/s)", "gyr_z (deg/s)"]].plot(ax=ax)
            ax.set_ylabel("gyr_y")
            ax.set_xlabel("samples")
            plt.title(f"{label} - Participant {participant}".title())
            plt.legend()

# --------------------------------------------------------------
# Combine plots in one figure -> one figure will contain a participants ACC + GYR for one exercise
# --------------------------------------------------------------

label = "row"
participant = "A"
combined_plot_df = (
    df.query(f"exercise_label == '{label}'")
    .query(f"participant == '{participant}'")
    .reset_index(drop=True)
)

# creating two figures in one, ACC on top [0] row and GYR on bottom [1] row
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
combined_plot_df[["acc_x (g)", "acc_y (g)", "acc_z (g)"]].plot(ax=ax[0])
combined_plot_df[["gyr_x (deg/s)", "gyr_y (deg/s)", "gyr_z (deg/s)"]].plot(ax=ax[1])

# adding styling to the legends
ax[0].legend(
    loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True
)
ax[1].legend(
    loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True
)
ax[1].set_xlabel("samples")

# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------

exercise_labels = df["exercise_label"].unique()
participants = df["participant"].unique()

for label in exercise_labels:
    for participant in participants:
        combined_plot_df = (
            df.query(f"exercise_label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
        )

        if len(combined_plot_df) > 0:
            fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
            combined_plot_df[["acc_x (g)", "acc_y (g)", "acc_z (g)"]].plot(ax=ax[0])
            combined_plot_df[["gyr_x (deg/s)", "gyr_y (deg/s)", "gyr_z (deg/s)"]].plot(
                ax=ax[1]
            )

            ax[0].legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.15),
                ncol=3,
                fancybox=True,
                shadow=True,
            )
            ax[1].legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.15),
                ncol=3,
                fancybox=True,
                shadow=True,
            )
            ax[1].set_xlabel("samples")
            plt.savefig(f"../../reports/figures/{label.title()} ({participant}).png")
            plt.show()
