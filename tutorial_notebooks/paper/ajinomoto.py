import matplotlib
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import Plotting Tools
import seaborn as sns

matplotlib.rc("xtick", labelsize=15)
matplotlib.rc("ytick", labelsize=15)


params = {
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
}
pylab.rcParams.update(params)


def preprocess(ajinomoto_df, plot_flag=False, data_col="Value"):
    # Remove unnecessary designs from the data set;
    # add Cycle number, Design number, Batch number and Replicate number columns
    # Lines C1-SX-B2.X-RX are considered as run in Cycle 1 (designs 11 and 32).

    # Add Cycle, Design Number, Batch Number and Replicate

    identifiers = [index.split("-") for index in ajinomoto_df["Line Name"]]
    # identifiers = [
    #   [int(ident[0][1]), int(ident[1][1:]), int(ident[2][3]), int(ident[3][1])]
    #   if len(ident) == 4
    #   else
    #   [int(ident[0][1]), int(ident[1][1:]), int(ident[2][3]), int(ident[3][1]), ident[4]]
    #       for ident in identifiers]

    # Remove 'IPTG' and 'del*' designs
    indices = [len(ident) == 4 for ident in identifiers]
    ajinomoto_df = ajinomoto_df.iloc[indices]
    ajinomoto_df = ajinomoto_df[["Line Name", "Measurement Type", data_col]]

    # Remove C0 '9017' Control Strains from the Dataframe
    ajinomoto_df = ajinomoto_df[~ajinomoto_df["Line Name"].str.contains("S9017")]

    # Add Pathway Number
    ajinomoto_df = assign_metadata(ajinomoto_df)

    ajinomoto_df["Pathway"] = 0
    ajinomoto_df.loc[
        ajinomoto_df["Cycle"].isin([1]) & ajinomoto_df["Design"].isin(range(13)),
        "Pathway",
    ] = 1
    ajinomoto_df.loc[
        ajinomoto_df["Cycle"].isin([1]) & ajinomoto_df["Design"].isin(range(13, 21)),
        "Pathway",
    ] = 2
    ajinomoto_df.loc[
        ajinomoto_df["Cycle"].isin([1]) & ajinomoto_df["Design"].isin(range(31, 35)),
        "Pathway",
    ] = 2
    ajinomoto_df.loc[
        ajinomoto_df["Cycle"].isin([1]) & ajinomoto_df["Design"].isin(range(21, 31)),
        "Pathway",
    ] = 3
    ajinomoto_df.loc[
        ajinomoto_df["Cycle"].isin([1]) & ajinomoto_df["Design"].isin([35, 36]),
        "Pathway",
    ] = 3

    ajinomoto_df.loc[
        ajinomoto_df["Cycle"].isin([2]) & ajinomoto_df["Design"].isin(range(15)),
        "Pathway",
    ] = 1
    ajinomoto_df.loc[
        ajinomoto_df["Cycle"].isin([2]) & ajinomoto_df["Design"].isin(range(15, 25)), "Pathway"
    ] = 2

    ajinomoto_df["Line Name"] = ajinomoto_df["Line Name"].str.replace("-", "_", 2)
    # measurements = [
    #     "A1U2T0",
    #     "A1U3L3",
    #     "Fatty acyl-CoA reductase",
    #     "Dodecanoyl-[acyl-carrier-protein] hydrolase, chloroplastic",
    #     "LCFA_ECOLI",
    #     "AHR_ECOLI",
    #     "dodecan-1-ol",
    # ]

    # noisy_line_names = utils.find_noisy_data(ajinomoto_df, measurements, percentile=98,
    #                                          plot_flag=plot_flag, show_nan=True)

    # Technical Replicates are measured for agreement and quality.

    #
    # # Targeted Proteomics and GC-MS are checked for replicate agreement and relative error is
    # # plotted against the mean value of the replicates.
    #
    # # Plot of Errors grouped per Protocol Type:
    # if plot_flag:
    #     measurements = ['Input Variables', 'Response Variables']
    #     logs = [True, True]
    #
    #     for measurement, log in zip(measurements, logs):
    #         means, percents = utils.analyze_replicate_error(
    #           ajinomoto_df,
    #           ajinomoto_df.columns.get_level_values(0) == measurement
    #         )
    #
    #         ind_nan = np.argwhere(np.isnan(percents))
    #         percents = np.delete(percents, ind_nan[:, 0])
    #         means = np.delete(means, ind_nan[:, 0])
    #
    #         plot.replicate_error(means, percents, measurement, log)
    #         plt.show()
    #
    # # Set to NaN any measurement that has greater than 50% coeff. of variation
    # #
    # # Attempt to clean up the data by dropping measurments with poor agreement.
    #
    # # Set Inclusion Threshold for Data
    # threshold = 50.0  # Percent Replicate Error Allowed
    #
    # # Get all Replicate Pairs
    # columns = ajinomoto_df.columns.get_level_values(0) != 'Metadata'
    #
    # for group in ajinomoto_df.groupby(['Batch', 'Design']):
    #     replicate_df = group[1].iloc[:, columns]
    #
    #     # Check Measurment Accuacy Per Column
    #     percent_df = replicate_df.fillna(0).apply(utils.coeff_of_variation, axis=0)
    #     percent_values = np.array(percent_df.values, dtype=np.float32)
    #
    #     # If its greater than the threshold (50%) Make the Measurement a NaN
    #     nan_cols = list(percent_df.loc[abs(percent_values) > threshold].index)
    #     nan_rows = list(replicate_df.index)
    #
    #     ajinomoto_df.loc[nan_rows, nan_cols] = float('NaN')
    #
    # if plot_flag:
    #     display(
    #       HTML(
    #           'After filtering out replicates with coefficient of variation higher than 50%.'
    #       )
    #     )
    #     for measurement, log in zip(measurements, logs):
    #         means, percents = utils.analyze_replicate_error(
    #           ajinomoto_df,
    #           ajinomoto_df.columns.get_level_values(0)==measurement
    #         )
    #
    #         ind_nan = np.argwhere(np.isnan(percents))
    #         percents = np.delete(percents, ind_nan[:, 0])
    #         means = np.delete(means, ind_nan[:, 0])
    #
    #         plot.replicate_error(means, percents, measurement, log)
    #         plt.show()

    ajinomoto_df["Line Name"] = ajinomoto_df["Line Name"].str.replace("_", "-")

    df_1 = ajinomoto_df[ajinomoto_df["Line Name"].str.contains("C1")]
    df_1 = df_1.pivot(index="Line Name", columns="Measurement Type", values=data_col)
    df_1 = df_1.reset_index()
    df_1 = assign_metadata(df_1)

    df_2 = ajinomoto_df[ajinomoto_df["Line Name"].str.contains("C2")]
    df_2 = df_2.pivot(index="Line Name", columns="Measurement Type", values=data_col)
    df_2 = df_2.reset_index()
    df_2 = assign_metadata(df_2)

    columns_1 = [
        "AHR_ECOLI",
        "LCFA_ECOLI",
        "Dodecanoyl-[acyl-carrier-protein] hydrolase, chloroplastic",
        "Fatty acyl-CoA reductase",
        "A1U2T0",
        "A1U3L3",
        "dodecan-1-ol",
        "Pathway",
    ]

    mean_df_1 = df_1.groupby(["Design"]).mean(numeric_only=True)
    mean_df_1 = mean_df_1.loc[:, mean_df_1.columns.get_level_values(0).isin(columns_1)]
    std_df_1 = df_1.groupby(["Design"]).std(numeric_only=True)
    std_df_1 = std_df_1.loc[:, std_df_1.columns.get_level_values(0).isin(columns_1)]

    mean_df_2_temp = df_2.groupby(["Design"]).mean(numeric_only=True)
    mean_df_2_temp = mean_df_2_temp.loc[
        :, mean_df_2_temp.columns.get_level_values(0).isin(columns_1)
    ]
    std_df_2_temp = df_2.groupby(["Design"]).std(numeric_only=True)
    std_df_2_temp = std_df_2_temp.loc[:, std_df_2_temp.columns.get_level_values(0).isin(columns_1)]

    columns_2 = [
        "LCFA_ECOLI",
        "Dodecanoyl-[acyl-carrier-protein] hydrolase, chloroplastic",
        "A1U2T0",
        "A1U3L3",
        "dodecan-1-ol",
        "Pathway",
    ]
    mean_df_2 = mean_df_2_temp.loc[:, mean_df_2_temp.columns.get_level_values(0).isin(columns_2)]
    std_df_2 = std_df_2_temp.loc[:, std_df_2_temp.columns.get_level_values(0).isin(columns_2)]

    # Cycles 1,2
    mean_df_2_temp = mean_df_2_temp.reset_index(drop=True)
    mean_df_2_temp.index = range(37, 58)

    # TODO: include the 'Pathway' column
    ajinomoto_mean_df = pd.concat((mean_df_1, mean_df_2_temp))
    ajinomoto_mean_df.index.name = "Design"

    if plot_flag:
        # Below are the heatmaps of all of the designs (their mean values and standard deviations
        # normalized with respect to the maximal value) created in cycle 1 and 2, and their
        # production data.
        def col_norm(col):
            return col / max(col)

        # Create Matrix of all categories for the heatmap and Normalize by Column with a maximum at
        # 1 and a min at 0

        # Cycle 1

        plt.figure(figsize=(25, 4))

        plt.subplot(1, 2, 1)
        normal_df = (
            mean_df_1[columns_1[:-1]].fillna(0).apply(col_norm, axis=0).sort_values("dodecan-1-ol")
        )

        # normal_df = mean_df_1.loc[:, mean_df_1.columns.get_level_values(1).isin(columns)]
        # .fillna(0).apply(col_norm,axis=0).sort_values(('Dependent Variable', 'dodecan-1-ol'))
        sns.heatmap(np.transpose(normal_df.values), cmap="viridis")

        # Set Ticks
        x_ticks = normal_df.index.values.astype(int)
        y_ticks = [
            "AHR_ECOLI",
            "LCFA_ECOLI",
            "FATB_UMBCA",
            "ACR1_ACIAD",
            "A1U2T0_MARHV",
            "A1U3L3_MARHV",
            "Dodecanol",
        ]
        y_ticks.reverse()
        ax = plt.gca()
        plt.yticks(rotation=0)
        plt.xticks(rotation=0)
        ax.set_xticklabels(x_ticks)
        ax.set_yticklabels(y_ticks)

        # Format Plot
        plt.tight_layout()
        plt.title("Cycle 1 Design Mean Values Overview")
        plt.xlabel("Designs")
        plt.ylabel("Proteomics")

        plt.subplot(1, 2, 2)
        normal_df = (
            std_df_1[columns_1[:-1]].fillna(0).apply(col_norm, axis=0).sort_values("dodecan-1-ol")
        )
        sns.heatmap(np.transpose(normal_df.values), cmap="viridis")
        # Set Ticks
        ax = plt.gca()
        plt.xticks(rotation=0)
        ax.set_xticklabels(x_ticks)
        plt.yticks([])

        # Format Plot
        plt.tight_layout()
        plt.title("Cycle 1 Design Standard Deviation Overview")
        plt.xlabel("Designs")
        plt.show()

        # Cycle 2
        plt.figure(figsize=(25, 4))

        plt.subplot(1, 2, 1)
        normal_df = (
            mean_df_2[columns_2[:-1]].fillna(0).apply(col_norm, axis=0).sort_values("dodecan-1-ol")
        )
        sns.heatmap(np.transpose(normal_df.values), cmap="viridis")

        # Set Ticks
        x_ticks = normal_df.index.values.astype(int)
        y_ticks = [
            "LCFA_ECOLI",
            "FATB_UMBCA",
            "A1U2T0_MARHV",
            "A1U3L3_MARHV",
            "Dodecanol",
        ]
        y_ticks.reverse()
        ax = plt.gca()
        plt.yticks(rotation=0)
        plt.xticks(rotation=0)
        ax.set_xticklabels(x_ticks)
        ax.set_yticklabels(y_ticks)

        # Format Plot
        plt.tight_layout()
        plt.title("Cycle 2 Design Mean Values Overview")
        plt.xlabel("Designs")
        plt.ylabel("Proteomics")

        plt.subplot(1, 2, 2)
        normal_df = (
            std_df_2[columns_2[:-1]].fillna(0).apply(col_norm, axis=0).sort_values("dodecan-1-ol")
        )
        sns.heatmap(np.transpose(normal_df.values), cmap="viridis")
        # Set Ticks
        ax = plt.gca()
        plt.xticks(rotation=0)
        ax.set_xticklabels(x_ticks)
        plt.yticks([])

        # Format Plot
        plt.tight_layout()
        plt.title("Cycle 2 Design Standard Deviation Overview")
        plt.xlabel("Designs")
        # plt.ylabel('Proteomics')
        plt.show()

        # Cycles 1 and 2
        plt.figure(figsize=(22, 4))

        normal_df = (
            ajinomoto_mean_df[columns_1[:-1]]
            .fillna(0)
            .apply(col_norm, axis=0)
            .sort_values("dodecan-1-ol")
        )
        sns.heatmap(np.transpose(normal_df.values), cmap="viridis")

        # Set Ticks
        x_ticks = normal_df.index.values.astype(int)
        y_ticks = [
            "AHR_ECOLI",
            "LCFA_ECOLI",
            "FATB_UMBCA",
            "ACR1_ACIAD",
            "A1U2T0_MARHV",
            "A1U3L3_MARHV",
            "Dodecanol",
        ]
        y_ticks.reverse()
        ax = plt.gca()
        plt.yticks(rotation=0)
        plt.xticks(rotation=0)
        ax.set_xticklabels(x_ticks)
        ax.set_yticklabels(y_ticks)

        # Format Plot
        plt.tight_layout()
        plt.title("Cycles 1-2 Design Mean Values Overview")
        plt.xlabel("Designs")
        plt.ylabel("Proteomics")

    ajinomoto_df["Line Name"] = ajinomoto_df["Line Name"].str.replace("-", "_", 2)

    return ajinomoto_df, ajinomoto_mean_df


def assign_metadata(df):
    identifiers = [index.split("-") for index in df["Line Name"]]
    identifiers = [
        [int(ident[0][1]), int(ident[1][1:]), int(ident[2][3]), int(ident[3][1])]
        for ident in identifiers
    ]
    df[["Cycle", "Design", "Batch", "Replicate"]] = pd.DataFrame(identifiers, index=df.index)
    return df
