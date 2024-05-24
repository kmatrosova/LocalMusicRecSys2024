# Several classes and functions helping making plots

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from globals import *


class MetadataProcesor:
    """Process and format metadata files to use them in MakePlots class"""

    def __init__(self, LFM_metadata_file, DEEZER_metadata_file) -> None:
        self.LFM_metadata_file = LFM_metadata_file
        self.DEEZER_metadata_file = DEEZER_metadata_file

    def clean_metadata(self, df):
        """Keeping wanted data columns"""
        return df[["country", "media_id"]]

    def process(self):
        """Import and clean data"""
        metadata_LFM = pd.read_csv(
            f"./dataset/{self.LFM_metadata_file}.csv", low_memory=False
        )
        metadata_DEEZER = pd.read_csv(
            f"./dataset/{self.DEEZER_metadata_file}.csv", low_memory=False
        )
        metadata_LFM = self.clean_metadata(metadata_LFM)
        metadata_DEEZER = self.clean_metadata(metadata_DEEZER)

        self.metadata = {"DEEZER": metadata_DEEZER, "LFM": metadata_LFM}


class MakePlots:
    """ "General class for making all plots"""

    def __init__(self, metadata, k_values, matadata_filename, global_models) -> None:
        self.metadata = metadata  # metadata attribute of MetadataProcesor class
        self.k_values = k_values  # k values in top-k reco. to consider
        self.matadata_filename = matadata_filename
        self.global_label = "GLOBAL" if global_models else "LOCAL"
        self.global_models = global_models
        if self.global_models:  # Load and transform GLOBAL data
            LFM_user_country = dict(
                pd.read_csv("dataset/LFM_GLOBAL/demo.txt", sep="\t")
                .T.reset_index()
                .T.reset_index(drop=True)
                .reset_index()[["index", 0]]
                .to_numpy()
            )
            DEEZER_user_country = dict(
                pd.read_csv("dataset/DEEZER_GLOBAL/user_country.csv").to_numpy()
            )

            self.user_country_dict = {
                "DEEZER": DEEZER_user_country,
                "LFM": LFM_user_country,
            }

    def get_try_indices(self, filenames):
        if self.n_tries == "max":
            try_indices = range(len(filenames))
        else:
            try_indices = range(self.n_tries)
        return try_indices

    def extract_top_k_reco(self, df, k):
        n_users = len(df.user_id.unique())
        k_max = int(len(df) / n_users)  # getting back k value in top-k reco.
        df["rank"] = list(range(1, k_max + 1)) * n_users  # Adding rank column
        return df[df["rank"] <= k].drop(columns=["rank"])

    def load_predictions(self, n_tries):

        self.n_tries = n_tries
        self.predictions = dict()

        for dataset in PLATFORMS:
            for model in ["NeuMF", "ItemKNN"]:

                if self.global_models:
                    filenames = sorted(
                        os.listdir(f"predicted/{dataset}/GLOBAL/{model}/"),
                        reverse=True,
                    )

                    try_indices = self.get_try_indices(filenames)

                    for try_index in tqdm(
                        try_indices, desc=f"loading {dataset} GLOBAL {model}"
                    ):
                        filename = filenames[try_index]
                        filepath = f"predicted/{dataset}/GLOBAL/{model}/{filename}"
                        all_predictions_df = pd.read_csv(filepath)[
                            ["user_id", "media_id"]
                        ]

                        for country in COUNTRIES:
                            country_user_ids = [
                                int(uid)
                                for uid, user_country in self.user_country_dict[
                                    dataset
                                ].items()
                                if user_country == country
                            ]
                            predictions_df = all_predictions_df[
                                all_predictions_df["user_id"].isin(country_user_ids)
                            ].copy()
                            self.predictions[(dataset, country, model, try_index)] = (
                                self.extract_top_k_reco(
                                    predictions_df, max(self.k_values)
                                )
                            )

                else:  # LOCAL models

                    for country in COUNTRIES:

                        filenames = sorted(
                            os.listdir(f"predicted/{dataset}/{country}/{model}/"),
                            reverse=True,
                        )

                        try_indices = self.get_try_indices(filenames)

                        for try_index in tqdm(
                            try_indices, desc=f"loading {dataset} {country} {model}"
                        ):

                            filename = filenames[try_index]
                            filepath = (
                                f"predicted/{dataset}/{country}/{model}/{filename}"
                            )
                            predictions_df = pd.read_csv(filepath)[
                                ["user_id", "media_id"]
                            ]

                            self.predictions[(dataset, country, model, try_index)] = (
                                self.extract_top_k_reco(
                                    predictions_df, max(self.k_values)
                                )
                            )

    def load_datasets(self):
        self.datasets = dict()
        for dataset in PLATFORMS:
            for country in COUNTRIES:
                print(f"Loading {dataset} {country} dataset")
                filename = f"{dataset}_{country}"
                df = pd.read_csv(f"dataset/{filename}/{filename}.inter")
                df = df.rename(
                    columns={"user_id:token": "user_id", "item_id:token": "media_id"}
                )
                df = pd.merge(df, self.metadata[dataset], on=["media_id"], how="left")

                self.datasets[(dataset, country)] = df

    def plot_dataset_local_streams_percents(self, save=False):
        proportion_local_datasets = []
        for dataset in PLATFORMS:
            for country in COUNTRIES:
                proportion_local_datasets.append(
                    [
                        dataset,
                        country,
                        self.datasets[(dataset, country)]["country"].value_counts(
                            normalize=True
                        )[country],
                    ]
                )

        self.proportion_local_datasets = pd.DataFrame(
            proportion_local_datasets,
            columns=["Dataset", "Country", "Proportion of Local Streams"],
        )

        self.proportion_local_datasets["Dataset_"] = self.proportion_local_datasets[
            "Dataset"
        ].replace({"LFM": "LFM-2b", "DEEZER": "XXX"})

        self.proportion_local_datasets["Country_"] = self.proportion_local_datasets[
            "Country"
        ].replace(COUNTRY_ALIASES)

        sns.set_style("white")
        sns.barplot(
            self.proportion_local_datasets,
            x="Country_",
            y="Proportion of Local Streams",
            hue="Dataset_",
            palette="cool",
            order=["France", "Germany", "Brazil"],
        )
        plt.xticks(fontsize=18)
        plt.yticks([0, 0.1, 0.2, 0.3], fontsize=16)
        plt.xlabel("")
        plt.legend(title="", fontsize=16)
        plt.ylabel("Proportion of Local Streams", fontsize=18)

        if save:
            plt.savefig(
                f"figures/proportion_local_datasets_{self.matadata_filename}.pdf"
            )
        plt.show()
        plt.close()

    def plot_local_listening_distribution_hist(self, save=False):

        for country in COUNTRIES:
            df = self.datasets[("LFM", country)][["user_id", "country"]]
            df = pd.DataFrame(
                df.groupby("user_id").value_counts(normalize=True)
            ).reset_index()
            LFM_proportions_list = df[df["country"] == country].proportion.tolist()

            df = self.datasets[("DEEZER", country)][["user_id", "country"]]
            df = pd.DataFrame(
                df.groupby("user_id").value_counts(normalize=True)
            ).reset_index()
            DEEZER_proportions_list = df[df["country"] == country].proportion.tolist()

            plt.figure()
            sns.set_style("white")
            sns.histplot(
                LFM_proportions_list,
                stat="proportion",
                bins=10,
                color=COUNTRY_COLORS[country],
                label="LFM",
            )
            plt.ylim(0, 0.5)
            plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
            plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            plt.ylabel("User Proportion", fontsize=18)
            plt.xlabel("Proportion of Local Streams", fontsize=18)
            if save:
                plt.savefig(
                    f"./figures/local_listening_distribution_hist_LFM_{country}.pdf"
                )
            plt.plot()
            plt.close()

            plt.figure()
            sns.set_style("white")
            sns.histplot(
                DEEZER_proportions_list,
                stat="proportion",
                bins=10,
                color=COUNTRY_COLORS[country],
                label="DEEZER",
            )

            plt.ylim(0, 0.5)
            plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
            plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            plt.ylabel("User Proportion", fontsize=18)
            plt.xlabel("Proportion of Local Streams", fontsize=18)
            if save:
                plt.savefig(
                    f"./figures/local_listening_distribution_hist_DEEZER_{self.matadata_filename}_{country}.pdf"
                )
            plt.plot()
            plt.close()

    def compute_reco_results(self):

        result_df = []

        for dataset in PLATFORMS:
            for country in COUNTRIES:
                for model in ["NeuMF", "ItemKNN"]:
                    if self.global_models:
                        filenames = sorted(
                            os.listdir(f"predicted/{dataset}/GLOBAL/{model}/"),
                            reverse=True,
                        )
                    else:
                        filenames = sorted(
                            os.listdir(f"predicted/{dataset}/{country}/{model}/"),
                            reverse=True,
                        )

                    try_indices = self.get_try_indices(filenames)

                    for try_index in tqdm(
                        try_indices, desc=f"processing {dataset} {country} {model}"
                    ):
                        predictions_df = self.predictions[
                            (dataset, country, model, try_index)
                        ]
                        predictions_df = pd.merge(
                            predictions_df,
                            self.metadata[dataset],
                            on=["media_id"],
                            how="left",
                        )

                        for k in self.k_values:

                            proportion_local_value = self.extract_top_k_reco(
                                predictions_df, k
                            )["country"].value_counts(normalize=True)[country]
                            result_df.append(
                                [dataset, model, country, proportion_local_value, k]
                            )

        result_df = pd.DataFrame(
            result_df, columns=["Data", "Model", "Country", "% local streams", "k"]
        )

        result_df["bias"] = result_df.apply(
            lambda row: row["% local streams"]
            - self.proportion_local_datasets[
                (self.proportion_local_datasets["Dataset"] == row.Data)
                & (self.proportion_local_datasets["Country"] == row.Country)
            ]["Proportion of Local Streams"].values[0],
            axis=1,
        )

        self.result_df = result_df

    def plot_bias_topk_k_reco(self, save=False):

        for dataset in PLATFORMS:
            filtered_data = self.result_df[
                self.result_df["Data"] == dataset
            ].sort_values(by="k")

            filtered_data["Country"] = filtered_data["Country"].replace(COUNTRY_ALIASES)
            sns.set_style("whitegrid")
            plt.figure(figsize=(14, 7))
            sns.lineplot(
                data=filtered_data,
                x="k",
                y="bias",
                hue="Country",
                style="Model",
                markers=True,
                dashes=False,
                err_style="band",
                hue_order=["France", "Germany", "Brazil"],
                markersize=10,
            )

            plt.axhline(y=0, color="black", linestyle="--")
            plt.text(113.5, 0, "No bias", color="black", ha="right", fontsize=18)
            plt.xticks(self.k_values, fontsize=18)
            plt.yticks(fontsize=18)

            plt.xlabel("k", fontsize=18)
            plt.ylabel("Local Bias", fontsize=18)

            if self.global_label == "LOCAL":
                plt.legend(loc="upper right", bbox_to_anchor=(1, 0.85), fontsize=16)
            else:
                plt.legend().remove()
            if dataset == "LFM":
                plt.ylim(-0.1, 0.08)


            if save:
                if dataset == "LFM":

                    plt.savefig(
                        f"./figures/bias_topk_{dataset}_{self.global_label}.pdf"
                    )
                else:
                    plt.savefig(
                        f"./figures/bias_topk_{dataset}_{self.matadata_filename}_{self.global_label}.pdf"
                    )

            plt.show()
            plt.close()
