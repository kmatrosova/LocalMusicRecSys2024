import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


class MetadataProcesor:
    def __init__(self, LFM_metadata_file, DEEZER_metadata_file) -> None:
        self.LFM_metadata_file = LFM_metadata_file
        self.DEEZER_metadata_file = DEEZER_metadata_file

    def clean_metadata(self, df):
        return df[["country", "media_id"]]

    def process(self):
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
    def __init__(self, metadata, k_values, matadata_filename, global_models) -> None:
        self.metadata = metadata
        self.k_values = k_values
        self.matadata_filename = matadata_filename
        self.global_label = 'GLOBAL' if global_models else 'LOCAL'
        self.global_models = global_models
        if self.global_models:
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


    def extract_top_k_reco(self, df, k):
        n_users = len(df.user_id.unique())
        k_max = int(len(df) / n_users)
        df["rank"] = list(range(1, k_max + 1)) * n_users
        return df[df["rank"] <= k].drop(columns=["rank"])

    def load_predictions(self, n_tries):
        self.n_tries = n_tries
        self.predictions = dict()
        for dataset in ["LFM", "DEEZER"]:
            for model in ["NeuMF", "ItemKNN"]:
                if self.global_models:
                    filenames = sorted(
                        os.listdir(f"predicted/{dataset}/GLOBAL/{model}/"),
                        reverse=True,
                    )
                    if n_tries == 'max':
                        try_indices = range(len(filenames))
                    else:
                        try_indices = range(n_tries)
                    for try_index in tqdm(
                        try_indices, desc=f"loading {dataset} GLOBAL {model}"
                    ):
                        filename = filenames[try_index]
                        filepath = f"predicted/{dataset}/GLOBAL/{model}/{filename}"
                        all_predictions_df = pd.read_csv(filepath)[["user_id", "media_id"]]
                        for country in ["FR", "DE", "BR"]:
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
                            predictions_df = self.extract_top_k_reco(
                                predictions_df, max(self.k_values)
                            )
                            self.predictions[(dataset, country, model, try_index)] = (
                                predictions_df
                            )
                else:
                    for country in ["FR", "DE", "BR"]:

                        filenames = sorted(
                            os.listdir(f"predicted/{dataset}/{country}/{model}/"),
                            reverse=True,
                        )
                        if n_tries == 'max':
                            try_indices = range(len(filenames))
                        else:
                            try_indices = range(n_tries)
                        for try_index in tqdm(
                            try_indices, desc=f"loading {dataset} {country} {model}"
                        ):

                            filename = filenames[try_index]
                            filepath = f"predicted/{dataset}/{country}/{model}/{filename}"
                            predictions_df = pd.read_csv(filepath)[["user_id", "media_id"]]

                            self.predictions[(dataset, country, model, try_index)] = (
                                self.extract_top_k_reco(
                                    predictions_df, max(self.k_values)
                                )
                            )

    def load_datasets(self):

        self.datasets = dict()
        for dataset in ["DEEZER", "LFM"]:
            for country in ["DE", "BR", "FR"]:
                print(f"Loading {dataset} {country} dataset")
                filename = dataset + "_" + country
                df = pd.read_csv(f"dataset/{filename}/{filename}.inter")
                df = df.rename(
                    columns={"user_id:token": "user_id", "item_id:token": "media_id"}
                )
                df = pd.merge(df, self.metadata[dataset], on=["media_id"], how="left")

                self.datasets[(dataset, country)] = df

    def plot_dataset_local_streams_percents(self, save=False):
        proportion_local_datasets = []
        for dataset in ["DEEZER", "LFM"]:
            for country in ["DE", "BR", "FR"]:
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
            columns=["Dataset", "Country", "Proportion of local streams"],
        )
        sns.set_style("whitegrid")
        sns.barplot(
            self.proportion_local_datasets,
            x="Country",
            y="Proportion of local streams",
            hue="Dataset",
        )
        plt.legend(title="")
        plt.xlabel("")
        if save:
            plt.savefig(
                f"figures/proportion_local_datasets_{self.matadata_filename}.pdf"
            )
        plt.show()
        plt.close()

    def plot_local_listening_distribution_hist(self, save=False):

        country_colors = ["tab:blue", "tab:green" ,"tab:orange"]
        for i, country in enumerate(["FR", "BR", "DE"]):
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
            sns.histplot(
                LFM_proportions_list,
                stat="proportion",
                bins=20,
                color=country_colors[i],
                label="LFM",
            )
            plt.title(f"{country}")
            plt.ylim(0, 0.5)
            sns.set_style("whitegrid")
            if save:
                plt.savefig(
                    f"./figures/local_listening_distribution_hist_LFM_{country}.pdf"
                )
            plt.plot()
            plt.close()

            plt.figure()
            sns.histplot(
                DEEZER_proportions_list,
                stat="proportion",
                bins=20,
                color=country_colors[i],
                label="DEEZER",
            )

            plt.title(f"{country}")
            plt.ylim(0, 0.5)
            sns.set_style("whitegrid")
            if save:
                plt.savefig(
                    f"./figures/local_listening_distribution_hist_DEEZER_{self.matadata_filename}_{country}.pdf"
                )
            plt.plot()
            plt.close()

    def compute_reco_results(self):

        result_df = []

        for dataset in ["LFM", "DEEZER"]:
            for country in ["FR", "DE", "BR"]:
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

                    if self.n_tries == 'max':
                        try_indices = range(len(filenames))
                    else:
                        try_indices = range(self.n_tries)

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
            ]["Proportion of local streams"].values[0],
            axis=1,
        )

        self.result_df = result_df

    def plot_bias_topk_k_reco(self, save=False):

        sns.set_style("whitegrid")
        for dataset in ["LFM", "DEEZER"]:
            filtered_data = self.result_df[
                self.result_df["Data"] == dataset
            ].sort_values(by="k")
            plt.figure(figsize=(12, 7))
            sns.lineplot(
                data=filtered_data,
                x="k",
                y="bias",
                hue="Country",
                style="Model",
                markers=True,
                dashes=False,
                err_style="band",
                hue_order=['FR', 'DE', 'BR']
            )

            plt.axhline(y=0, color="black", linestyle="--")
            plt.text(113, 0, "No bias", color="black", ha="right")
            plt.xticks(self.k_values)

            plt.xlabel("k")
            plt.ylabel("% local recommended songs bias value")
            plt.legend(loc="center left", bbox_to_anchor=(1, 0.85))

            if save:
                if dataset == 'LFM':

                    plt.savefig(f"./figures/bias_topk_{dataset}_{self.global_label}.pdf")
                else:
                    plt.savefig(f"./figures/bias_topk_{dataset}_{self.matadata_filename}_{self.global_label}.pdf")


            plt.show()
            plt.close()
