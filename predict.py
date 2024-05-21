# Computes and saves predictions for all computed models

import os
import pandas as pd

from recbole.quick_start.quick_start import load_data_and_model

from helpers_predict import get_users_top_k_items


K_MAX = 100  # Computing top-K_MAX recommendations

for platform in ["XXX", "LFM"]:
    for country in ["FR", "BR", "DE", "GLOBAL"]:
        for model in ["ItemKNN", "NeuMF"]:

            saved_models_filenames = os.listdir(f"saved/{platform}/{country}/{model}")

            for file_index, file_name in enumerate(saved_models_filenames):

                print(
                    f"Running : {platform}, {country}, {model}, mode number {file_index + 1}/{len(saved_models_filenames)}"
                )

                config, model, dataset, train_data, valid_data, test_data = (
                    load_data_and_model(
                        model_file=f"saved/{platform}/{country}/{model}/{file_name}"
                    )
                )
                interaction_dataset = pd.read_csv(
                    f"dataset/{platform}_{country}/{platform}_{country}.inter"
                )
                users = interaction_dataset["user_id:token"].unique().tolist()

                top_k_items = get_users_top_k_items(
                    users, dataset, model, test_data, K_MAX, config
                )

                df = pd.DataFrame(top_k_items, columns=["user_id", "media_id", "score"])
                df.to_csv(
                    f"predicted/{platform}/{country}/{model}/{file_name[:-4]}.csv"
                )
