# Computes and saves predictions for all computed models

import os
from globals import *
from recbole.quick_start.quick_start import load_data_and_model

from helpers_predict import (
    get_users_top_k_items,
    get_users_from_interaction_dataset,
    save_predictions,
)


K_MAX = 100  # Computing top-K_MAX recommendations

for platform in PLATFORMS:
    for country in COUNTRIES + ["GLOBAL"]:
        for model in MODELS:

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

                users = get_users_from_interaction_dataset(platform, country)
                top_k_items = get_users_top_k_items(
                    users, dataset, model, test_data, K_MAX, config
                )
                save_predictions(top_k_items, platform, country, model, file_name)
