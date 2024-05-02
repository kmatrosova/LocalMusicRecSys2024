import pandas as pd
from recbole.quick_start.quick_start import load_data_and_model
from recbole.utils.case_study import *
import os

k = 100

for platform in ["DEEZER"]:
    for country in ["GLOBAL", "FR", "BR", "DE"]:
        for model_name in ["ItemKNN", "NeuMF"]:

            # load prediction data, interactions and users
            # print(platform, country, model_name)
            saved_models_filenames = os.listdir(
                "saved/" + platform + "/" + country + "/" + model_name
            )
            for file_name in saved_models_filenames:
                print(platform, country, model_name, file_name)
                config, model, dataset, train_data, valid_data, test_data = (
                    load_data_and_model(
                        model_file="saved/"
                        + platform
                        + "/"
                        + country
                        + "/"
                        + model_name
                        + "/"
                        + file_name,
                    )
                )
                inter = pd.read_csv(
                    "dataset/"
                    + platform
                    + "_"
                    + country
                    + "/"
                    + platform
                    + "_"
                    + country
                    + ".inter"
                )
                users = inter["user_id:token"].unique().tolist()

                # for each user, get top k predictions
                res = []
                for i, u in enumerate(users):
                    uid_series = dataset.token2id(dataset.uid_field, [str(u)])
                    try:
                        topk_score, topk_iid_list = full_sort_topk(
                            uid_series, model, test_data, k=k, device=config["device"]
                        )
                        # get the item id corresponding to the recbole item id
                        external_item_list = dataset.id2token(
                            dataset.iid_field, topk_iid_list.cpu()
                        )
                        for j in range(0, k):
                            res.append(
                                [
                                    u,
                                    external_item_list[0][j].item(),
                                    topk_score[0][j].item(),
                                ]
                            )
                    except:
                        print("smth went wrong")
                    # if i % 1000 == 0:
                    #     print(i)

                df = pd.DataFrame(res, columns=["user_id", "media_id", "score"])
                df.to_csv(
                    "predicted/"
                    + platform
                    + "/"
                    + country
                    + "/"
                    + model_name
                    + "/"
                    + file_name[:-4]
                    + ".csv"
                )
