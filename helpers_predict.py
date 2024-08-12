from recbole.utils.case_study import full_sort_topk
import pandas as pd


def get_users_top_k_items(users, dataset, model, test_data, K_MAX, config):
    """Computes users top k items from a trained model"""
    top_k_items = []

    for u in users:
        uid_series = dataset.token2id(dataset.uid_field, [str(u)])

        try:
            topk_score, topk_iid_list = full_sort_topk(
                uid_series, model, test_data, k=K_MAX, device=config["device"]
            )

            # get the item id corresponding to the recbole item id
            external_item_list = dataset.id2token(
                dataset.iid_field, topk_iid_list.cpu()
            )
            for j in range(0, K_MAX):
                top_k_items.append(
                    [
                        u,
                        external_item_list[0][j].item(),
                        topk_score[0][j].item(),
                    ]
                )
        except:
            print("Something went wrong")

    return top_k_items


def get_users_from_interaction_dataset(platform, country):
    """Gets the user_id list from interaction dataset"""
    interaction_dataset = pd.read_csv(
        f"dataset/{platform}_{country}/{platform}_{country}.inter"
    )
    users = interaction_dataset["user_id:token"].unique().tolist()
    return users


def save_predictions(top_k_items, platform, country, model, file_name):

    df = pd.DataFrame(top_k_items, columns=["user_id", "item_id", "score"])
    df.to_csv(f"predicted/{platform}/{country}/{model}/{file_name[:-4]}.csv")
