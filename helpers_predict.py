from recbole.utils.case_study import full_sort_topk


def get_users_top_k_items(users, dataset, model, test_data, K_MAX, config):

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
