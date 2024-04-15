from recbole.quick_start import run_recbole

if __name__ == "__main__":
    for _ in range(100):
        for platform in ["LFM", "DEEZER"]:
            for country in ["FR", "DE", "BR", "GLOBAL"]:
                for model in ["ItemKNN", "NeuMF"]:
                    run_recbole(
                        model=model,
                        dataset=platform + "_" + country,
                        config_dict={
                            "field_separator": ",",
                            "platform": platform,
                            "country": country,
                            "topk": [10, 100],
                            "valid_metric": "MRR@10",
                        },
                    )
