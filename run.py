# Run and saves Recbole models for different datasets

import numpy as np
from globals import *
from recbole.quick_start import run_recbole

N_REPETITIONS = 2  # Number of time to compute a model with same parameters to control models randomness

if __name__ == "__main__":
    for n in range(N_REPETITIONS):
        for platform in PLATFORMS:
            for country in COUNTRIES + ["GLOBAL"]:
                for model in MODELS:
                    run_recbole(
                        model=model,
                        dataset=f"{platform}_{country}",
                        config_dict={
                            "field_separator": ",",
                            "platform": platform,
                            "country": country,
                            "topk": 10,
                            "valid_metric": "MRR@10",
                            "reproducibility": False,
                            "seed": np.random.randint(0, 10000),
                        },
                    )
