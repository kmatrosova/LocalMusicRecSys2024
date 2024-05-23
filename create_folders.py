import os
from globals import *

base_dirs = ["dataset", "saved", "predicted"]


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory {path} created.")
    else:
        print(f"Directory {path} already exists.")


for base in base_dirs:
    create_dir(base)
    if base in ["saved", "predicted"]:
        for platform in PLATFORMS:
            platform_path = os.path.join(base, platform)
            create_dir(platform_path)
            for country in COUNTRIES + ["GLOBAL"]:
                country_path = os.path.join(platform_path, country)
                create_dir(country_path)
                for model in MODELS:
                    model_path = os.path.join(country_path, model)
                    create_dir(model_path)

print("Directory structure created successfully.")
