# FairnessRecsys2024

## Onboarding

Install recbole :

`pip install recbole`

More info on :

https://github.com/RUCAIBox/RecBole 

https://recbole.io/

Change the path of saved files. First open recbole file `trainer.py`, using nano or any other text editor :

`nano /opt/anaconda3/envs/tina_env/lib/python3.9/site-packages/recbole/trainer/trainer.py`

Then change the line

`saved_model_file = "{}-{}.pth".format(self.config["model"], get_local_time())`

to

`saved_model_file = "{}/{}/{}.pth".format(self.config["platform"], self.config["country"], self.config["model"])`

## Running the code

Training and testing a model on a specified dataset

`python run.py`

Load trained data and get top k items for each user

`predict.ipynb`

Visualisation

`figures.ipynb`




