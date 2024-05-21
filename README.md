# FairnessRecsys2024

Code for reproducibility paper "Do Recommender Systems Promote Local Music? A Reproducibility Study Using Music Streaming Data"  
(Anonymous athor(s))

## Onboarding

Install recbole :

```
pip install recbole
```

More info on :

https://github.com/RUCAIBox/RecBole 

https://recbole.io/

Change the path of saved files. First open recbole file `trainer.py`, using nano or any other text editor :

```
nano /opt/anaconda3/envs/tina_env/lib/python3.9/site-packages/recbole/trainer/trainer.py
```

Add the following line in the begining of the file :

```
import pathlib
```

Then change the line

```
saved_model_file = "{}-{}.pth".format(self.config["model"], get_local_time())
```

to

```
saved_model_file = f"{self.config["platform"]}/{self.config["country"]}/{self.config["model"]}/get_local_time().pth"
```

## Running the code

Training and testing a model on a specified dataset

```
python run.py
```

Get and save top k items for each user for each model

```
python predict.py
```

Make figures

```
python figures.py
```

## Data and model folders

```
|
|
|____________ dataset
|                |
|                |
|                |____________ XXX_FR
|                |                |____________ XXX_FR.inter
|                |
|                |____________ XXX_BR
|                |                |____________ XXX_BR.inter
|                |
|                |____________ XXX_DE
|                |                |____________ XXX_DE.inter
|                |
|                |____________ XXX_GLOBAL
|                |                |____________ XXX_GLOBAL.inter
|                |
|                |____________ LFM_FR
|                |                |____________ LFM_FR.inter
|                |
|                |____________ LFM_BR
|                |                |____________ LFM_BR.inter
|                |
|                |____________ LFM_DE
|                |                |____________ LFM_DE.inter
|                |
|                |____________ LFM_GLOBAL
|                                 |____________ LFM_GLOBAL.inter
|
|
|____________ saved
|                |
|                |
|                |____________   XXX
|                |                |____________ FR
|                |                |              |___________ ItemKNN
|                |                |              |___________ NeuMF
|                |                |     
|                |                |____________ BR
|                |                |              |___________ ItemKNN
|                |                |              |___________ NeuMF
|                |                |     
|                |                |____________ DE
|                |                |              |___________ ItemKNN
|                |                |              |___________ NeuMF
|                |                |     
|                |                |____________ GLOBAL
|                |                               |___________ ItemKNN
|                |                               |___________ NeuMF
|                |____________   LFM
|                                 |____________ FR
|                                 |              |___________ ItemKNN
|                                 |              |___________ NeuMF
|                                 |     
|                                 |____________ BR
|                                 |              |___________ ItemKNN
|                                 |              |___________ NeuMF
|                                 |     
|                                 |____________ DE
|                                 |              |___________ ItemKNN
|                                 |              |___________ NeuMF
|                                 |     
|                                 |____________ GLOBAL
|                                                |___________ ItemKNN
|                                                |___________ NeuMF
|
|
|____________ predicted
                 |
                 |
                 |____________   XXX
                 |                |____________ FR
                 |                |              |___________ ItemKNN
                 |                |              |___________ NeuMF
                 |                |     
                 |                |____________ BR
                 |                |              |___________ ItemKNN
                 |                |              |___________ NeuMF
                 |                |     
                 |                |____________ DE
                 |                |              |___________ ItemKNN
                 |                |              |___________ NeuMF
                 |                |     
                 |                |____________ GLOBAL
                 |                               |___________ ItemKNN
                 |                               |___________ NeuMF
                 |____________   LFM
                                  |____________ FR
                                  |              |___________ ItemKNN
                                  |              |___________ NeuMF
                                  |     
                                  |____________ BR
                                  |              |___________ ItemKNN
                                  |              |___________ NeuMF
                                  |     
                                  |____________ DE
                                  |              |___________ ItemKNN
                                  |              |___________ NeuMF
                                  |     
                                  |____________ GLOBAL
                                                 |___________ ItemKNN
                                                 |___________ NeuMF
```






