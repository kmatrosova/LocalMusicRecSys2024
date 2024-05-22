# Do Recommender Systems Promote Local Music? A Reproducibility Study Using Music Streaming Data


This anonymized repository provides Python code to reproduce experiments from the paper "Do Recommender Systems Promote Local Music? A Reproducibility Study Using Music Streaming Data", submitted to The ACM Conference on Recommender Systems 2024 (RecSys ‘24) in the reproducibility track.


## Abstract

This paper examines the influence of recommender systems on local music representation, discussing prior findings from an empirical study on the LFM-2b public dataset. This prior study argued that different recommender systems exhibit algorithmic biases shifting music consumption either towards or against local content.
However, LFM-2b users do not reflect the diverse audience of music streaming services.
To assess the robustness of this study's conclusions, we conduct a comparative analysis using proprietary listening data from a global music streaming service, which we publicly release alongside this paper. We observe significant differences in local music consumption patterns between our dataset and LFM-2b, suggesting that caution should be exercised when drawing conclusions on local music based solely on LFM-2b.
Moreover, we show that the algorithmic biases exhibited in the original work vary in our dataset, and that several unexplored model parameters can significantly influence these biases and affect the study's conclusion on both datasets. Finally, we discuss the complexity of accurately labeling local music, emphasizing the risk of misleading conclusions due to unreliable, biased, or incomplete labels.

## Datasets

Please download the datasets used in experiments in the links provided below, and put them in the `dataset` directory.
- [LFM-2B](https://drive.google.com/file/d/1a7DG9UNKNZQlXVjS9zoYdzl4jZC06Rhz/view?usp=drive_link)
- XXX (masked for anonymity purpose, link available after publication)

## Environment
- python 3.10.12
- recbole 1.2.0
- pandas 2.2.1
- numpy 1.26.4
- seaborn 0.13.2
- matplotlib 3.8.4

## Onboarding

Change of the path of saved files must be made. First open recbole file `trainer.py`, using nano or any other text editor :

```
nano path_to_recbole/trainer/trainer.py
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