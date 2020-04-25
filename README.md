# Sales-Prediction
This repository contains the implementation of a forecasting method to optimize promotions and warehouse stocks, based on machine learning regression methods for the 2018 Data Mining class at Politecnico di Milano. 

The [paper](https://github.com/gmenchetti/Sales-Prediction/blob/master/docs/Paper.pdf) and the [presentation](https://github.com/gmenchetti/Sales-Prediction/blob/master/docs/Paper.pdf) give more information about the problem and the data.

**NOTE**: The dataset used is not provided since it is not public.

## Introduction
The presented problem is a forecasting problem based on the sales of 769 stores located in 11 different regions. the data is available for a period of 23 months starting from the 01/03/2016.
We decided to treat our problem as a multi-step time series forecasting problem since we have the sales of each store for each day from March 2016 until the end of February 2018 and the objective is to predict sales of the stores for the period 01/03/2018-30/04/2018 based on the given data.

In order to deal with this problem, two main different approaches have been tried:
- A regression model that does not maintain information on previous predictions
- An autoregressive model that uses information concerning the time maintaining information of previous predictions

In both cases, we first applied some preprocessing to the data, then some Machine Learning models have been tested to find the most suitable learning method for this specific problem.

For both methods, we tested different types of regression models, both ensemble or simple models.

The performances are evaluated on a test set built with a cross-validation method for time series data.

## Evaluation Metric
The evaluation metric used for this problem is the following:

<p align="center">
  <a href="https://www.codecogs.com/eqnedit.php?latex=$$E_r&space;=&space;\frac{\sum_{i\in&space;S_r}{}\sum_{j\in&space;\{3,4\}}{}|a_{i,j}-p_{i,j}|}{\sum_{i\in&space;S_r}{}\sum_{j\in&space;\{3,4\}}{}|a_{i,j}|}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$E_r&space;=&space;\frac{\sum_{i\in&space;S_r}{}\sum_{j\in&space;\{3,4\}}{}|a_{i,j}-p_{i,j}|}{\sum_{i\in&space;S_r}{}\sum_{j\in&space;\{3,4\}}{}|a_{i,j}|}$$" title="$$E_r = \frac{\sum_{i\in S_r}{}\sum_{j\in \{3,4\}}{}|a_{i,j}-p_{i,j}|}{\sum_{i\in S_r}{}\sum_{j\in \{3,4\}}{}|a_{i,j}|}$$" /></a>
</p>
<p align="center">
  <a href="https://www.codecogs.com/eqnedit.php?latex=$$E&space;=&space;\frac{\sum_{r\in&space;R}{}E_r}{|R|}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$E&space;=&space;\frac{\sum_{r\in&space;R}{}E_r}{|R|}$$" title="$$E = \frac{\sum_{r\in R}{}E_r}{|R|}$$" /></a>
</p>

where:
- <a href="https://www.codecogs.com/eqnedit.php?latex=$E_r$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$E_r$" title="$E_r$" /></a>: *Region Error*
- <a href="https://www.codecogs.com/eqnedit.php?latex=$E$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$E$" title="$E$" /></a>: *Total Error*
- <a href="https://www.codecogs.com/eqnedit.php?latex=$R$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$R$" title="$R$" /></a>: *Regions*

## Dataset
The provided dataset is composed of 523,021 entries each one representing the sale for a specific store in a specific day. Each entry in the dataset is composed of various features, that can be grouped in the following categories:
- Features concerning the store (ID, region, type of store, type of assortment, nearest competitor)
- Features concerning the day, indicating if the store is open or not, if it has promotions, the number of customers and other information about the weather
- Features concerning the region

The dataset is composed by samples taken from 749 stores, such that:
- 624 stores with 729 samples
- 125 stores with 545 samples

A more detailed description of the dataset can be found in the <a href="https://github.com/gmenchetti/Sales-Prediction/blob/master/docs/Paper.pdf">reference paper</a>, while an analysis of the data is provided in the <a href="https://github.com/gmenchetti/Sales-Prediction/blob/master/Exploratory_Data_analysis.ipynb">Jupyter notebook</a>.

## Implementation
This project was implemented and tested using **Python 3.6**.

Two regression approaches have been explored:
- **Standard Regression**: A model built without features containing information about the number of sales on previous time-steps.
- **Auto Regression**: An autoregressive model that uses informations from previous previously predicted values as input to a regression equation to predict the value of the next time step. The proposed method, builds a model *M* such that: <a href="https://www.codecogs.com/eqnedit.php?latex=$y_{t&plus;1}&space;=&space;M(y_{t},&space;y_{t-1},&space;...,&space;y_{t-n})$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$y_{t&plus;1}&space;=&space;M(y_{t},&space;y_{t-1},&space;...,&space;y_{t-n})$" title="$y_{t+1} = M(y_{t}, y_{t-1}, ..., y_{t-n})$" /></a>. In order to build this model, the train set has been enriched with the following informations:
  - The mean of the *n* previous days
  - The sales of *k* previous time-steps (i.e. the *lag* of the sales)

The above approaches have been tested with the following machine learning algorithms:
- k-Nearest Neighbors
- Random Forest
- Others (Linear Regression, SVM, AdaBoost, Stacking method)

## Results
|**Model**|**Learner**|**Mean error**|**Error variance**|
|---------|-----------|--------------|------------------|
| Regression | k-NN | 0.087 | 0.05 |
| Regression | Random Forest | 0.062 | 0.03 |
| Autoregression | k-NN | 0.097 | 0.06 |
| Autoregression | Random Forest | 0.074 | 0.05 |

Where the *Mean error* is calculated as described in the Evaluation Metric section.

More information about the implementation can be found in the <a href="https://github.com/gmenchetti/Sales-Prediction/blob/master/docs/Paper.pdf">reference paper</a>.


## Other Contributors
* **[Lorenzo Norcini](https://github.com/LorenzoNorcini)**
