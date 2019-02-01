# Sales-Prediction
A forecasting method to optimize promotions and warehouse stocks, based on classical machine learning regression methods.

## Introduction
The presented problem is a forecasting problem based on the sales of 769 stores located on 11 different regions. the data is available for a period of 23 months starting from the 01/03/2016.
We decided to treat our problem as a multi-step time series forecasting problem since we have the sales of each store for each day from March 2016 until the end of February 2018 and the objective is to predict sales of the stores for the period 01/03/2018-30/04/2018 based on the given data.

In order to deal with this problem, two main different approaches have been tried:
- A regression model that does not maintain information of previous predictions
- An autoregressive model that uses information concerning the time maintaining information of previous predictions

In both cases, we first applied some preprocessing to the data, then some Machine Learning models have been tested in order to find the most suitable learning method for this specific problem.

For both methods we tested different type of regression model, both ensemble or simple models.

The performance are evaluated on a test set built with a cross-validation method for time series data.

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

A more detailed description of the dataset can be found in the <a href="https://github.com/gmenchetti/Sales-Prediction/blob/master/Paper/Menchetti-Norcini.pdf">reference paper</a>, while an analysis of the data is provided in the <a href="https://github.com/gmenchetti/Sales-Prediction/blob/master/Exploratory_Data_analysis.ipynb">Jupyter notebook</a>.

