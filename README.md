# Sales-Prediction
A forecasting method to optimize promotions and warehouse stocks, based on classical machine learning regression methods.

## Introduction
The presented problem is a forecasting problem based on the sales of 769 stores located on 11 different regions. the data is available for a period of 23 months starting from the 01/03/2016.
We decided to treat our problem as a multi-step time series forecasting problem since we have the sales of each store for each day from March 2016 until the end of February 2018 and the objective is to predict sales of the stores for the period 01/03/2018-30/04/2018 based on the given data.

In order to deal with this problem, two main different approaches have been tried:
- A regression model that does not maintain information of previous predictions
- An autoregressive model that uses information concerning the time maintaining information of previous predictions

## Evaluation Metric
The evaluation metric used for this problem is the following:

<a href="https://www.codecogs.com/eqnedit.php?latex=$$E_r&space;=&space;\frac{\sum_{i\in&space;S_r}{}\sum_{j\in&space;\{3,4\}}{}|a_{i,j}-p_{i,j}|}{\sum_{i\in&space;S_r}{}\sum_{j\in&space;\{3,4\}}{}|a_{i,j}|}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$E_r&space;=&space;\frac{\sum_{i\in&space;S_r}{}\sum_{j\in&space;\{3,4\}}{}|a_{i,j}-p_{i,j}|}{\sum_{i\in&space;S_r}{}\sum_{j\in&space;\{3,4\}}{}|a_{i,j}|}$$" title="$$E_r = \frac{\sum_{i\in S_r}{}\sum_{j\in \{3,4\}}{}|a_{i,j}-p_{i,j}|}{\sum_{i\in S_r}{}\sum_{j\in \{3,4\}}{}|a_{i,j}|}$$" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=$$E&space;=&space;\frac{\sum_{r\in&space;R}{}E_r}{|R|}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$E&space;=&space;\frac{\sum_{r\in&space;R}{}E_r}{|R|}$$" title="$$E = \frac{\sum_{r\in R}{}E_r}{|R|}$$" /></a>

where:
- <a href="https://www.codecogs.com/eqnedit.php?latex=$E_r$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$E_r$" title="$E_r$" /></a>: *Region Error*
- *E*: *Total Error*
- *R*: *Regions*

