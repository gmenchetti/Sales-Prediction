# Sales-Prediction
A forecasting method to optimize promotions and warehouse stocks, based on classical machine learning regression methods.

## Introduction
The presented problem is a forecasting problem based on the sales of 769 stores located on 11 different regions. the data is available for a period of 23 months starting from the 01/03/2016.
We decided to treat our problem as a multi-step time series forecasting problem since we have the sales of each store for each day from March 2016 until the end of February 2018 and the objective is to predict sales of the stores for the period 01/03/2018-30/04/2018 based on the given data.
In order to deal with this problem, two main different approaches have been tried:
- A regression model that does not maintain information of previous predictions
- An autoregressive model that uses information concerning the time maintaining information of previous predictions
