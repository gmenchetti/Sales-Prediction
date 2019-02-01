from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from scipy.stats.mstats import winsorize
from sklearn.svm import LinearSVR, SVR
from sklearn.metrics import r2_score
from sklearn.utils import shuffle
from datetime import datetime, timedelta
from time import time
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import gumbel_l, gamma
from tqdm import *
import warnings
import copy

warnings.simplefilter('ignore')

features = [
	'StoreID', 'Date', 'IsOpen', 'IsHoliday', 'HasPromotions', 'NumberOfSales',
	'StoreType_Hyper Market','StoreType_Shopping Center', 'StoreType_Standard Market',
	'StoreType_Super Market',
	'NearestCompetitor',
	'WeekDay_0', 'WeekDay_1', 'WeekDay_2','WeekDay_3',
	'WeekDay_4', 'WeekDay_5', 'WeekDay_6', 'Region_0',
	'Region_1', 'Region_2', 'Region_3', 'Region_4', 'Region_5',
	'Region_6','Region_7', 'Region_8', 'Region_9', 'Region_10',
	'Month_1', 'Month_2','Month_3', 'Month_4', 'Month_5', 'Month_6',
	'Month_7', 'Month_8','Month_9', 'Month_10', 'Month_11', 'Month_12',
	'AssortmentType_General','AssortmentType_With Fish Department', 'AssortmentType_With Non-Food Department'
]

class TrainPreprocessing():

	def __init__(self, data, to_date):

		self.data = copy.deepcopy(data)

		fields = ["StoreType", "WeekDay", "Region", "Month", "AssortmentType"]

		self.data["Date"] = pd.to_datetime(self.data['Date'], format='%d/%m/%Y')

		self.add_date_columns(date_column_name="Date", month=True, day_of_week=True)

		self.one_hot_encode(fields=fields)

		self.data = self.data[features]

		self.data = self.data[self.data["Date"] < to_date]

		self.data = self.data[self.data["IsOpen"] == 1]

		self.X = np.array(self.data.drop(["StoreID", "IsOpen", "Date", "NumberOfSales"], inplace=False, axis = 1))
		self.y = np.array(self.data["NumberOfSales"])

	def get_X_y(self):
		return self.X, self.y

	def one_hot_encode(self, fields):
		self.data = pd.get_dummies(self.data, columns=fields)

	def split_attribute_list(self, column, attributes, fillna):
		mlb = MultiLabelBinarizer(classes=attributes)
		if fillna is not None:
			self.data[column] = self.data[column].fillna(fillna, inplace=False)
		self.data[column] = self.data[column].apply(lambda x: x.split('-'))
		new_columns_values = mlb.fit_transform(self.data[column].values.tolist())
		self.data[attributes] = pd.DataFrame(new_columns_values, index=self.data.index)

	def add_date_columns(self, date_column_name, year=False, month=False, day_n=False, day_of_week=False):
		if year:
			self.data["Year"] = self.data[date_column_name].dt.year
		if month:
			self.data["Month"] = self.data[date_column_name].dt.month
		if day_n:
			self.data["Day"] = self.data[date_column_name].dt.day
		if day_of_week :
			self.data["WeekDay"] = self.data[date_column_name].dt.dayofweek


class TestPreprocessing():

	def __init__(self, data, from_date):

		self.data = copy.deepcopy(data)

		fields = ["StoreType", "WeekDay", "Region", "Month", "AssortmentType"]

		self.data["Date"] = pd.to_datetime(self.data['Date'], format='%d/%m/%Y')

		self.add_date_columns(date_column_name="Date", month=True, day_of_week=True)

		self.one_hot_encode(fields=fields)

		self.data = self.data[features]

		self.data = self.data[self.data["Date"] >= from_date]
		
		self.dates = sorted(list(self.data["Date"].value_counts().index))

		self.X = self.data.drop(["IsOpen", "NumberOfSales"], inplace=False, axis = 1)
		self.y = self.data[["StoreID", "Date", "NumberOfSales"]]


	def one_hot_encode(self, fields):
		self.data = pd.get_dummies(self.data, columns=fields)

	def add_date_columns(self, date_column_name, year=False, month=False, day_n=False, day_of_week=False):
		if year:
			self.data["Year"] = self.data[date_column_name].dt.year
		if month:
			self.data["Month"] = self.data[date_column_name].dt.month
		if day_n:
			self.data["Day"] = self.data[date_column_name].dt.day
		if day_of_week :
			self.data["WeekDay"] = self.data[date_column_name].dt.dayofweek

	def split_attribute_list(self, column, attributes, fillna):
		mlb = MultiLabelBinarizer(classes=attributes)
		if fillna is not None:
			self.data[column] = self.data[column].fillna(fillna, inplace=False)
		self.data[column] = self.data[column].apply(lambda x: x.split('-'))
		new_columns_values = mlb.fit_transform(self.data[column].values.tolist())
		self.data[attributes] = pd.DataFrame(new_columns_values, index=self.data.index)

	def get_day_X_y(self, limit, region):
		for d in self.dates[:limit]:
			mask = (self.X["Date"] == d) & (self.X["Region_"+str(region)] == 1)
			X = self.X[mask].sort_values(["Date", "StoreID"], axis=0, ascending=True, inplace=False)
			y = self.y[mask].sort_values(["Date", "StoreID"], axis=0, ascending=True, inplace=False)
			yield X.drop(["Date","StoreID"], inplace=False, axis = 1), y.drop(["Date","StoreID"], inplace=False, axis = 1) , d
