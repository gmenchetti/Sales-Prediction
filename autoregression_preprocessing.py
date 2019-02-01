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

class TrainPreprocessing():

	def __init__(self, data, to_date):

		self.data = copy.deepcopy(data)
		self.win_size = 60

		event_list = ['Rain', 'Fog', 'NoEvent', 'Snow', 'Thunderstorm', 'Hail']
		fields = ["StoreType", "WeekDay", "Region", "Month", "AssortmentType"]
		features = [
			'StoreID', 'Date', 'IsOpen', 'IsHoliday', 'HasPromotions', 'NumberOfSales',
			'StoreType_Hyper Market','StoreType_Shopping Center', 'StoreType_Standard Market',
			'StoreType_Super Market','NearestCompetitor',
			'WeekDay_0', 'WeekDay_1', 'WeekDay_2','WeekDay_3',
			'WeekDay_4', 'WeekDay_5', 'WeekDay_6', 'Region_0',
			'Region_1', 'Region_2', 'Region_3', 'Region_4', 'Region_5',
			'Region_6','Region_7', 'Region_8', 'Region_9', 'Region_10',
			'Month_1', 'Month_2','Month_3', 'Month_4', 'Month_5', 'Month_6',
			'Month_7', 'Month_8','Month_9', 'Month_10', 'Month_11', 'Month_12',
			'AssortmentType_General','AssortmentType_With Fish Department', 'AssortmentType_With Non-Food Department']

		self.data["Date"] = pd.to_datetime(self.data['Date'], format='%d/%m/%Y')

		self.add_date_columns(date_column_name="Date", month=True, day_of_week=True)

		self.split_attribute_list(column="Events", attributes=event_list, fillna="NoEvent")

		self.one_hot_encode(fields=fields)

		self.data = self.data[features]

		self.data = self.data[self.data["Region_2"] != 1]

		self.data = self.data[self.data["Date"] < to_date]

		self.add_history_features(lags=[7, 14], mean=True)

		self.data = self.data[self.data["IsOpen"] == 1]

		self.X = np.array(self.data.drop(["StoreID", "IsOpen", "Date", "NumberOfSales"], inplace=False, axis = 1))
		self.y = np.array(self.data["NumberOfSales"])

	def get_X_y(self):
		return self.X, self.y

	def one_hot_encode(self, fields):
		self.data = pd.get_dummies(self.data, columns=fields)

	def interpolate_closed_days(self):
		self.data['NumberOfSales'] = self.data['NumberOfSales'].replace({0: np.nan})
		self.data['NumberOfSales'] = self.data.groupby('StoreID')["NumberOfSales"].apply(lambda x: x.interpolate())
		self.data['NumberOfSales'] = self.data['NumberOfSales'].replace({np.nan: 0})

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

	def add_history_features(self, lags, mean=True):

		tmp = copy.deepcopy(self.data)
		
		if mean:
			self.data['mean'] = tmp.sort_values(["Date"], axis=0, ascending=True, inplace=False) \
								.groupby('StoreID')[['StoreID','NumberOfSales']] \
								.rolling(self.win_size).mean().reset_index(0,drop=True).groupby("StoreID")['NumberOfSales'].shift(1)

		for l in lags:
			self.data['lag_'+str(l)] = tmp.sort_values(["Date", "StoreID"], axis=0, ascending=True, inplace=False) \
									  .groupby('StoreID')['NumberOfSales'].shift(l)

		self.data = self.data.dropna()


class TestPreprocessing():

	def __init__(self, data, from_date, region):

		self.data = copy.deepcopy(data)
		self.win_size = 60

		event_list = ['Rain', 'Fog', 'NoEvent', 'Snow', 'Thunderstorm', 'Hail']
		fields = ["StoreType", "WeekDay", "Region", "Month", "AssortmentType"]
		features = [
			'StoreID', 'Date', 'IsOpen', 'IsHoliday', 'HasPromotions', 'NumberOfSales',
			'StoreType_Hyper Market','StoreType_Shopping Center', 'StoreType_Standard Market',
			'StoreType_Super Market','NearestCompetitor',
			'WeekDay_0', 'WeekDay_1', 'WeekDay_2','WeekDay_3',
			'WeekDay_4', 'WeekDay_5', 'WeekDay_6', 'Region_0',
			'Region_1', 'Region_2', 'Region_3', 'Region_4', 'Region_5',
			'Region_6','Region_7', 'Region_8', 'Region_9', 'Region_10',
			'Month_1', 'Month_2','Month_3', 'Month_4', 'Month_5', 'Month_6',
			'Month_7', 'Month_8','Month_9', 'Month_10', 'Month_11', 'Month_12',
			'AssortmentType_General','AssortmentType_With Fish Department', 'AssortmentType_With Non-Food Department']

		self.data["Date"] = pd.to_datetime(self.data['Date'], format='%d/%m/%Y')

		n_days_ago = datetime.strptime(from_date, "%Y-%m-%d") - timedelta(days=self.win_size)

		self.add_date_columns(date_column_name="Date", month=True, day_of_week=True)

		self.split_attribute_list(column="Events", attributes=event_list, fillna="NoEvent")

		self.one_hot_encode(fields=fields)

		self.data = self.data[self.data["Region_"+str(region)] == 1]

		self.history = self.data[(self.data["Date"] < from_date) & (self.data["Date"] >= n_days_ago)]

		self.history["NumberOfSales"] = self.history["NumberOfSales"]

		self.history = self.history.sort_values(["Date", "StoreID"], axis=0, ascending=True, inplace=False)

		self.history = np.vstack(self.history.groupby("Date")["NumberOfSales"].apply(lambda x : np.array(x)))
		
		self.data = self.data[features]

		self.store_count = len(self.data[self.data["Date"] == from_date]["StoreID"].value_counts())

		self.data = self.data[self.data["Date"] >= from_date]
		
		self.dates = sorted(list(self.data["Date"].value_counts().index))

		self.X = self.data.drop(["IsOpen", "NumberOfSales"], inplace=False, axis = 1)
		self.y = self.data[["StoreID", "Date", "NumberOfSales"]]


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

	def get_day_X_y(self, limit):
		for d in self.dates[:limit]:
			X = self.X[self.X["Date"] == d].sort_values(["Date", "StoreID"], axis=0, ascending=True, inplace=False)
			y = self.y[self.y["Date"] == d].sort_values(["Date", "StoreID"], axis=0, ascending=True, inplace=False)
			X["mean"], X["lag_7"], X["lag_14"] = self.add_history_features()
			yield X.drop(["Date","StoreID"], inplace=False, axis = 1), y.drop(["Date","StoreID"], inplace=False, axis = 1) , d

	def add_history_features(self):
		mean = np.mean(self.history, axis=0)
		lag_7 = self.history[-7]
		lag_14 = self.history[-14]
		return mean, lag_7, lag_14

	def add_to_history(self, new):
		self.history[:self.win_size-1, :] = self.history[1:self.win_size, :]
		self.history[self.win_size-1] = new


