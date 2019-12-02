from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from tqdm import *
from regression_preprocessing import TrainPreprocessing, TestPreprocessing
import calendar
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

start_date = "2018-03"

scores = []

for dt in pd.date_range(start_date, periods=23, freq='M'):
	dt = dt.replace(day=1)
	dataset = pd.read_csv("train.csv")
	trp = TrainPreprocessing(dataset, to_date=str(dt.date()))
	
	X_train, y_train = trp.get_X_y()
	X_train, y_train = shuffle(X_train, y_train)

	split = int(len(X_train)*0.3)

	X_train, y_train = X_train[split:], y_train[split:]
	print('Base models train shape')
	print(X_train.shape, y_train.shape)
	
	X_stack, y_stack = X_train[:split], y_train[:split]
	print('Stack model train shape')
	print(X_stack.shape, y_stack.shape)

	model_1 = RandomForestRegressor(n_jobs=-1, n_estimators=50, bootstrap=True, random_state=1234)
	model_2 = AdaBoostRegressor(KNeighborsRegressor(n_neighbors=5, n_jobs=-1), random_state=1234, n_estimators=50)
	model_stack = KNeighborsRegressor(n_neighbors=110, n_jobs=-1)
	
	print("Fitting Model 1")

	model_1.fit(X_train, y_train)

	print("Fitting Model 2")

	model_2.fit(X_train, y_train)

	print("Predicting with base models")

	y1 = model_1.predict(X_stack)
	y2 = model_2.predict(X_stack)

	print("Fitting Model Stack")

	model_stack.fit(np.hstack([X_stack, y1.reshape(-1,1), y2.reshape(-1,1)]), y_stack)

	print("Predicting with Model Stack")
	regions = [0,1,2,3,4,5,6,7,8,9,10]

	real = 0
	pred = 0
	naes = []
	tep = TestPreprocessing(dataset, from_date=str(dt.date()))

	for r in regions:
		month_real = None
		month_pred = None
		n_days = calendar.monthrange(dt.year, dt.month)[1]
		count = 0
		new_X = []
		new_y = []
		for X_test, y_test, d in tqdm(tep.get_day_X_y(limit=n_days, region=r)):
			if len(X_test) == 0:
				continue
			y_pred_1 = model_1.predict(np.array(X_test))
			y_pred_2 = model_2.predict(np.array(X_test))
			targets = np.array(y_test).flatten()
			relevant_mask = targets != 0
			y_pred_1 = np.around(y_pred_1)
			y_pred_2 = np.around(y_pred_2)
			y_pred_1[~relevant_mask] = 0
			y_pred_2[~relevant_mask] = 0

			new = np.hstack([X_test, y_pred_1.reshape(-1,1), y_pred_2.reshape(-1,1)])
			y_pred = model_stack.predict(new)
			relevant_mask = targets != 0
			y_pred = np.around(y_pred)
			y_pred[~relevant_mask] = 0

			if month_real is None:
				month_real = targets.reshape(-1,1)
				month_pred = y_pred.reshape(-1,1)
			else:
				month_real += targets.reshape(-1,1)
				month_pred += y_pred.reshape(-1,1)

			count += 1

		if month_real is not None:
			real_month_total = np.sum(month_real)
			nae = np.sum(np.abs(month_real - month_pred))/real_month_total

			naes.append(nae)

			real += np.sum(month_real)
			pred += np.sum(month_pred)

		print("nae:", nae, "month:", dt.month, "region:", r)

	score = np.sum(naes)/len(regions)
	scores.append(score)
	print("\nScore:", score, "total real:", real, "total pred:", pred, "month", dt.month, "\n")

print("Total score:", np.mean(scores))


