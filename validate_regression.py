from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from regression_preprocessing import TrainPreprocessing, TestPreprocessing
import calendar
import pandas as pd
import numpy as np
import pickle

start_date = "2016-04"

scores = []

m = 'rf'

for dt in pd.date_range(start_date, periods=22, freq='M'):

	dt = dt.replace(day=1)

	dataset = pd.read_csv("train.csv")

	trp = TrainPreprocessing(dataset, to_date=str(dt.date()))

	X_train, y_train = trp.get_X_y()

	if m == 'knn':
		model = KNeighborsRegressor(n_neighbors=110, n_jobs=-1, p=2)
	elif m == 'rf':
		model = RandomForestRegressor(n_jobs=-1, n_estimators=100, bootstrap=True, random_state=1234, min_samples_leaf=5)

	model.fit(X_train, y_train)

	regions = [0,1,2,3,4,5,6,7,8,9,10]

	real = 0
	pred = 0

	naes = []

	tep = TestPreprocessing(dataset, from_date=str(dt.date()))

	for r in regions:
		
		month_real = None
		month_pred = None

		if dt.month != 12:
			n_days = calendar.monthrange(dt.year, dt.month)[1] + calendar.monthrange(dt.year, dt.month+1)[1]
		else:
			n_days = calendar.monthrange(dt.year, dt.month)[1] + calendar.monthrange(dt.year+1, 1)[1]

		for X_test, y_test, d in tep.get_day_X_y(limit=n_days, region=r):

			if len(X_test) == 0:
				continue

			y_pred = model.predict(np.array(X_test))

			targets = np.array(y_test).flatten()

			relevant_mask = targets != 0

			y_pred = np.around(y_pred)

			y_pred[~relevant_mask] = 0

			if month_real is None:
				month_real = targets.reshape(-1,1).astype(float)
				month_pred = y_pred.reshape(-1,1).astype(float)
			else:
				month_real += targets.reshape(-1,1).astype(float)
				month_pred += y_pred.reshape(-1,1).astype(float)

		if month_real is not None:
			real_month_total = np.sum(month_real)
			nae = np.sum(np.abs(month_real - month_pred))/real_month_total

			naes.append(nae)

			real += np.sum(month_real)
			pred += np.sum(month_pred)
		else:
			nae = 0

		print("nae:", nae, "month:", dt.month, "region:", r)

	score = np.sum(naes)/len(regions)
	scores.append(score)
	print("\nScore:", score, "total real:", real, "total pred:", pred, "month", dt.month, "\n")

print("Total score:", np.mean(scores))