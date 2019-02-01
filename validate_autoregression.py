from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from autoregression_preprocessing import TrainPreprocessing, TestPreprocessing
import calendar
import pandas as pd
import numpy as np

start_date = "2016-06"

m = 'rf'

for dt in pd.date_range(start_date, periods=22, freq='M'):

	dataset = pd.read_csv("train.csv")

	dt = dt.replace(day=1)

	trp = TrainPreprocessing(dataset, to_date=str(dt.date()))

	X_train, y_train = trp.get_X_y()

	if m == 'knn':
		model = KNeighborsRegressor(n_neighbors=110, n_jobs=-1, p=2)
	elif m == 'rf':
		model = RandomForestRegressor(n_jobs=-1, n_estimators=100, bootstrap=True, random_state=1234, min_samples_leaf=5)

	model.fit(X_train, y_train)

	regions = [0,1,3,4,5,6,7,8,9,10]

	real = 0
	pred = 0

	naes = []

	for r in regions:

		tep = TestPreprocessing(dataset, from_date=str(dt.date()), region=r)
		
		month_real = np.zeros(shape=(tep.store_count, 1), dtype=float)
		month_pred = np.zeros(shape=(tep.store_count, 1), dtype=float)

		if dt.month != 12:
			n_days = calendar.monthrange(dt.year, dt.month)[1] + calendar.monthrange(dt.year, dt.month+1)[1]
		else:
			n_days = calendar.monthrange(dt.year, dt.month)[1] + calendar.monthrange(dt.year+1, 1)[1]

		for X_test, y_test, _ in tep.get_day_X_y(limit=n_days):

			y_pred = model.predict(np.array(X_test))
			targets = np.array(y_test).flatten()

			relevant_mask = targets != 0

			y_pred = np.around(y_pred)

			y_pred[~relevant_mask] = 0

			tep.add_to_history(y_pred)

			month_real += targets.reshape(-1,1)
			month_pred += y_pred.reshape(-1,1)

		real_month_total = np.sum(month_real)
		nae = np.sum(np.abs(month_real - month_pred))/real_month_total

		if real_month_total == 0.:
			nae = 0.

		naes.append(nae)

		real += np.sum(month_real)
		pred += np.sum(month_pred)

		print("nae:", nae, "month:", dt.month, "region:", r)

	print("final score ", np.sum(naes)/len(regions), "total real:", real, "total pred:", pred, "month", dt.month)
	print()