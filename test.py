import pandas as pd
from sklearn.metrics import mean_squared_error
import final_project
import geopandas


def main():
    # reads taxi records
    print('Loading taxi trip records...')
    print('This may take awhile.')
    data = pd.read_csv('yellow_tripdata_2017-01.csv')
    # reads taxi zone lookup file
    print('Loading zone lookup table...')
    lookup = pd.read_csv('taxi+_zone_lookup.csv')
    # reads taxi zone geometry file
    print('Loading geospatial data...')
    taxi_zones = geopandas.read_file('taxi_zones/taxi_zones.shp')
    # Convert datatime column from String to DateTime object
    data['tpep_pickup_datetime'] = pd.to_datetime(data['tpep_pickup_datetime'])
    data['tpep_dropoff_datetime'] = \
        pd.to_datetime(data['tpep_dropoff_datetime'])

    print('Testing Research Question 1')

    print('Testing Research Question 2')
    sub = data[(data['DOLocationID'] == 1) | (data['DOLocationID'] == 2) |
               (data['DOLocationID'] == 3)]
    final_project.problem2(sub, taxi_zones, lookup)

    print('Testing Research Question 3')

    print('Testing Research Question 4')
    model, X_train, X_test, y_train, y_test = final_project.problem4(data)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    print("Mean Squared Errors")
    print('Training:', mean_squared_error(y_train, y_train_pred))
    print('Testing:', mean_squared_error(y_test, y_test_pred))


if __name__ == '__main__':
    main()
