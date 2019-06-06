import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import geopandas
import datetime
from scipy import stats
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
matplotlib.use("TkAgg")


def problem1(data, taxi_zones):
    data['index_num'] = np.arange(len(data))
    # The 2 lines of code below takes awhile to run
    data['pickup_time'] = data['tpep_pickup_datetime'].apply(
        lambda x: x.time())
    data['dropoff_time'] = data['tpep_dropoff_datetime'].apply(
        lambda x: x.time())

    # Creates the various time intervals for classifying the different times
    # of the day
    zero_hour = datetime.datetime.strptime('00:00:00', '%H:%M:%S').time()
    morning_6 = datetime.datetime.strptime('06:00:00', '%H:%M:%S').time()
    noon = datetime.datetime.strptime('12:00:00', '%H:%M:%S').time()
    afternoon_6 = datetime.datetime.strptime('18:00:00', '%H:%M:%S').time()
    midnight = datetime.datetime.strptime('23:59:59', '%H:%M:%S').time()

    # Split data into various partitions that represent the part of the day
    # where the taxi was taken
    early_morning = data[(data['pickup_time'] >= zero_hour) &
                         (data['pickup_time'] < morning_6)]
    morning = data[(data['pickup_time'] >= morning_6) &
                   (data['pickup_time'] < noon)]
    afternoon = data[(data['pickup_time'] >= noon) &
                     (data['pickup_time'] < afternoon_6)]
    night = data[(data['pickup_time'] >= afternoon_6) &
                 (data['pickup_time'] <= midnight)]

    # Joins the taxi trip table with the taxizone(geopandas) table for each
    # part of the day
    combine_early_morning = taxi_zones.merge(early_morning, how='inner',
                                             left_on='LocationID',
                                             right_on='PULocationID')
    combine_morning = taxi_zones.merge(morning, how='inner',
                                       left_on='LocationID',
                                       right_on='PULocationID')
    combine_afternoon = taxi_zones.merge(afternoon, how='inner',
                                         left_on='LocationID',
                                         right_on='PULocationID')
    combine_night = taxi_zones.merge(night, how='inner', left_on='LocationID',
                                     right_on='PULocationID')

    agg_earlymorning = combine_early_morning \
        .groupby('LocationID')['index_num'].count().reset_index()
    agg_morning = combine_morning.groupby('LocationID')['index_num'] \
        .count().reset_index()
    agg_afternoon = combine_afternoon.groupby('LocationID')['index_num'] \
        .count().reset_index()
    agg_night = combine_night.groupby('LocationID')['index_num'] \
        .count().reset_index()
    agg_earlymorning = taxi_zones.merge(agg_earlymorning, how='inner',
                                        left_on='LocationID',
                                        right_on='LocationID')
    agg_morning = taxi_zones.merge(agg_morning, how='inner',
                                   left_on='LocationID', right_on='LocationID')
    agg_afternoon = taxi_zones.merge(agg_afternoon, how='inner',
                                     left_on='LocationID',
                                     right_on='LocationID')
    agg_night = taxi_zones.merge(agg_night, how='inner', left_on='LocationID',
                                 right_on='LocationID')

    agg_earlymorning = agg_earlymorning[
        agg_earlymorning['borough'] == 'Manhattan']
    agg_morning = agg_morning[agg_morning['borough'] == 'Manhattan']
    agg_afternoon = agg_afternoon[agg_afternoon['borough'] == 'Manhattan']
    agg_night = agg_night[agg_night['borough'] == 'Manhattan']

    # Plots 4 maps of the same map of NewYorkCity(different times of the day)
    # made up of the various taxi-zones which are coloured based on the number
    # of taxi rides that started from that zone.
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, figsize=(20, 20), ncols=2)
    agg_earlymorning.plot(ax=ax1, column='index_num', legend=True)
    agg_morning.plot(ax=ax2, column='index_num', legend=True)
    agg_afternoon.plot(ax=ax3, column='index_num', legend=True)
    agg_night.plot(ax=ax4, column='index_num', legend=True)
    ax1.set_title('Early Morning 00:00-06:00')
    ax2.set_title('Morning 06:00-12:00')
    ax3.set_title('Afternoon 12:00-18:00')
    ax4.set_title('Night 18:00-24:00')

    fig.savefig('pickup_prevalence.png')
    plt.show()
    print("A visual response to this question has been saved to the " +
          "source directory as 'pickup_prevalence.png'")


def problem2(data, taxi_zones, lookup):
    """
    """
    zones = data.groupby('DOLocationID')['tip_amount'].mean().to_frame()
    taxi_DO = taxi_zones.merge(zones, right_on='DOLocationID',
                               left_on='OBJECTID', how='inner')
    taxi_DO.plot(column='tip_amount', figsize=(10, 10), legend=True)
    plt.title("Average Tip (in USD) Across Drop-Off Locations")
    top_8 = zones.merge(lookup, left_on='DOLocationID', right_on='LocationID',
                        how='right')
    top_8 = top_8.sort_values(by=['tip_amount'], ascending=False)
    top_8 = top_8.iloc[:8]
    top_8.head()


def problem3(data):
    afternoon_rush_hour1 = \
        datetime.datetime.strptime('16:00:00', '%H:%M:%S').time()
    afternoon_rush_hour2 = \
        datetime.datetime.strptime('20:00:00', '%H:%M:%S').time()

    data = data[['pickup_time', 'tip_amount']]
    data.loc[:, 'traffic?'] = 'non-rush-hour'
    data.loc[(data['pickup_time'] >= afternoon_rush_hour1) &
             (data['pickup_time'] <= afternoon_rush_hour2),
             'traffic?'] = 'rush-hour'
    data.loc[:, 'tip?'] = 'no'
    data.loc[data['tip_amount'] > 0.0, 'tip?'] = 'yes'

    contingency_table = pd.crosstab(data['traffic?'], data['tip?'],
                                    margins=True)
    f_obs = np.array([contingency_table.iloc[0][0:2].values,
                      contingency_table.iloc[1][0:2].values])

    # get p-value for chi-squared tests
    p_value = stats.chi2_contingency(f_obs)[1]

    if (p_value < 0.05):
        print("The null hypothesis is rejected, there is correlation " +
              "between driving through rush-hour traffic and " +
              "whether or not the driver gets a tip")
    else:
        print("The null hypothesis is accepted, whether or not the driver " +
              "gets a tip is independent of him driving through rush-hour " +
              "traffic")


def problem4(data):
    """
    """
    sub = data[(data['PULocationID'] == 1)]
    sub = sub.dropna()
    sub['trip_duration'] = sub['tpep_dropoff_datetime'] - \
        sub['tpep_pickup_datetime']
    sub['trip_duration'] = sub['trip_duration'].apply(
        lambda x: x.total_seconds())
    sub['trip_duration'] = sub['trip_duration'].apply(lambda x: float(x))
    sub['tpep_pickup_datetime'] = sub['tpep_pickup_datetime'].apply(
        lambda x: (float(x.hour) * 60) + float(x.minute))
    X = sub[['DOLocationID', 'tpep_pickup_datetime', 'trip_distance']]
    y = sub['trip_duration']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    # plot_tree(model, X, y)


def plot_tree(model, X, y):
    """
    """
    dot_data = StringIO()
    export_graphviz(model, out_file=dot_data,
                    feature_names=X.columns,
                    class_names=y.unique(),
                    filled=True, rounded=True,
                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('decision.png')
    Image(graph.create_png())


def main():
    # reads taxi records
    print('NYC Yellow Taxi Trip Analysis')
    print('Loading taxi trip records...')
    print('This may take awhile.')
    taxi_record = pd.read_csv('yellow_tripdata_2017-01.csv')
    # Convert datatime column from String to DateTime object
    taxi_record['tpep_pickup_datetime'] = \
        pd.to_datetime(taxi_record['tpep_pickup_datetime'])
    taxi_record['tpep_dropoff_datetime'] = \
        pd.to_datetime(taxi_record['tpep_dropoff_datetime'])

    # reads taxi zone lookup file
    print('Loading zone lookup table...')
    lookup = pd.read_csv('taxi+_zone_lookup.csv')

    # # reads taxi zones Geopandas data
    print('Loading geospatial data...')
    taxi_zones = geopandas.read_file('taxi_zones/taxi_zones.shp')

    print('Research Question 1:')
    print("Which ‘taxi zones’ are associated with the greatest prevalence " +
          "of taxi pickups?")
    problem1(taxi_record, taxi_zones)
    print('Research Question 2:')
    print("Which taxi zones are associated with the highest average tip " +
          "received?")
    problem2(taxi_record, taxi_zones, lookup)
    print('Research Question 3:')
    print("Is receiving a tip independent of rush hour traffic?")
    problem3(taxi_record)


if __name__ == '__main__':
    main()
