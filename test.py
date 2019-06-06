import pandas as pd
from sklearn.metrics import mean_squared_error
import final_project


def main():
    data = pd.read_csv('yellow_tripdata_2017-01.csv')
    print('Testing Research Question 1')

    print('Testing Research Question 2')
    sub = data[(data['PULocationID'] == 1) | (data['PULocationID'] == 2) |
               (data['PULocationID'] == 139)]
    final_project.problem2(sub)

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
