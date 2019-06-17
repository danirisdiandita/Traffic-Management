# Traffic-Management
Predicting the traffic demand at certain location, day, and timestamp

The necessary classes are shown below


```python
# importing necessary libraries

import pandas as pd
import numpy as np
# import geohash
from itertools import combinations
# from sklearn.neighbors import NearestNeighbors
import time

# from sklearn.linear_model import Lasso
# from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse


class TrainTestPreprocessor:
    def __init__(self, training_df, test_df):#
        """
        the input would be two pandas dataframes
        with features:
        geohash6 : object
        day: int with the range from 
        timestamp
        """
        self.training_df = training_df
        self.test_df = test_df 
        self.X_train = None 
        self.X_test = None 
        self.y_train = None 
        self.y_test = None 

    def preprocessing(self):

        """ 
        input: 

        training_day = (default = None) a list consists of unique day for training_set. Example: [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
        test_day = (default = None) a list consists of unique day for test_df. Example: [15,16,17,18,19]
        training_hour = (default = None) a list consists of unique hour for training_df.


        Preprocessing consists of several steps that
        needs to be done

        
        1) feature engineering of original features:
            2. a) converting timestamp (str) into hour (int)
            and minute (int) under 
        2) filling missing demand data that is actually 0 in value
        
        """

        # step 1) converting timestamp values with new columns hour and minute in both training

        self.training_df["hour"] = self.training_df["timestamp"].apply(lambda x: int(x.split(":")[0]))
        self.training_df["minute"] = self.training_df["timestamp"].apply(lambda x: int(x.split(":")[1]))
        self.training_df = self.training_df[["geohash6", "day", "hour", "minute", "demand"]]

        self.test_df["hour"] = self.test_df["timestamp"].apply(lambda x: int(x.split(":")[0]))
        self.test_df["minute"] = self.test_df["timestamp"].apply(lambda x: int(x.split(":")[1]))
        self.test_df = self.test_df[["geohash6", "day", "hour", "minute", "demand"]]

        # step 2) filling the missing demand data that is actually 0 in value

        self.training_df['key'] = 0
        self.training_df = \
        self.training_df[["key","geohash6"]].drop_duplicates().merge(self.training_df[["key","day"]].drop_duplicates(), how = "outer").\
        merge(self.training_df[["key", "hour"]].drop_duplicates(), how = "outer").\
        merge(self.training_df[["key", "minute"]].drop_duplicates(), how = "outer").drop(columns = ["key"]).\
        merge(self.training_df.drop(columns = ["key"]), how = "left").fillna(0)

        self.test_df['key'] = 0
        self.test_df = \
        self.test_df[["key","geohash6"]].drop_duplicates().merge(self.test_df[["key","day"]].drop_duplicates(), how = "outer").\
        merge(self.test_df[["key", "hour"]].drop_duplicates(), how = "outer").\
        merge(self.test_df[["key", "minute"]].drop_duplicates(), how = "outer").drop(columns = ["key"]).\
        merge(self.test_df.drop(columns = ["key"]), how = "left").fillna(0)

        self.training_df["day_of_week"] = self.training_df["day"].apply(lambda x: 7 if x%7 == 0 else x%7)
        self.test_df["day_of_week"] = self.test_df["day"].apply(lambda x: 7 if x%7 == 0 else x%7)

        # step 3) adding feature coupling and applying mean encoding into it:
        # original features would be: geohash6, day_of_week, hour, minute
        # coupled features would be: geohash6-day_of_week, 

        features = ['geohash6', 'day_of_week', 'hour', 'minute']
        original_features = ["geohash6", "day", "hour", "minute", "day_of_week"]
        for i in range(1, len(features)): # +1
            for comb in combinations(features, i):
                temp_df = self.training_df.copy().rename(columns = {"demand" : "-".join(list(comb)) + "(encoded)"}).\
                groupby(list(comb), as_index = False)\
                ["-".join(list(comb)) + "(encoded)"].mean()

                self.training_df = self.training_df.merge(temp_df, how = "left")
                self.test_df = self.test_df.merge(temp_df, how = "left")
                del temp_df

        self.X_train = self.training_df.copy().drop(columns = ["demand"] + original_features)
        self.y_train = self.training_df.copy()["demand"]
        self.X_test = self.test_df.copy().drop(columns = ["demand"] + original_features)
        self.y_test = self.test_df.copy()["demand"]

class Model:
    def __init__(self):
        self.regressor = XGBRegressor(colsample_bytree= 0.4,
                            gamma= 0.21756976616440335,
                            min_child_weight= 10.0,
                            learning_rate= 0.05358746065589267,
                            max_depth= 4,
                            reg_alpha= 0.6870315939145919,
                            reg_lambda= 0.24937700167665464,
                            subsample= 0.845934912033431)

    def fit(self, X, y):
        self.regressor.fit(X, y)
    
    def predict(self, X):
        return self.regressor.predict(X)

    def rms(self, X_test,y_test):
        y_pred = self.regressor.predict(X)
        return np.sqrt(mse(y_test, y_pred))

                
class timeSeriesValidator:
    def __init__(self, dataset, minimum = 1, maximum = 60):
        self.minimum = minimum 
        self.maximum = maximum
        self.performance_results = {}
        self.dataset = dataset 

    def performance(self):

        start = time.time()

#         features = ['geohash6', 'day_of_week', 'hour', 'minute']
#         original_features = ["geohash6", "day", "hour", "minute", "day_of_week"]

        # defining the dataset copy

        datasetcopy = self.dataset.copy()
        datasetcopy.index = datasetcopy["day"]
        datasetcopy.index = datasetcopy.index.set_names(["id"])

        # with open("result002_features.txt", "a") as resultfile:
        for i in range(self.minimum, self.maximum + 1):
            training_index = [j for j in range(i, min(i + 14, self.maximum + 1))]
            test_index = [j for j in range(i+14, min(i+19, self.maximum + 1))]
            if len(test_index) == 5:
                print("{})with indexes given below".format(i))
                print("="*60)
                print("training = {}".format(training_index))
                print("test = {}".format(test_index))

                training_set = datasetcopy.loc[training_index,:]
                test_set = datasetcopy.loc[test_index,:]

                # standardising the day into day of week

                # training_set["day_of_week"] = training_set["day"].apply(lambda x: 7 if x%7 == 0 else x%7)
                # test_set["day_of_week"] = test_set["day"].apply(lambda x: 7 if x%7 == 0 else x%7)

                # for i in range(1, len(features)): # +1
                #     for comb in combinations(features, i):
                #         temp_df = training_set.copy().rename(columns = {"demand" : "-".join(list(comb)) + "(encoded)"}).\
                #         groupby(list(comb), as_index = False)\
                #         ["-".join(list(comb)) + "(encoded)"].mean()

                #         training_set = training_set.merge(temp_df, how = "left")
                #         test_set = test_set.merge(temp_df, how = "left")
                #         del temp_df

                # X_train = training_set.copy().drop(columns = ["demand"] + original_features)
                # y_train = training_set.copy()["demand"]
                # X_test = test_set.copy().drop(columns = ["demand"] + original_features)
                # y_test = test_set.copy()["demand"]

                train_test_preprocessor = TrainTestPreprocessor(training_set, test_set)
                train_test_preprocessor.preprocessing()
                X_train, X_test, y_train, y_test = train_test_preprocessor.X_train, train_test_preprocessor.X_test, \
                    train_test_preprocessor.y_train, train_test_preprocessor.y_test

        #         regressor = LinearRegression()
        #         regressor = Lasso(alpha = 0.000302)
        #         regressor.fit(X_train, y_train)
                # regressor = regressorModel(X = X_train.values, y = y_train.values.reshape(-1,1))
                model = Model()
                
                model.fit(X_train.values, y_train.values.reshape(-1,1))
#                 model.fit(X_train, y_train)
                y_pred = model.predict(X_test.values)
                r2test = r2_score(y_test.values.reshape(-1,1), y_pred)
                mse_test = mse(y_test.values.reshape(-1,1), y_pred)
                print("test r2_score = {}".format(r2test))
                print("mse test = {}".format(mse_test))
                y_train_pred = model.predict(X_train.values)
                r2train = r2_score(y_train.values.reshape(-1,1), y_train_pred)
                mse_train = mse(y_train.values.reshape(-1,1), y_train_pred)
                print("training r2_score = {}".format(r2train))
                print("mse train = {}".format(mse_train))
                print("="*60)
                
                if len(self.performance_results.get("mse_train", [])) == 0:
                    self.performance_results["mse_train"] = [mse_train]
                else:
                    self.performance_results["mse_train"].append(mse_train)
                    
                if len(self.performance_results.get("mse_test", [])) == 0:
                    self.performance_results["mse_test"] = [mse_test]
                else:
                    self.performance_results["mse_test"].append(mse_test)
                    
                if len(self.performance_results.get("r2train", [])) == 0:
                    self.performance_results["r2train"] = [r2train]
                else:
                    self.performance_results["r2train"].append(r2train)
                    
                if len(self.performance_results.get("r2test", [])) == 0:
                    self.performance_results["r2test"] = [r2test]
                else:
                    self.performance_results["r2test"].append(r2test)
                
        del datasetcopy
        end = time.time()
        print("time taken {} seconds".format(end - start))
        return self.performance_results

class SampleTestValidation:
    """
    description : input a pandas dataframe with 
    
    
    """
    def __init__(self, dataset):
        self.dataset = dataset
        
    def trainTestByDayPicked(self, training_day = list(range(42, 55 + 1)), test_day = list(range(56,60 + 1))):
        temp_df = self.dataset.copy()
        temp_df["index_"] = temp_df["day"]
        temp_df.index = temp_df["index_"]
        temp_df.drop(columns = ["index_"], inplace = True)
        
        training_df = temp_df.copy().loc[training_day,:]
        test_df = temp_df.copy().loc[test_day,:]
        
        del temp_df
        
        return training_df.reset_index(drop = True), test_df.reset_index(drop = True)
        
```

### Part 1 - Validation

The model using the best tuned features and hyperparameters, and Time Series Validation using Training Data is shown below - Try to run 

Reading the dataset in CSV file 


```python
dataset = pd.read_csv("training.csv")
```


```python
score = timeSeriesValidator(dataset = dataset, minimum = 1, maximum = 60)
```


```python
result_performance = score.performance()
```

    1)with indexes given below
    ============================================================
    training = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    test = [15, 16, 17, 18, 19]
    test r2_score = 0.836441150124991
    mse test = 0.0026662764614753982
    training r2_score = 0.9380071236889248
    mse train = 0.0009506698110609654
    ============================================================
    2)with indexes given below
    ============================================================
    training = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    test = [16, 17, 18, 19, 20]
    test r2_score = 0.8263859402958267
    mse test = 0.002575917369383597
    training r2_score = 0.9439506836700813
    mse train = 0.0008740689635174653
    ============================================================
    3)with indexes given below
    ============================================================
    training = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    test = [17, 18, 19, 20, 21]
    test r2_score = 0.8224765779737464
    mse test = 0.0024324749045046257
    training r2_score = 0.9426059533344497
    mse train = 0.0009012832533783139
    ============================================================
    4)with indexes given below
    ============================================================
    training = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    test = [18, 19, 20, 21, 22]
    test r2_score = 0.8101727766927032
    mse test = 0.002453279849600445
    training r2_score = 0.9421053906602095
    mse train = 0.000905968291901865
    ============================================================
    5)with indexes given below
    ============================================================
    training = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    test = [19, 20, 21, 22, 23]
    test r2_score = 0.9202647020133389
    mse test = 0.0010625633064296727
    training r2_score = 0.9288788178668195
    mse train = 0.0010746103937389557
    ============================================================
    6)with indexes given below
    ============================================================
    training = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    test = [20, 21, 22, 23, 24]
    test r2_score = 0.9213317228852784
    mse test = 0.0011303571211801488
    training r2_score = 0.9289899953643159
    mse train = 0.0010651280811399399
    ============================================================
    7)with indexes given below
    ============================================================
    training = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    test = [21, 22, 23, 24, 25]
    test r2_score = 0.9097537158836329
    mse test = 0.0014812847447421254
    training r2_score = 0.9272531386966847
    mse train = 0.001081775906986286
    ============================================================
    8)with indexes given below
    ============================================================
    training = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    test = [22, 23, 24, 25, 26]
    test r2_score = 0.9097356297008917
    mse test = 0.001459835675193732
    training r2_score = 0.9274455541136085
    mse train = 0.0010635609890196516
    ============================================================
    9)with indexes given below
    ============================================================
    training = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    test = [23, 24, 25, 26, 27]
    test r2_score = 0.909895736518607
    mse test = 0.0013901722904219424
    training r2_score = 0.9264265536782126
    mse train = 0.0010563011231797736
    ============================================================
    10)with indexes given below
    ============================================================
    training = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    test = [24, 25, 26, 27, 28]
    test r2_score = 0.9125909936152726
    mse test = 0.0013226149981246098
    training r2_score = 0.9337723842016906
    mse train = 0.0009692093118721773
    ============================================================
    11)with indexes given below
    ============================================================
    training = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    test = [25, 26, 27, 28, 29]
    test r2_score = 0.9031651495552621
    mse test = 0.0013783028863133045
    training r2_score = 0.9379416657802129
    mse train = 0.0009200290157190631
    ============================================================
    12)with indexes given below
    ============================================================
    training = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    test = [26, 27, 28, 29, 30]
    test r2_score = 0.9258191264775706
    mse test = 0.001020258160598212
    training r2_score = 0.9382974851773289
    mse train = 0.0009209920954466741
    ============================================================
    13)with indexes given below
    ============================================================
    training = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
    test = [27, 28, 29, 30, 31]
    test r2_score = 0.9293740871735074
    mse test = 0.0010678691153868823
    training r2_score = 0.9383913109888005
    mse train = 0.0009113967141713752
    ============================================================
    14)with indexes given below
    ============================================================
    training = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
    test = [28, 29, 30, 31, 32]
    test r2_score = 0.9148710066302074
    mse test = 0.0014915476933866079
    training r2_score = 0.9389646191767576
    mse train = 0.0009014642553272064
    ============================================================
    15)with indexes given below
    ============================================================
    training = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
    test = [29, 30, 31, 32, 33]
    test r2_score = 0.9121080750416536
    mse test = 0.0015211506631329003
    training r2_score = 0.9433166859721307
    mse train = 0.0008451937553999132
    ============================================================
    16)with indexes given below
    ============================================================
    training = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    test = [30, 31, 32, 33, 34]
    test r2_score = 0.9165953857445548
    mse test = 0.001427288380464181
    training r2_score = 0.9418185762939069
    mse train = 0.0008500034266249904
    ============================================================
    17)with indexes given below
    ============================================================
    training = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    test = [31, 32, 33, 34, 35]
    test r2_score = 0.9125353854995347
    mse test = 0.0014778576086329023
    training r2_score = 0.9445605958725846
    mse train = 0.0008029271463231863
    ============================================================
    18)with indexes given below
    ============================================================
    training = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    test = [32, 33, 34, 35, 36]
    test r2_score = 0.9028139897476887
    mse test = 0.0016121546022183806
    training r2_score = 0.9418566382280641
    mse train = 0.0008497243088968654
    ============================================================
    19)with indexes given below
    ============================================================
    training = [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    test = [33, 34, 35, 36, 37]
    test r2_score = 0.920354923148378
    mse test = 0.0012572318480429904
    training r2_score = 0.9631796168118701
    mse train = 0.0005589944964490997
    ============================================================
    20)with indexes given below
    ============================================================
    training = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
    test = [34, 35, 36, 37, 38]
    test r2_score = 0.925406278345847
    mse test = 0.001244577359225774
    training r2_score = 0.9623883737655898
    mse train = 0.0005749993627693068
    ============================================================
    21)with indexes given below
    ============================================================
    training = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
    test = [35, 36, 37, 38, 39]
    test r2_score = 0.926249086666925
    mse test = 0.0014362328504959078
    training r2_score = 0.9624100463340987
    mse train = 0.0005800702146645715
    ============================================================
    22)with indexes given below
    ============================================================
    training = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
    test = [36, 37, 38, 39, 40]
    test r2_score = 0.9193663362506614
    mse test = 0.0015861590731810849
    training r2_score = 0.9584662789392465
    mse train = 0.0006490961555089359
    ============================================================
    23)with indexes given below
    ============================================================
    training = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
    test = [37, 38, 39, 40, 41]
    test r2_score = 0.9216879981140038
    mse test = 0.0014516661810515033
    training r2_score = 0.9585803215702043
    mse train = 0.0006586669708066569
    ============================================================
    24)with indexes given below
    ============================================================
    training = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]
    test = [38, 39, 40, 41, 42]
    test r2_score = 0.9169773524933547
    mse test = 0.0015608492951038608
    training r2_score = 0.9555309268200347
    mse train = 0.0007138889900611478
    ============================================================
    25)with indexes given below
    ============================================================
    training = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
    test = [39, 40, 41, 42, 43]
    test r2_score = 0.8992528331965055
    mse test = 0.0018344284439059545
    training r2_score = 0.9579071264117461
    mse train = 0.0006790636177567926
    ============================================================
    26)with indexes given below
    ============================================================
    training = [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    test = [40, 41, 42, 43, 44]
    test r2_score = 0.8917571236450446
    mse test = 0.0018869935031835693
    training r2_score = 0.9573867715839854
    mse train = 0.0007047042467069088
    ============================================================
    27)with indexes given below
    ============================================================
    training = [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
    test = [41, 42, 43, 44, 45]
    test r2_score = 0.888040447426101
    mse test = 0.0020574192431525506
    training r2_score = 0.956695152354658
    mse train = 0.0007298020975963332
    ============================================================
    28)with indexes given below
    ============================================================
    training = [28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
    test = [42, 43, 44, 45, 46]
    test r2_score = 0.8559713864938666
    mse test = 0.0027281441185587193
    training r2_score = 0.9562624764903527
    mse train = 0.0007420342658194609
    ============================================================
    29)with indexes given below
    ============================================================
    training = [29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]
    test = [43, 44, 45, 46, 47]
    test r2_score = 0.8451902754063243
    mse test = 0.0027741393231732457
    training r2_score = 0.9552890424133861
    mse train = 0.0007735521941092869
    ============================================================
    30)with indexes given below
    ============================================================
    training = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]
    test = [44, 45, 46, 47, 48]
    test r2_score = 0.8480210338773757
    mse test = 0.0026652243251671943
    training r2_score = 0.9525957401013541
    mse train = 0.0008287671299531561
    ============================================================
    31)with indexes given below
    ============================================================
    training = [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]
    test = [45, 46, 47, 48, 49]
    test r2_score = 0.8423967434004659
    mse test = 0.002608620064905856
    training r2_score = 0.9523229477284373
    mse train = 0.000848952101130185
    ============================================================
    32)with indexes given below
    ============================================================
    training = [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]
    test = [46, 47, 48, 49, 50]
    test r2_score = 0.8339313146475884
    mse test = 0.002657967077533931
    training r2_score = 0.9481413372962169
    mse train = 0.000933681237759586
    ============================================================
    33)with indexes given below
    ============================================================
    training = [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
    test = [47, 48, 49, 50, 51]
    test r2_score = 0.8718756807174993
    mse test = 0.0022086516222346313
    training r2_score = 0.9408502156299964
    mse train = 0.0010321145048844467
    ============================================================
    34)with indexes given below
    ============================================================
    training = [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
    test = [48, 49, 50, 51, 52]
    test r2_score = 0.8647379508303422
    mse test = 0.0026243445794693165
    training r2_score = 0.9366498541382655
    mse train = 0.001106861282978842
    ============================================================
    35)with indexes given below
    ============================================================
    training = [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]
    test = [49, 50, 51, 52, 53]
    test r2_score = 0.8726241304096723
    mse test = 0.0027427008832977455
    training r2_score = 0.9326967929524267
    mse train = 0.001184223287900488
    ============================================================
    36)with indexes given below
    ============================================================
    training = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
    test = [50, 51, 52, 53, 54]
    test r2_score = 0.8728373982274173
    mse test = 0.0027106501833663566
    training r2_score = 0.9338633959510012
    mse train = 0.0011666321823246603
    ============================================================
    37)with indexes given below
    ============================================================
    training = [37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    test = [51, 52, 53, 54, 55]
    test r2_score = 0.8823242204667701
    mse test = 0.0023324090040413785
    training r2_score = 0.9314381380440604
    mse train = 0.0012175318165686324
    ============================================================
    38)with indexes given below
    ============================================================
    training = [38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    test = [52, 53, 54, 55, 56]
    test r2_score = 0.8873665637978865
    mse test = 0.0021923230160454795
    training r2_score = 0.9320390444606741
    mse train = 0.0012209814400994073
    ============================================================
    39)with indexes given below
    ============================================================
    training = [39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]
    test = [53, 54, 55, 56, 57]
    test r2_score = 0.9095295759670093
    mse test = 0.0016363682128148376
    training r2_score = 0.9275548565503444
    mse train = 0.0013360412997304395
    ============================================================
    40)with indexes given below
    ============================================================
    training = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]
    test = [54, 55, 56, 57, 58]
    test r2_score = 0.9106921359096887
    mse test = 0.0015135455575075649
    training r2_score = 0.9307540217767752
    mse train = 0.0012638689313347012
    ============================================================
    41)with indexes given below
    ============================================================
    training = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
    test = [55, 56, 57, 58, 59]
    test r2_score = 0.9092141556691673
    mse test = 0.0015848527755931037
    training r2_score = 0.9326123945234261
    mse train = 0.0012262569476797318
    ============================================================
    42)with indexes given below
    ============================================================
    training = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]
    test = [56, 57, 58, 59, 60]
    test r2_score = 0.9115567191405599
    mse test = 0.0017616985653184241
    training r2_score = 0.934319720887253
    mse train = 0.001197181714527209
    ============================================================
    time taken 3567.7408468723297 seconds



```python
average_rmse_train = np.mean([np.sqrt(ms_error) for ms_error in result_performance["mse_train"]])
average_rmse_test = np.mean([np.sqrt(ms_error) for ms_error in result_performance["mse_test"]])
print("the train dataset's average root mean square for all of the time series train-testing is {}".format(average_rmse_train))
print("the test dataset's average root mean square for all of the time series train-testing is {}".format(average_rmse_test))
```

    the train dataset's average root mean square for all of the time series train-testing is 0.030236219003556015
    the test dataset's average root mean square for all of the time series train-testing is 0.0426028202358991


### Part 2 - How to Use the Model for your Testing - Validation Process

    from the Validation Above, it is already generated the training and testing data, 
    we can convert the training and the testing data for simulating the testing and validation data
    
    like below:
    
    X_test, y_test = X_train, y_train
    X_validation, y_validation = X_test, y_test

    ===================================================================================
    I prepared a testing and validation dataframe


```python
test_val = SampleTestValidation(dataset = dataset)
testing_day = list(range(42, 55 + 1))
validation_day = list(range(56,60 + 1))
testing_df, validation_df = test_val.trainTestByDayPicked(training_day = testing_day,\
                                                             test_day = validation_day)
```

    showing below the mean encoded X_test and y_test


```python
# original dataset for comparison

dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>geohash6</th>
      <th>day</th>
      <th>timestamp</th>
      <th>demand</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>qp03wc</td>
      <td>18</td>
      <td>20:0</td>
      <td>0.020072</td>
    </tr>
    <tr>
      <th>1</th>
      <td>qp03pn</td>
      <td>10</td>
      <td>14:30</td>
      <td>0.024721</td>
    </tr>
    <tr>
      <th>2</th>
      <td>qp09sw</td>
      <td>9</td>
      <td>6:15</td>
      <td>0.102821</td>
    </tr>
    <tr>
      <th>3</th>
      <td>qp0991</td>
      <td>32</td>
      <td>5:0</td>
      <td>0.088755</td>
    </tr>
    <tr>
      <th>4</th>
      <td>qp090q</td>
      <td>15</td>
      <td>4:0</td>
      <td>0.074468</td>
    </tr>
  </tbody>
</table>
</div>




```python
# testing_df consists of data for given testing_day, I choose day 42 to 55 (inclusive)
testing_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>geohash6</th>
      <th>day</th>
      <th>timestamp</th>
      <th>demand</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>qp03m3</td>
      <td>42</td>
      <td>12:45</td>
      <td>0.012462</td>
    </tr>
    <tr>
      <th>1</th>
      <td>qp03zn</td>
      <td>42</td>
      <td>5:45</td>
      <td>0.049627</td>
    </tr>
    <tr>
      <th>2</th>
      <td>qp03z5</td>
      <td>42</td>
      <td>15:30</td>
      <td>0.010035</td>
    </tr>
    <tr>
      <th>3</th>
      <td>qp03nz</td>
      <td>42</td>
      <td>5:15</td>
      <td>0.221294</td>
    </tr>
    <tr>
      <th>4</th>
      <td>qp03xd</td>
      <td>42</td>
      <td>16:45</td>
      <td>0.020314</td>
    </tr>
  </tbody>
</table>
</div>




```python
# validation_df consists of data for given validation_day, I choose day 56 to 60(inclusive)
validation_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>geohash6</th>
      <th>day</th>
      <th>timestamp</th>
      <th>demand</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>qp091w</td>
      <td>56</td>
      <td>10:0</td>
      <td>0.008772</td>
    </tr>
    <tr>
      <th>1</th>
      <td>qp09de</td>
      <td>56</td>
      <td>15:30</td>
      <td>0.090775</td>
    </tr>
    <tr>
      <th>2</th>
      <td>qp09gy</td>
      <td>56</td>
      <td>14:30</td>
      <td>0.035722</td>
    </tr>
    <tr>
      <th>3</th>
      <td>qp0901</td>
      <td>56</td>
      <td>13:15</td>
      <td>0.055648</td>
    </tr>
    <tr>
      <th>4</th>
      <td>qp09gy</td>
      <td>56</td>
      <td>11:15</td>
      <td>0.062912</td>
    </tr>
  </tbody>
</table>
</div>



    Applying Preprocessing to Our Data


```python
# train test preprocessing
        
train_test_preprocessor = TrainTestPreprocessor(training_df = testing_df, test_df = validation_df)
train_test_preprocessor.preprocessing()

X_test = train_test_preprocessor.X_train
y_test = train_test_preprocessor.y_train
X_val = train_test_preprocessor.X_test
y_val = train_test_preprocessor.y_test
```

    Applying Machine Learning Algorithm


```python
regressor = Model()
regressor.fit(X_test, y_test.values.reshape(-1,1))
y_pred = regressor.predict(X_val)
print("root mean square = {}".format(np.sqrt(mse(y_val, y_pred))))
```

    root mean square = 0.041972593025907085


### Part 3 - Your Own Dataset


```python

```

