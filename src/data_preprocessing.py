import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import OneHotEncoder
from scipy import stats
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler


class ScoreDataPreprocessor:
    
    def __init__(self):
        pass
    
    def fit(self, X=None, y=None):
        return self
    
    
    
    def load_data():
        conn = sqlite3.connect('data/score.db')
        query = "SELECT * FROM score"

        df = pd.read_sql(query, conn)
        conn.close()
        return df




    def missing_values(df):

        df = df.drop(['index', 'student_id', 'bag_color'], axis=1)

        missing_values = df.isnull().sum()
        missing_percentage = 100 * df.isnull().sum() / len(df)
        missing_table = pd.concat([missing_values, missing_percentage], axis=1, keys=['Missing Count', 'Missing Percentage'])

        # calculating the median
        final_test_median = df['final_test'].median()
        attendance_rate_median = df['attendance_rate'].median()

        df['final_test'] = df['final_test'].fillna(final_test_median)
        df['attendance_rate'] = df['attendance_rate'].fillna(attendance_rate_median)
        return df




    def feature_engineering(df):
        # 3) FEATURE CREATION

        # converting object dtype (normally stored as strings) > datetime
        df['sleep_time'] = pd.to_datetime(df['sleep_time'], format='%H:%M').dt.time
        df['wake_time'] = pd.to_datetime(df['wake_time'], format='%H:%M').dt.time

        # creating function to calculate sleep duration
        def calculate_sleep_duration(sleep_time, wake_time):
            sleep_datetime = datetime.combine(datetime.min, sleep_time)
            wake_datetime = datetime.combine(datetime.min, wake_time)

            if sleep_time > wake_time:
                wake_datetime += timedelta(days=1)

            duration = wake_datetime - sleep_datetime
            return duration.total_seconds() / 3600 
        # 3600 in an hour (60mins * 60secs)

        df['sleep_duration'] = df.apply(lambda row: calculate_sleep_duration(row['sleep_time'], row['wake_time']), axis=1)
        return df





    def encoding(df):
        # 4) ENCODING 

        # standardizing CCA column
        def standardize_column(column):
            column = column.str.lower()
            unique_values = column.unique()
            value_map = {value: value for value in unique_values}
            return column.map(value_map)

        df['CCA'] = standardize_column(df['CCA'])

        # standardizing tuition column
        def standardize_tuition(column):
            column = column.str.lower()

            value_map = {
                'yes': 'yes',
                'y': 'yes', 
                'no': 'no',
                'n': 'no'
            }
            
            return column.map(value_map)

        df['tuition'] = standardize_tuition(df['tuition'])



        columns_to_encode = ['direct_admission', 'CCA', 'learning_style', 'gender', 'tuition', 'mode_of_transport']

        # initializing the onehotencoder
        one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        # fit and transform the selected columns
        encoded_columns = one_hot_encoder.fit_transform(df[columns_to_encode])

        # getting feature names
        feature_names = one_hot_encoder.get_feature_names_out(columns_to_encode)

        # creating new df with encoded columns, combining it with original df
        encoded_df = pd.DataFrame(encoded_columns, columns=feature_names, index=df.index)
        df = pd.concat([df.drop(columns=columns_to_encode), encoded_df], axis=1)


        # converting sleep_time, wake_time to datetime
        df['sleep_time'] = pd.to_datetime(df['sleep_time'], format='%H:%M:%S')
        df['wake_time'] = pd.to_datetime(df['wake_time'], format='%H:%M:%S')

        # Hour of day (cyclical encoding)
        # good for lin reg, which assumes linear relationships, also tree-based models like RF, GBM
        df['sleep_hour_sin'] = np.sin(df['sleep_time'].dt.hour * (2 * np.pi / 24))
        df['sleep_hour_cos'] = np.cos(df['sleep_time'].dt.hour * (2 * np.pi / 24))
        df['wake_hour_sin'] = np.sin(df['wake_time'].dt.hour * (2 * np.pi / 24))
        df['wake_hour_cos'] = np.cos(df['wake_time'].dt.hour * (2 * np.pi / 24))

        df = df.drop(['sleep_time', 'wake_time'], axis=1)
        
        return df




    def outliers(df):
        # 5) OUTLIERS

        # outliers
        numerical_columns = ['number_of_siblings', 'final_test', 'n_male', 'n_female', 'age', 
                            'hours_per_week', 'attendance_rate', 'sleep_duration']

        def detect_outliers_zscore(data, threshold=3):
            z_scores = np.abs(stats.zscore(data))
            return z_scores > threshold

        def detect_outliers_iqr(data):
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 - 1.5 * IQR
            return (data < lower_bound) | (data > upper_bound)

        outliers_zscore = {}
        for col in numerical_columns:
            outliers_zscore[col] = detect_outliers_zscore(df[col])

        outliers_iqr = {}
        for col in numerical_columns:
            outliers_iqr[col] = detect_outliers_iqr(df[col])
            


        # remove 6, 5, -4, -5 since its not possible for secondary school 
        mode_age = df['age'].mode().iloc[0]
        df['age'] = df['age'].replace([6, 5, -4, -5], mode_age)

        return df




    def scaling(df):
        # 6) SCALING

        columns_to_scale = ['final_test', 'n_male', 'n_female', 'age', 'hours_per_week', 
                            'attendance_rate', 'number_of_siblings', 'sleep_duration']

        scaler = StandardScaler()
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
        return df




    def correlation_analysis(df):
        # 7) CORRELATION ANALYSIS

        # select all columns except for 'final_test'
        feature_columns = [col for col in df.columns if col != 'final_test']

        # calculate correlations with 'final_test'
        correlations_finaltest = df[feature_columns].corrwith(df['final_test'])

        # sort correlations by absolute value
        correlations_sorted = correlations_finaltest.abs().sort_values(ascending=False)


        corr_matrix = df.corr()

                        
        columns_to_drop = ['direct_admission_No', 'learning_style_Auditory', 'gender_Male', 
                        'tuition_no', 'wake_hour_sin']

        df.drop(columns=columns_to_drop, inplace=True)    
        return df           


    def main():
        df = load_data()
        print("Data loaded. Shape:", df.shape)
        df = missing_values(df)
        print("Missing values handled. Shape:", df.shape)
        df = feature_engineering(df)
        print("Feature engineering complete. Shape:", df.shape)
        df = encoding(df)
        print("Encoding complete. Shape:", df.shape)
        df = outliers(df)
        print("Outliers handled. Shape:", df.shape)
        df = scaling(df)
        print("Scaling complete. Shape:", df.shape)
        df = correlation_analysis(df)
        print("Correlation analysis complete. Shape:", df.shape)
        print("Final columns:", df.columns)
        
        # separate features and target
        X = df.drop('final_test', axis=1)
        y = df['final_test']

        return df, X, y
        
    if __name__ == "__main__":
        df, X, y = main()
        print(df.head())
