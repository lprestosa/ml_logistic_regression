###
# Logistic Regression - Classification Algorithm
# Success:  This code gave Accuracy: 0.8501209275857163 2022-03-22
###

import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns

# Global variables
csv_infile = r'/py/data/kaggle/weatherAUS.csv'

def main():
    df = pd.read_csv(csv_infile)
    data_preprocessing(df)


def data_preprocessing(df):

    ##ok print(df.head())
    ##ok print(df.describe())
    ##ok print(df.isnull().sum)

    # Calculate missing value percentage for each column
    missing_count = df.isnull().sum()  # count of missing values
    value_count = df.isnull().count()  # count of all values
    missing_percentage = round(missing_count / value_count * 100, 1)
    missing_df = pd.DataFrame({'count': missing_count, 'percentage': missing_percentage})
    ##ok print(missing_df)

    # Drop columns with a large amount of missing values
    df = df.drop(['Evaporation', 'Sunshine', 'Cloud3pm', 'Cloud9am'], axis=1)

    # Drop rows with missing labels
    df = df.dropna(subset=['RainTomorrow'])

    ##ok print(df.shape)

    num_list = []
    cat_list = []

    for column in df:
        if column != 'RainTomorrow':
            if is_numeric_dtype(df[column]):
                num_list.append(column)
            elif is_string_dtype(df[column]):
                cat_list.append(column)

    ##ok print(num_list)
    ##ok print(cat_list)

    # Numerical variables: impute missing values with mean
    df.fillna(df.mean(), inplace=True)

    # Categorical variables: impute missing values with 'Unknown'
    for i in (cat_list):
        if df[i].isnull().any():
            df[i].fillna('Unknown', inplace=True)

    for column in df:
        plt.figure(column, figsize=(5, 5))
        plt.title(column)
        if is_numeric_dtype(df[column]):
            #df[column].plot(kind='hist')
            pass
        elif is_string_dtype(df[column]):
            # show only the top 10 value count in each categorical column
            df[column].value_counts()[:10].plot(kind='bar')
            pass
        ##ok plt.show()
#    return df

# def feature_engineering(df):

    # Step 1: address outliers in Rainfall
    maximum =df['Rainfall'].quantile(0.9)
    df = df[df['Rainfall'] < maximum]
    df['Rainfall'].plot(kind = 'hist')
    # print(df.shape)
    # plt.show()

    # Step 2. feature transformation
    # Date variable was transformed into Month.  Because Date has such
    # high cardinality which makes it impossible to bring out patterns.

    df['Month'] = pd.to_datetime(df['Date']).dt.month.apply(str)
    # df['Month'].value_counts().plot(kind = 'bar')
    # print(df['Month'].value_counts())

    # Step 3. Categorical feature encoding
    # Logistic regression only accepts numeric values.
    # Therefore encode categorical data into numbers using
    # one-hot encoding - used for low cardinality non-ordinal data
    # label encoding -  for ordinal data with high cardinality
    # https://www.analyticsvidhya.com/blog/2020/03/one-hot-encoding-vs-label-encoding-using-scikit-learn/

    categorical_features = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'Month', 'RainTomorrow']
    for i in categorical_features:
        df[i] = LabelEncoder().fit_transform(df[i])

    # Verify all columns are either int or float
    ##ok print(df.info())

    # Step 4: Feature selection
    # https://towardsdatascience.com/feature-selection-and-eda-in-python-c6c4eb1058a3

    # Use Correlation Analysis is a common multivariate EDA method that assists in identifying higly correlated values.
    plt.figure(1,figsize=(15,15))
    correlation = df.corr()
    sns.heatmap(correlation, cmap='GnBu', annot = True)
    plt.show()

    # Drop and rearrange columns
    df = df[['Month', 'Location', 'MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
             'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure3pm', 'RainToday', 'RainTomorrow']]
    print(df.info())
#    return df

# def model_building(df):
    # X - input features matrix: all rows using ":" and all columns except the last one ":-1"
    X = df.iloc[:,:-1]
    # y - output target vector: all rows using ":" and last column "-1"
    y = df.iloc[:,-1]

    # split into train and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    print( X_train.shape , X_test.shape, y_train.shape, y_test.shape)

    # execute logistic regression
    reg = LogisticRegression(max_iter = 500)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    print(y_pred)

# def model_evaluation(df):
    confusion_matrix =  metrics.plot_confusion_matrix(reg, X_test, y_test, cmap ='GnBu' )
    #confusion_matrix = metrics.ConfusionMatrixDisplay (reg, X_test, y_test )
    ##print(confusion_matrix)
    plt.show()

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    y_pred_prob = reg.predict_proba(X_test[:,1])
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_prob)
    plt.plot(fpr,tpr)
    plt.show()

    auc = metrics.roc_auc_score(y_test, y_pred_prob)
    print("AUC:", round(auc,2))



if __name__ == '__main__':
    main()
