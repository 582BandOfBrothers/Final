import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def clean_column(df, column):
    column_value = column + '_value';
    values = sorted(df[column].unique())
    value_mapping = dict(zip(values, range(0, len(values) + 1)))
    df[column_value] = df[column].map(value_mapping).astype(int)
    
    df = df.drop([column], axis=1)
    
    return df


def clean_data(df, columns):
    for column in columns:
        df = clean_column(df, column)
        
    return df

def clean_df(df):
    columns = [
        'class',
        'cap-shape',
        'cap-surface',
        'cap-color',
        'bruises',
        'odor',
        'gill-attachment',
        'gill-spacing',
        'gill-size',
        'gill-color',
        'stalk-shape',
        'stalk-root',
        'stalk-surface-above-ring',
        'stalk-surface-below-ring',
        'stalk-color-above-ring',
        'stalk-color-below-ring',
        'veil-type',
        'veil-color',
        'ring-number',
        'ring-type',
        'spore-print-color',
        'population',
        'habitat'
    ]
    
    return clean_data(df, columns)


df_train = pd.read_csv('./data/mushrooms.csv')
df_train = clean_df(df_train)


train_data = df_train.values

clf = RandomForestClassifier(n_estimators=100)


train_features = train_data[:, 1:]
train_target = train_data[:, 0]

# Fit the model to our training data
clf = clf.fit(train_features, train_target)
score = clf.score(train_features, train_target)
print( "Mean accuracy of Random Forest: {0}".format(score) )


df_test = pd.read_csv('./data/mushrooms.csv')

df_test = clean_df(df_test)
test_data = df_test.values

test_x = test_data[:, 1:]
test_y = clf.predict(test_x)


df_test['class_value'] = test_y.astype(int)

df_test['class_value'].to_csv('./data/result.csv', index=False)





