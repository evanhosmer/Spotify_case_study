import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import six
from pandas.plotting import scatter_matrix
from itertools import combinations
import scipy.stats as stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from fancyimpute import KNN
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools import add_constant
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import metrics, cross_validation
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor

### Remove unneeded columns from the data

def clean_data(dflist):
    '''
    INPUT: list of dataframes
    OUTPUT: list of dataframes with unneeded columns removed
    '''
    genres = ['Alternative','Hard Rock','Metalcore']
    for idx, df in enumerate(dflist):
        df.drop('track_href', axis=1,inplace = True)
        df.drop(df.columns[0],axis = 1,inplace = True)
        df['Genre'] = genres[idx]
    return dflist

### Combine three dfs into one for modeling/plotting
def combine_dfs(dfs):
    '''
    INPUT: list of dataframes
    OUTPUT: merged dataframe

    '''
    mergeddf = pd.concat([dfs[0],dfs[1],dfs[2]],axis = 0, ignore_index=True)
    return mergeddf

### Plot distribution of popularity for each genre
def genre_distributions(dfs):
    '''
    INPUT: list of dataframes for each genre
    OUTPUT: histograms of popularity distribution
    '''
    n_row = 2
    n_col = 2
    fig, axs = plt.subplots(n_row, n_col, figsize=(12,6))
    for idx, ax in enumerate(axs.flatten()):
        if idx >= len(dfs):
            break
        ax.hist(dfs[idx]['Popularity'], bins=25, color="blue", alpha=0.5)
    # plt.show()

### K Nearest Neighbors to deal with popularity values of 0
def knn_imputation(df):
    '''
    INPUT: Dataframe
    OUTPUT: Dataframe with 0 values replaced with KNN technique
    '''
    # Replace 0's with NaN's for KNN method
    df['Popularity'].replace(0,np.nan,inplace = True)
    X_incomplete = df.drop(['Artist','Title','Genre'],axis = 1)
    # Call KNN method
    X_filled_knn = KNN(k=10).complete(X_incomplete)
    cols = X_incomplete.columns
    f = pd.DataFrame(X_filled_knn, columns = cols).reset_index()
    f1 = df[['Title','Artist','Genre']].reset_index()
    # Combine KNN data with Title, Artist, Genre
    finalkn = pd.concat([f1,f],axis = 1)
    finalknn = finalkn.drop(['index'],axis = 1)
    return finalknn

## Examine the population distribution where 0 values imputed with KNN
def popularity_distribution(df):
    '''
    INPUT: Dataframe
    OUTPUT: Histogram of popularity distribution for Combined dataframe
    '''
    fig = plt.figure(figsize = (12,6))
    ax = fig.add_subplot(111)
    df['Popularity'].hist(alpha=0.7, bins=30)
    ax.set_title('Popularity distribution with K nearest Neighbors')
    ax.set_xlabel('Popularity')
    ax.set_ylabel('Count')
    # plt.show()

## Plot histograms for each feature
def feature_distribution(df):
    '''
    INPUT: Dataframe
    OUTPUT: Histograms of distribution of features
    '''
    features = [df[column] for column in df]
    n_row = 3
    n_col = 3
    fig, axs = plt.subplots(n_row, n_col, figsize=(12,6))
    for idx, ax in enumerate(axs.flatten()):
        if idx >= len(features):
            break
        ax.hist(features[idx],alpha = 0.7,bins = 30)
    plt.show()

def scatter_mat(df):
    '''
    INPUT: Dataframe
    OUTPUT: Scatter Matrix
    '''
    scatt_mat = scatter_matrix(finalknn, figsize=(20, 20))
    return scatt_mat

def corr_heat(df):
    '''
    INPUT: Dataframe
    OUTPUT: Correlation Heat Map
    '''
    corr = df.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(12, 12))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5},xticklabels=corr.index, yticklabels=corr.columns)
    plt.xticks(rotation=60, ha="right")
    plt.yticks(rotation=0)
    ax.set_title("Correlation Heat Map")
    # plt.show()

### Function to test statistical significance of
### difference in mean popularity for the three genres.
def genre_hyp_test(df):
    '''
    INPUT: Dataframe
    OUTPUT: Results of hypothesis test
    '''
    ## Pull all combinations of the three genres
    combos = combinations(pd.unique(df['Genre']), 2)
    results = pd.DataFrame()
    for genre_1, genre_2 in combos:
        ## Pull popularity data for each genre
        genre_1_ctr = df[df.Genre == genre_1]['Popularity']
        genre_2_ctr = df[df.Genre == genre_2]['Popularity']
        ## Run t test
        p_value = stats.ttest_ind(genre_1_ctr, genre_2_ctr, equal_var=True)[1]
        genre_1_ctr_mean = genre_1_ctr.mean()
        genre_2_ctr_mean = genre_2_ctr.mean()
        diff = genre_1_ctr_mean-genre_2_ctr_mean
        absolute_diff = abs(genre_1_ctr_mean-genre_2_ctr_mean)
        results = results.append({
              'first_genre':genre_1, 'second_genre':genre_2,
              'first_genre_mean':genre_1_ctr_mean, 'second_genre_mean':genre_2_ctr_mean,
              'mean_diff':diff, 'absolute_mean_diff':absolute_diff, 'p_value':p_value},
              ignore_index=True)

    results = results[['first_genre', 'second_genre',
                   'first_genre_mean', 'second_genre_mean',
                   'mean_diff', 'absolute_mean_diff', 'p_value']]
    return results

### Turn the genre feature into a dummy variable
def genre_dummies(df):
    '''
    INPUT: Dataframe
    OUTPUT: Dataframe with dummy variables
    '''
    s = df['Genre']
    dummies = pd.get_dummies(s)
    x = df.drop('Genre',axis = 1)
    df_dummies = pd.concat([x,dummies],axis = 1 )
    return df_dummies

### Function to create train-test split in data
def train_test(df):
    '''
    INPUT: Dataframe
    OUTPUT: Train/Test split datasets
    '''
    X = df.drop(['Title','Artist','Popularity'],axis = 1)
    y = df['Popularity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

if __name__=='__main__':
    alternative = pd.read_csv('alternative.csv')
    hard_rock = pd.read_csv('hard_rock.csv')
    metal_core = pd.read_csv('metal_core.csv')
    dflist = [alternative,hard_rock,metal_core]
    dfslist = clean_data(dflist)
    mergeddf = combine_dfs(dfslist)
    finaldata = knn_imputation(mergeddf)
    genre_distributions(dfslist)
    popularity_distribution(finaldata)
    corr_heat(finaldata)
    # feature_distribution(finaldata)
    t_test_results = genre_hyp_test(finaldata)
    data_w_dummies = genre_dummies(finaldata)
    X_train, X_test, y_train, y_test = train_test(data_w_dummies)
