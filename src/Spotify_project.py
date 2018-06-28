import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
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

### Remove unneeded columns

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
        if idx >= len(dfs) - 1:
            break
        ax.hist(dfs[idx]['Popularity'], bins=25, color="blue", alpha=0.5)
    return fig, axs

### K Nearest Neighbors to deal with 0 popularity values
def knn_imputation(df):
    '''
    INPUT: Dataframe
    OUTPUT: Dataframe with 0 values replaced with KNN technique
    '''
    df['Popularity'].replace(0,np.nan,inplace = True)
    X_incomplete = df.drop(['Artist','Title','Genre'],axis = 1)
    X_filled_knn = KNN(k=10).complete(X_incomplete)
    cols = X_incomplete.columns
    f = pd.DataFrame(X_filled_knn, columns = cols).reset_index()
    f1 = df[['Title','Artist','Genre']].reset_index()
    finalkn = pd.concat([f1,f],axis = 1)
    finalknn = finalkn.drop(['index'],axis = 1)
    return finalknn


if __name__=='__main__':
    alternative = pd.read_csv('alternative.csv')
    hard_rock = pd.read_csv('hard_rock.csv')
    metal_core = pd.read_csv('metal_core.csv')
    dflist = [alternative,hard_rock,metal_core]
    dfslist = clean_data(dflist)
    mergeddf = combine_dfs(dfslist)
    genre_distributions(dfslist)
    finaldata = knn_imputation(mergeddf)
