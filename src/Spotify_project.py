from math import ceil
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
from sklearn.metrics import confusion_matrix

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
    plt.show()

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
    plt.show()

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
    plt.show()

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
    ## Pull genre column and create dummy variables
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
    ## Split data into target and features
    X = df.drop(['Title','Artist','Popularity'],axis = 1)
    y = df['Popularity']
    ## Use Sklearn Train/Test split function
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

## Function for Root sum of Squares
def rss(y, y_hat):
    return np.mean((y  - y_hat)**2)

## Function for K Fold Cross Validation
def cv(X, y, base_estimator, n_folds, random_seed=154):
    '''
    INPUT: Feature Matrix, Target Matrix, Regression Method
    number of folds for k fold
    OUTPUT: Train and Test errors for each fold
    '''

    kf = KFold(n_folds, random_state=random_seed)
    test_cv_errors, train_cv_errors = np.empty(n_folds), np.empty(n_folds)
    for idx, (train, test) in enumerate(kf.split(X)):
        # Split into train and test
        X_cv_train, y_cv_train = X[train], y[train]
        X_cv_test, y_cv_test = X[test], y[test]
        # Standardize data
        standardizer = StandardScaler()
        standardizer.fit(X_cv_train, y_cv_train)
        X_cv_train_std = standardizer.transform(X_cv_train)
        #y_cv_train_std = standardizer.transform(y_cv_train)
        X_cv_test_std = standardizer.transform(X_cv_test)
        #y_cv_test_std = standardizer.transform(y_cv_test)

        # not standardizing targets
        y_cv_train_std = y_cv_train
        y_cv_test_std = y_cv_test

        # Fit estimator
        estimator = clone(base_estimator)
        estimator.fit(X_cv_train_std, y_cv_train_std)

        # Measure performance
        y_hat_train = estimator.predict(X_cv_train_std)
        y_hat_test = estimator.predict(X_cv_test_std)
        # Calculate the error metrics
        train_cv_errors[idx] = rss(y_cv_train_std, y_hat_train)
        test_cv_errors[idx] = rss(y_cv_test_std, y_hat_test)
    return train_cv_errors, test_cv_errors

## Function to test different alpha values
def train_at_various_alphas(X, y, model, alphas, n_folds=10, **kwargs):
    '''
    INPUT: Feature matrix, Target Matrix, regression model,
    list of alphas to test, number of folds for kfold
    cross validation
    OUTPUT: Train and Test errors for each alpha value
    '''
    cv_errors_train = pd.DataFrame(np.empty(shape=(n_folds, len(alphas))),
                                     columns=alphas)
    cv_errors_test = pd.DataFrame(np.empty(shape=(n_folds, len(alphas))),
                                        columns=alphas)
    for alpha in alphas:
        train_fold_errors, test_fold_errors = cv(X, y, model(alpha=alpha, **kwargs), n_folds=n_folds)
        cv_errors_train.loc[:, alpha] = train_fold_errors
        cv_errors_test.loc[:, alpha] = test_fold_errors
    return cv_errors_train, cv_errors_test

## Function to test different alphas for Elastic net
def train_at_various_alphas_en(X, y, model, alphas, n_folds=10, **kwargs):
    '''
    INPUT: Feature matrix, Target Matrix, regression model,
    list of alphas to test, number of folds for kfold
    cross validation
    OUTPUT: Train and Test errors for each alpha value
    '''
    cv_errors_train = pd.DataFrame(np.empty(shape=(n_folds, len(alphas))),
                                     columns=alphas)
    cv_errors_test = pd.DataFrame(np.empty(shape=(n_folds, len(alphas))),
                                        columns=alphas)
    for alpha in alphas:
        train_fold_errors, test_fold_errors = cv(X, y, model(alpha=alpha, l1_ratio = 0.5,**kwargs), n_folds=n_folds)
        cv_errors_train.loc[:, alpha] = train_fold_errors
        cv_errors_test.loc[:, alpha] = test_fold_errors
    return cv_errors_train, cv_errors_test

## Function to obtain optimal value for alpha
def get_optimal_alpha(mean_cv_errors_test):
    '''
    INPUT: List of mean test errors from cross validation
    OUTPUT: optimal alpha value where test error is minimized
    '''
    alphas = mean_cv_errors_test.index
    optimal_idx = np.argmin(mean_cv_errors_test.values)
    optimal_alpha = alphas[optimal_idx]
    return optimal_alpha

# MSE vs log(alpha) plot to visualize optimal alpha
def ridge_plot(ridge_alphas,ridge_mean_cv_errors_test,ridge_cv_errors_train, ridge_optimal_alpha):
    '''
    INPUT: List of alpha values, list of mean test errors
    list of mean train errors, optimal alpha value
    OUTPUT: Plot of mean train and test error for each alpha
    '''
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(np.log10(ridge_alphas), ridge_mean_cv_errors_train)
    ax.plot(np.log10(ridge_alphas), ridge_mean_cv_errors_test)
    ax.axvline(np.log10(ridge_optimal_alpha), color='grey')
    ax.set_title("Ridge Regression Train and Test MSE")
    ax.set_xlabel(r"$\log(\alpha)$")
    ax.set_ylabel("MSE")
    plt.show()

# MSE vs log(alpha) plot to visualize optimal alpha
def lasso_plot(lasso_alphas,lasso_mean_cv_errors_train,lasso_mean_cv_errors_test,lasso_optimal_alpha):
    '''
    INPUT: List of alpha values, list of mean test errors
    list of mean train errors, optimal alpha value
    OUTPUT: Plot of mean train and test error for each alpha
    '''
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(np.log10(lasso_alphas), lasso_mean_cv_errors_train)
    ax.plot(np.log10(lasso_alphas), lasso_mean_cv_errors_test)
    ax.axvline(np.log10(lasso_optimal_alpha), color='grey')
    ax.set_title("LASSO Regression Train and Test MSE")
    ax.set_xlabel(r"$\log(\alpha)$")
    ax.set_ylabel("MSE")
    plt.show()

# MSE vs log(alpha) plot to visualize optimal alpha
def elastic_net_plot(en_alphas,en_mean_cv_errors_test,en_mean_cv_errors_train,en_optimal_alpha):
    '''
    INPUT: List of alpha values, list of mean test errors
    list of mean train errors, optimal alpha value
    OUTPUT: Plot of mean train and test error for each alpha
    '''
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(np.log10(en_alphas), en_mean_cv_errors_train)
    ax.plot(np.log10(en_alphas), en_mean_cv_errors_test)
    ax.axvline(np.log10(en_optimal_alpha), color='grey')
    ax.set_title("EN Regression Train and Test MSE")
    ax.set_xlabel(r"$\log(\alpha)$")
    ax.set_ylabel("MSE")
    plt.show()


def model_comparison(X_train, X_test, y_train, y_test):
    '''
    INPUT: Train and Test sets
    OUTPUT: Comparison of RMSE for each model
    '''
    standardizer = StandardScaler()
    standardizer.fit(X_train.values, y_train.values)
    X_train_std = standardizer.transform(X_train.values)
    X_test_std = standardizer.transform(X_test.values)
    y_train_std = y_train
    y_test_std = y_test
    final_ridge = Ridge(alpha=ridge_optimal_alpha).fit(X_train_std, y_train_std)
    final_lasso = Lasso(alpha=lasso_optimal_alpha).fit(X_train_std, y_train_std)
    final_lr = LinearRegression().fit(X_train_std, y_train_std)
    final_en = ElasticNet(alpha = en_optimal_alpha,l1_ratio = 0.5).fit(X_train_std,y_train_std)
    final_ridge_rss = rss(y_test_std, final_ridge.predict(X_test_std))
    final_lasso_rss = rss(y_test_std, final_lasso.predict(X_test_std))
    final_lr_rss = rss(y_test_std, final_lr.predict(X_test_std))
    final_en_rss = rss(y_test_std, final_en.predict(X_test_std))
    print("Final Ridge RMSE: {:2.3f}".format(np.sqrt(final_ridge_rss)))
    print("Final Lasso RMSE: {:2.3f}".format(np.sqrt(final_lasso_rss)))
    print("Final Linear Regression RMSE: {:2.3f}".format(np.sqrt(final_lr_rss)))
    print("Final EN RMSE: {:2.3f}".format(np.sqrt(final_en_rss)))

def bootstrap_train(model, X, y, bootstraps=1000, **kwargs):
    '''
    INPUT: Model, numpy array of features, numpy array of target, number of
           bootstrap samples
    OUTPUT: bootstrap models for each sample
    '''
    bootstrap_models = []
    for i in range(bootstraps):
        boot_idxs = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
        X_boot = X[boot_idxs, :]
        y_boot = y[boot_idxs]
        M = model(alpha = en_optimal_alpha, l1_ratio = 0.5,**kwargs)
        M.fit(X_boot, y_boot)
        bootstrap_models.append(M)
    return bootstrap_models

def get_bootstrap_coefs(bootstrap_models):
    '''
    INPUT: List of elastic net models for each bootstrap sample
    OUTPUT: Coefficients for each elastic net model for each
    bootstrap sample
    '''
    n_models, n_coefs = len(bootstrap_models), len(bootstrap_models[0].coef_)
    bootstrap_coefs = np.empty(shape=(n_models, n_coefs))
    for i, model in enumerate(bootstrap_models):
        bootstrap_coefs[i, :] = model.coef_
    return bootstrap_coefs

def plot_bootstrap_coefs(models, coef_names, n_col=4):
    '''
    INPUT: Bootstrapped models, numpy array of coefficient names
    OUTPUT: histograms of the bootstrapped parameter estimates from a model.
    '''
    bootstrap_coefs = get_bootstrap_coefs(models)
    n_coeffs = bootstrap_coefs.shape[1]
#     n_row = int(ceil(n_coeffs / n_col)) + 1
    n_row = 4
    fig, axs = plt.subplots(n_row, n_col, figsize=(n_col*6, n_row*2))
    for idx, ax in enumerate(axs.flatten()):
        if idx >= bootstrap_coefs.shape[1]:
            break
        ax.hist(bootstrap_coefs[:, idx], bins=25, color="grey", alpha=0.5)
        ax.set_title(coef_names[idx])
        fig.tight_layout()
    plt.show()

def like_distribution_plot(df):
    '''
    INPUT: Dataframe
    OUTPUT: Plot of distribution of features for when
    a song is either liked or disliked.
    '''
    feats = []
    for column in df:
        pos = df[df['target'] == 1][column]
        neg = df[df['target'] == 0][column]
        feats.append((pos,neg))
    n_row = 3
    n_col = 3
    fig, axs = plt.subplots(n_row, n_col, figsize=(12,6))
    for idx, ax in enumerate(axs.flatten()):
        if idx >= len(features):
            break
        feats[idx][0].hist(alpha=0.7, bins=30, label='Like')
        feats[idx][1].hist(alpha=0.7, bins=30, label='Dislike')
    plt.show()

def pos_neg_plot(df):
    '''
    INPUT: Dataframe
    OUTPUT: Plot of positive and negative responses
    '''
    fig = plt.figure(figsize = (12,6))
    ax = fig.add_subplot(111)
    x = np.arange(len(df))
    y = df['target']
    ax.scatter(x,y)
    plt.show()


# Logistic Regression Model
def log_model(df):
    '''
    INPUT: Dataframe
    OUTPUT: Recall, Precision, F1 Score, Confusion Matrix
    '''
    x = df.drop(['song_title','artist','target'], axis = 1).values #returns a numpy array
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    X = pd.DataFrame(x_scaled)
    y = df['target'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    print('Recall of logistic regression classifier on test set: {:.2f}'.format(recall_score(y_test,y_pred)))
    print('Precision of logistic regression classifier on test set: {:.2f}'.format(precision_score(y_test,y_pred)))
    print('F1 Score of logistic regression classifier on test set: {:.2f}'.format(f1_score(y_test,y_pred)))
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    modelCV = LogisticRegression()
    scoring = 'recall'
    results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
    print("10-fold cross validation recall accuracy: %.3f" % (results.mean()))
    print(classification_report(y_test, y_pred))
    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(y_test, y_pred)
    print(confusion_matrix)
    return X_train, X_test, y_train, y_test

# ROC curve
def ROC_curve(X_train, X_test, y_train, y_test):
    '''
    INPUT: Test/Train split data
    OUTPUT: ROC Curve for the regression
    '''
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
    plt.figure(figsize = (12,6))
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()

if __name__=='__main__':
    # Clean Data
    alternative = pd.read_csv('alternative.csv')
    hard_rock = pd.read_csv('hard_rock.csv')
    metal_core = pd.read_csv('metal_core.csv')
    dflist = [alternative,hard_rock,metal_core]
    dfslist = clean_data(dflist)
    mergeddf = combine_dfs(dfslist)
    finaldata = knn_imputation(mergeddf)
    # EDA Plots
    genre_distributions(dfslist)
    popularity_distribution(finaldata)
    corr_heat(finaldata)
    # feature_distribution(finaldata)
    # Hypothesis Test
    t_test_results = genre_hyp_test(finaldata)
    # Linear Regression
    data_w_dummies = genre_dummies(finaldata)
    X_train, X_test, y_train, y_test = train_test(data_w_dummies)
    ridge_alphas = np.logspace(-2, 4, num=250)
    ridge_cv_errors_train, ridge_cv_errors_test = train_at_various_alphas(X_train.values, y_train.values, Ridge, ridge_alphas)
    ridge_mean_cv_errors_train = ridge_cv_errors_train.mean(axis=0)
    ridge_mean_cv_errors_test = ridge_cv_errors_test.mean(axis=0)
    ridge_optimal_alpha = get_optimal_alpha(ridge_mean_cv_errors_test)
    ridge_plot(ridge_alphas,ridge_mean_cv_errors_test,ridge_cv_errors_train, ridge_optimal_alpha)
    lasso_alphas = np.logspace(-3, 1, num=250)
    lasso_cv_errors_train, lasso_cv_errors_test = train_at_various_alphas(X_train.values, y_train.values, Lasso, lasso_alphas, max_iter=5000)
    lasso_mean_cv_errors_train = lasso_cv_errors_train.mean(axis=0)
    lasso_mean_cv_errors_test = lasso_cv_errors_test.mean(axis=0)
    lasso_optimal_alpha = get_optimal_alpha(lasso_mean_cv_errors_test)
    lasso_plot(lasso_alphas,lasso_mean_cv_errors_train,lasso_mean_cv_errors_test,lasso_optimal_alpha)
    en_alphas = np.logspace(-3, 1, num=250)
    en_cv_errors_train, en_cv_errors_test = train_at_various_alphas_en(X_train.values, y_train.values, ElasticNet, en_alphas, max_iter=5000)
    en_mean_cv_errors_train = en_cv_errors_train.mean(axis=0)
    en_mean_cv_errors_test = en_cv_errors_test.mean(axis=0)
    en_optimal_alpha = get_optimal_alpha(en_mean_cv_errors_test)
    elastic_net_plot(en_alphas,en_mean_cv_errors_test,en_mean_cv_errors_train,en_optimal_alpha)
    model_comparison(X_train, X_test, y_train, y_test)
    # bootstrap
    features = data_w_dummies.drop(['Title','Artist','Popularity'],axis = 1)
    models = bootstrap_train(ElasticNet, X_train.values, y_train.values,fit_intercept=False)
    # plot_bootstrap_coefs(models, features.columns, n_col=4)
    # Logistic regression
    logisticdata = pd.read_csv('kag_data.csv')
    data = logisticdata.drop('Unnamed: 0',axis = 1)
    data['duration_ms'] = data['duration_ms'] / 60000
    like_distribution_plot(data)
    pos_neg_plot(data)
    X_train, X_test, y_train, y_test = log_model(data)
    ROC_curve(X_train, X_test, y_train, y_test)
