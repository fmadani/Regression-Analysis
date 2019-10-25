import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import warnings

from sklearn.metrics import mean_absolute_error, r2_score #,mean_squared_error
from sklearn.model_selection import cross_validate
#from sklearn.pipeline import make_pipeline


def settings():
    path="D:/Projects/XPO/"
    os.chdir(path)
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    global correlation_treshold, test_size, eta, n_iterations, dependent_var, n_split_cv,model_scores, poly_nom_degree,\
        parameters, p_value_treshold, fix_outliers_in_test_data, num_top_features, num_of_trees
    test_size=0.30
    dependent_var="DWELL_TIME"
    #Gradient Descent parameters
    eta = 0.1 # [0.1,0.5,1] #learning rate
    n_iterations=100 #for Gradient Descent
    n_split_cv=5 #number of splits of training set
    fix_outliers_in_test_data = False
    #model_scores=["neg_mean_squared_error", "r2"] #measures used in cross validation
    model_scores=["neg_mean_absolute_error", "r2"] #measures used in cross validation

    correlation_treshold = 0.9
    p_value_treshold = 0.05 # used for feature selection
    num_top_features=3 #determined emperically according to the scree plot of f scores of SelectKBest in feature selection

    poly_nom_degree = 2  #for polynomial regression
    #parameters = {'kernel': ('linear', 'rbf','poly'), 'C': [.1,1,5,10,20,100],'epsilon':[.1,.2,.3,.4,.5]}  #Linear SVM parameters used in GridSearchCV
    parameters = {'kernel': ('linear','rbf'), 'C': [.1,.5,10,50,100],'epsilon':[.1,.2,.3,.4,.5]}  #Linear SVM parameters used in GridSearchCV
    num_of_trees=20  #randon forest parameter
# in feature_selection I emirically picked k=3. I looked at the scores_ and pvalues_

def reading_data():
    dataset = pd.read_csv('DataSet.csv')
    return dataset

def visualization(dataset):
    import seaborn as sns
    i = 0
    #quant_columns=[] #used for pairgrid by seaborn
    for column in dataset.columns: # histogram or bar chart drawing for all variables
        i += 1

        if (dataset[column].dtype in ['float64', 'int64']) and (dataset[column].nunique() > 1):
            plt.figure(i)
            dataset[column].plot.hist(title=column)

            plt.show()
            #quant_columns.append(column)

        elif dataset[column].nunique() == 1:
            print('%s has only one value, so no histogram is available' % column)
        else:
            plt.figure(i)
            xlabels = (dataset[column].unique())
            plt.xlabel(dataset['customer_location_type'].unique())
            dataset.groupby([column]).size().plot(kind='bar', title=column)
            plt.show()

    from pandas.plotting import scatter_matrix
    scatter_matrix(dataset)                     #Doesn't work well for numerous variables
    corr_analysis(dataset)

    #sns.pairplot(dataset[quant_columns]) #doesn't work for more than 6 variables

    return

def fix_missing_values(dataset):
    row_index_to_be_deleted1=set(dataset.dock_availability_indicator[dataset.dock_availability_indicator=='U'].index)
    row_index_to_be_deleted2 = set(dataset.forklift_availability_indicator[dataset.forklift_availability_indicator == 'U'].index)
    row_index_to_be_deleted3 = set(dataset.customer_location_type[dataset.customer_location_type == 'U'].index)
    row_index_to_be_deleted= row_index_to_be_deleted1 | row_index_to_be_deleted2 | row_index_to_be_deleted3
    dataset=dataset.drop(row_index_to_be_deleted)
    dataset.reset_index(drop=True,inplace=True)

    return dataset

def binning_to_dummies(dataset,culomn_name,values_to_binned):
    for value in values_to_binned:  # converting values to dummy variables by binning the values, according to
                                    # the histogram of each variable

        new_culomn_data = dataset[culomn_name].where(dataset[culomn_name] == value, other=0)
        new_culomn_data.replace(to_replace=value, value=1, inplace=True)
        new_culomn_name = value
        dataset[new_culomn_name] = new_culomn_data

    new_culomn_data = dataset[culomn_name].where(dataset[culomn_name].isin(values_to_binned), other=1)
    new_culomn_data = new_culomn_data.where(new_culomn_data==1,other=0)
    new_culomn_name = culomn_name+'_the_rest'
    dataset[new_culomn_name] = new_culomn_data
    dataset = dataset.drop(columns=culomn_name)

    return dataset

def data_transformation(dataset):
    from sklearn.preprocessing import LabelBinarizer, LabelEncoder, OneHotEncoder
    dataset['total_activities']=dataset.DL+dataset.DP+dataset.HE+dataset.HK+dataset.HL+dataset.PU+dataset.SE+dataset.SL
    dataset=dataset.drop(columns=["DL","PU","DP","HE","HK","SE","HL","SL"]) #these columns are removed after reviewing the data saummary

    dataset['unusual_mat']=np.logical_or(dataset.HAZMAT_FLAG,dataset.FREEZABLE_FLAG)
    dataset = dataset.drop(columns=['HAZMAT_FLAG','FREEZABLE_FLAG'])

    lb_style = LabelBinarizer()
    #dataset.HAZMAT_FLAG=lb_style.fit_transform(dataset.HAZMAT_FLAG)
    #dataset.FREEZABLE_FLAG=lb_style.fit_transform(dataset.FREEZABLE_FLAG)
    dataset.unusual_mat=lb_style.fit_transform(dataset.unusual_mat)

    #dataset=binning_to_dummies(dataset,'customer_location_type',['Business/Commercial','Distribution Center'])
    ##binning_to_dummies function acts like using OneHotEncoder used below. Also, it allows to consider the rest of
    ## features as a single column.

    ohe_encoder=OneHotEncoder()
    dataset_cat_hot=ohe_encoder.fit_transform(dataset.customer_location_type.to_numpy().reshape(-1,1)).toarray()
    df_dataset_cat_hot = pd.DataFrame(dataset_cat_hot, columns=dataset.customer_location_type.unique())
    df_dataset_cat_hot=df_dataset_cat_hot[["Business/Commercial","Distribution Center"]]
    dataset = dataset.join(df_dataset_cat_hot)
    dataset.drop(["customer_location_type"],axis=1,inplace=True)

    dataset.dock_availability_indicator.replace(to_replace='Y',value=1, inplace=True)
    dataset.dock_availability_indicator.replace(to_replace='N',value=0, inplace=True)

    dataset.forklift_availability_indicator.replace(to_replace='Y', value=1, inplace=True)
    dataset.forklift_availability_indicator.replace(to_replace='N', value=0, inplace=True)

    dataset=dataset.drop(columns=['Distribution Center'])#,'dock_availability_indicator','forklift_availability_indicator','unusual_mat','Business/Commercial'])



    return dataset

def corr_analysis(dataset):
    ##finding the correlation more than correlation threshhold
    correlations = dataset.select_dtypes(include=['int32','int64', 'float64']).corr()
    for i in range(correlations.shape[0]):
        for j in range(correlations.shape[1]):
            number_of_high_corr = 0
            if i != j and abs(correlations.iat[i, j]) >= correlation_treshold:
                number_of_high_corr += 1
                print(correlations.columns[i], correlations.index.values[j], correlations.iat[i, j])
    if number_of_high_corr == 0: print('there is no high correlation between the features')

    correlations[dependent_var].sort_values()  # looking at the correlation between the DV and IV's

    # Visualizing the correlation matrix
    corr = dataset.corr()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, len(dataset.columns), 1)
    ax.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    ax.set_xticklabels(dataset.columns)
    ax.set_yticklabels(dataset.columns)
    plt.show()

    return

def feature_univariate_selection(dataset):
    from sklearn.feature_selection import GenericUnivariateSelect, chi2
    X=dataset.drop(dependent_var,axis=1)
    print(X.shape)
    #y=list(dataset[dependent_var].values)
    y=np.asarray(dataset[dependent_var],dtype='int32') #y is float, but transformer only worked for int32
    transformer = GenericUnivariateSelect(chi2, 'k_best', param='all')
    X_new = transformer.fit_transform(X,y )
    print(X_new.shape)

    # no feature was deleted by this method!

    return dataset

def dataset_splitting(dataset):
    from sklearn.model_selection import train_test_split
    train_set, test_set= train_test_split(dataset, test_size= test_size)#, random_state= 42)
    X_train = train_set.drop([dependent_var], axis=1)
    X_test = test_set.drop([dependent_var], axis=1)
    y_train = train_set[dependent_var]
    y_test = test_set[dependent_var]
    #X_train_tmp=deepcopy(X_train)

    X_train.reset_index(drop=True,inplace=True)
    X_test.reset_index(drop=True,inplace=True)
    y_train.reset_index(drop=True,inplace=True)
    y_test.reset_index(drop=True,inplace=True)


    return X_train, y_train, X_test, y_test

def fix_outliers_by_three_sigma(X_train, y_train, X_test, y_test):
    X_train_outlier_index= set()
    X_test_outlier_index=set()
    for var in X_train.columns:  # those data are removed that thier ratio to DWELL_TIME is abnormal!
        if X_train[var].nunique() >2:  #ignoring binary variables
            ratio=y_train/X_train[var]
            #up_limit = X_train[var].mean() + 3 * X_train[var].std()
            #up_limit=X_train[var].mean()+3*X_train[var].std()
            up_limit = ratio.mean() + 3 *ratio.std()
            #X_train_outlier_index=X_train_outlier_index.union(np.where(X_train[var]>=up_limit)[0])
            X_train_outlier_index=X_train_outlier_index.union(np.where(ratio>=up_limit)[0])
            ratio_test= y_test/X_test[var]
            #X_test_outlier_index=X_test_outlier_index.union(np.where(X_test[var]>=up_limit)[0])
            X_test_outlier_index=X_test_outlier_index.union(np.where(ratio_test>=up_limit)[0])

    X_train.drop(X_train_outlier_index,inplace=True)
    y_train.drop(X_train_outlier_index,inplace=True)
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)

    if fix_outliers_in_test_data == True:
        X_test.drop(X_test_outlier_index,inplace=True)
        y_test.drop(X_test_outlier_index,inplace=True)
        X_test.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)

    return X_train, y_train, X_test, y_test

def fix_outliers_by_LocalOutlierFactor(X_train, y_train, X_test, y_test):
    from sklearn.neighbors import LocalOutlierFactor
    lof = LocalOutlierFactor(novelty=True)
    X_train_outlier_index= set()
    X_test_outlier_index=set()
    for var in X_train.columns:
        if X_train[var].nunique() >2:
            lof.fit(X_train[var].values.reshape(-1, 1))
            outlier_indexes=set((np.where(lof.predict(X_train[var].values.reshape(-1,1))==-1))[0])
            X_train_outlier_index=X_train_outlier_index.union(outlier_indexes)
            #must not remove outliers from test dataset to see how the model performs for all posibilities
            #outlier_indexes_test_set = set((np.where(lof.predict(X_test[var].values.reshape(-1, 1)) == -1))[0])
            #X_test_outlier_index=X_test_outlier_index.union(outlier_indexes_test_set)

    X_train.drop(X_train_outlier_index,inplace=True)
    y_train.drop(X_train_outlier_index,inplace=True)
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)

    if fix_outliers_in_test_data == True:
        X_test.drop(X_test_outlier_index,inplace=True)
        y_test.drop(X_test_outlier_index,inplace=True)
        X_test.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)

    return X_train, y_train, X_test, y_test

def data_scaling(X_train, y_train, X_test, y_test):
    from sklearn.preprocessing import MinMaxScaler  # ,StandardScaler,  #feature scaling. IV's are only scaled.
    # scaler = StandardScaler(with_mean=True, with_std=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X_train)
    X_train = pd.DataFrame(data=scaler.transform(X_train),columns=X_train.columns)
    scaler.fit(X_test)
    X_test = pd.DataFrame(data=scaler.transform(X_test),columns=X_train.columns)

    return X_train, y_train, X_test, y_test

def feature_selection(X_train, y_train, X_test, y_test):#dataset):#

    from sklearn.feature_selection import SelectKBest, f_regression

    selector = SelectKBest(f_regression, k=num_top_features)  #  is determined empirically based on the scree diagram
    data_selected = selector.fit_transform(X_train, y_train)
    X_train_new = pd.DataFrame(data=data_selected, columns=X_train.columns[[np.where(selector.get_support())]][0])
    print('selected features: ', X_train_new.columns.values)
    X_test_new = pd.DataFrame(data=selector.transform(X_test),columns=X_test.columns[[np.where(selector.get_support())]][0])

    plt.figure()
    fscores=pd.DataFrame(data=[X_train.columns,selector.scores_]).T
    fscores.columns=['feature','Fscore']
    fscores.sort_values(by=['Fscore'], ascending=False, inplace=True)
    fscores.reset_index(inplace=True)
    p_values = selector.pvalues_
    plt.title('Scree plot of F scores')
    plt.xticks(np.arange(fscores.shape[0]),labels=fscores.feature,rotation=90)
    plt.plot(fscores.Fscore)
    for x,y in zip(np.arange(fscores.shape[0]),fscores.Fscore):
        plt.annotate('%.2f' % (fscores.Fscore[x]), xy=(x,y))

    plt.tight_layout()
    plt.show()

    print('Variables with p-value > %.2f:' % p_value_treshold,dataset.drop(columns=dependent_var).columns[np.where(p_values > p_value_treshold)])

    return X_train_new, y_train, X_test_new, y_test

def display_scores(model_name, rmae_mean_cv,rmae_std_cv, r2_mean_cv,r2_std_cv,rmae_test,r2_test):
    print("=========%s=========" % model_name)
    print("Cross Validation:")
    print("RMAE:{0:.2f} +/- {1:.2f}".format(rmae_mean_cv, rmae_std_cv))
    print("R2: %.2f +/- %.2f" % (r2_mean_cv,r2_std_cv))
    print()
    print("test:")
    print("RMAE:{0:.2f}".format(rmae_test))
    print("R2: %.2f" % (r2_test))

def residual_plots(y_test, y_pred, method):
    plt.figure()
    plt.plot(range(0, int(max(y_test))), 'r')
    plt.scatter(y_test, y_pred)
    plt.title('Scatter plot y predicted vs y train - '+method)
    plt.xlabel('y test')
    plt.ylabel('y predicted')
    #plt.ylim(0,300)
    plt.show()
    plt.figure()
    plt.title('residuals histogram-%s'% method)
    plt.hist(np.abs(y_pred - y_test), bins=50, histtype='step')
    plt.xlabel('residual amount')
    plt.ylabel('number')
    plt.show()
    return

def training_vs_test_performance_curve(regressor,X_train, y_train, X_test, y_test,regressor_name):
    from sklearn.feature_selection import SelectKBest, f_regression
    rmae_training = []
    rmae_test = []
    for i in range(1,X_train.shape[1]+1):#dataset.drop(dependent_var,axis=1).shape[1]+1):#
        selector = SelectKBest(f_regression, k=i)  # is determined empirically based on the scree diagram
        '''
        rmae_training_j = []
        rmae_test_j = []
        for j in range(1,11):
            X_train, y_train, X_test, y_test = dataset_splitting(dataset)
            X_train, y_train, X_test, y_test = fix_outliers_by_three_sigma(X_train, y_train, X_test, y_test)
            X_train, y_train, X_test, y_test = data_scaling(X_train, y_train, X_test, y_test)

            data_selected = selector.fit_transform(X_train, y_train)
            X_train_new = pd.DataFrame(data=data_selected, columns=X_train.columns[[np.where(selector.get_support())]][0])
            X_test_new = selector.transform(X_test)

            regr.fit(X_train_new, y_train)
            y_pred = regr.predict(X_test_new)

            lr_rmae_mean_cv, lr_rmae_std_cv, lr_r2_mean_cv, lr_r2_std_cv = cross_validation(regr, X_train_new, y_train)
            lr_rmae_test = np.sqrt(mean_absolute_error(y_test, y_pred))

            rmae_training_j.append(lr_rmae_mean_cv)
            rmae_test_j.append(lr_rmae_test)'''

        data_selected = selector.fit_transform(X_train, y_train)
        X_train_new = pd.DataFrame(data=data_selected, columns=X_train.columns[[np.where(selector.get_support())]][0])
        X_test_new = pd.DataFrame(data=selector.transform(X_test), columns= X_test.columns[[np.where(selector.get_support())]][0])

        regressor.fit(X_train_new, y_train)
        y_pred = regressor.predict(X_test_new)

        lr_rmae_mean_cv, lr_rmae_std_cv, lr_r2_mean_cv, lr_r2_std_cv = cross_validation(regressor, X_train_new, y_train)
        lr_rmae_test = np.sqrt(mean_absolute_error(y_test, y_pred))


        #rmae_training.append(np.mean(rmae_training_j))
        #rmae_test.append(np.mean(rmae_test_j))
        rmae_training.append(lr_rmae_mean_cv)
        rmae_test.append(lr_rmae_test)


    plt.figure()
    plt.plot(rmae_training,label='training')
    plt.plot(rmae_test,label='test')
    plt.ylabel('Error (RMAE)')
    plt.xticks(np.arange(0,X_train.shape[1]+1))
    plt.xlabel('Model Complexity (# of features)')
    plt.xlim(1,X_train.shape[1])
    #plt.ylim(0,4)
    plt.legend()
    plt.title('test vs training - %s' % regressor_name)
    plt.show()

    return

def cross_validation(model, X_train, y_train):

    scores = cross_validate(model, X_train, y_train, scoring=model_scores, cv=n_split_cv)
    #rmae_mean = np.sqrt(-scores['test_neg_mean_squared_error']).mean()
    #rmae_std= np.sqrt(-scores['test_neg_mean_squared_error']).std()
    rmae_mean = np.sqrt(-scores['test_neg_mean_absolute_error']).mean()
    rmae_std = np.sqrt(-scores['test_neg_mean_absolute_error']).std()
    r2_mean = scores["test_r2"].mean()
    r2_std = scores["test_r2"].std()


    return rmae_mean, rmae_std, r2_mean, r2_std

def linear_regression(X_train, y_train, X_test, y_test,regression_type):
    from sklearn import linear_model

    regr = linear_model.LinearRegression()
    regr.fit(X_train,y_train)
    y_pred=regr.predict(X_test)

    lr_rmae_mean_cv,lr_rmae_std_cv, lr_r2_mean_cv, lr_r2_std_cv = cross_validation(regr, X_train, y_train)

    #lr_rmae_test=np.sqrt(mean_squared_error(y_test, y_pred))
    lr_rmae_test = np.sqrt(mean_absolute_error(y_test, y_pred)) #mean absolute error (MAE) performed greatly better than
    # MSE because of the existance of many outliers (even after fixing outliers)
    lr_r2_test =r2_score(y_test,y_pred)

    display_scores(regression_type, lr_rmae_mean_cv, lr_rmae_std_cv, lr_r2_mean_cv, lr_r2_std_cv,lr_rmae_test,lr_r2_test)
    residual_plots(y_test,y_pred,regression_type)
    training_vs_test_performance_curve(regr,X_train, y_train, X_test, y_test,regression_type)

    return lr_rmae_test, lr_r2_test

def LR_GD_modelling(X_train, y_train, X_test, y_test, learning_rate):
    m=X_train.shape[0]
    n=X_train.shape[1]

    theta=np.random.randn(n,1) #random initialization
    theta=pd.DataFrame(theta)
    cost_fun=[]


    theta_tmp=[]
    for iteration in range(n_iterations):
        gradients=pd.DataFrame(2/m*pd.DataFrame(X_train).T.dot(pd.DataFrame(X_train).dot(theta).iloc[:,0]-pd.DataFrame(y_train).iloc[:,0]))
        theta=theta-learning_rate*gradients
        y_pred = (X_test.dot(theta))
        theta_tmp.append(theta.iloc[0]) #collecting theta 0 for drawing the learning curve
        cost_fun.append(np.sqrt(mean_squared_error(y_test, y_pred)))

    plt.plot(theta_tmp,cost_fun)  #drawing cost func vs thetha 0
    plt.xlabel("theta 0")
    plt.ylabel("cost function (rmae)")
    plt.title("Learning rate: "+ str(learning_rate))
    plt.show()

    gdlr_rmae= np.sqrt(mean_squared_error(y_test, y_pred))
    gdlr_r2= r2_score(y_test, y_pred)
    print("========Gradient Descent LR with learning rate "+str(learning_rate)+' =======')
    print("rmae:",gdlr_rmae)
    print("R2:",gdlr_r2)

    return gdlr_rmae, gdlr_r2

def decision_tree_reg(X_train, y_train, X_test, y_test):
    from sklearn.tree import DecisionTreeRegressor
    tree_reg=DecisionTreeRegressor(criterion='mae')
    tree_reg.fit(X_train,y_train)
    y_pred=tree_reg.predict(X_test)

    tree_rmae_mean_cv, tree_rmae_std_cv, tree_r2_mean_cv, tree_r2_std_cv = cross_validation(tree_reg,X_train,y_train)
    tree_lr_rmae_test = np.sqrt(mean_absolute_error(y_test, y_pred))
    tree_lr_r2_test = r2_score(y_test, y_pred)
    display_scores('decision tree regression', tree_rmae_mean_cv, tree_rmae_std_cv, tree_r2_mean_cv, tree_r2_std_cv,tree_lr_rmae_test,tree_lr_r2_test)
    residual_plots(y_test,y_pred,'decision tree regression')
    training_vs_test_performance_curve(tree_reg,X_train, y_train, X_test, y_test,'decision tree regression')


    return tree_lr_rmae_test, tree_lr_r2_test

def random_forest_reg(X_train, y_train, X_test, y_test):
    from sklearn.ensemble import RandomForestRegressor
    forest_reg=RandomForestRegressor(n_estimators=num_of_trees,criterion='mae')
    forest_reg.fit(X_train,y_train)
    y_pred=forest_reg.predict(X_test)

    rf_reg_rmae_mean_cv,rf_reg_rmae_std_cv, rf_reg_r2_mean_cv, rf_reg_r2_std_cv = cross_validation(forest_reg, X_train, y_train)

    rf_reg_rmae_test = np.sqrt(mean_absolute_error(y_test, y_pred))
    rf_reg_r2_test = r2_score(y_test, y_pred)

    display_scores('Random Forest regression', rf_reg_rmae_mean_cv,rf_reg_rmae_std_cv, rf_reg_r2_mean_cv,
                   rf_reg_r2_std_cv,rf_reg_rmae_test,rf_reg_r2_test)

    residual_plots(y_test,y_pred,'Random Forest regression')
    training_vs_test_performance_curve(forest_reg,X_train, y_train, X_test, y_test,'Random Forest regression')



    return rf_reg_rmae_test, rf_reg_r2_test


def poly_reg(X_train, y_train, X_test, y_test):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn import linear_model
    from sklearn.feature_selection import SelectKBest, f_regression


    poly_reg = PolynomialFeatures(poly_nom_degree)
    poly_reg.fit_transform(X_train)
    X_train_new=pd.DataFrame(data=poly_reg.fit_transform(X_train),columns= poly_reg.get_feature_names(X_train.columns))
    X_test_new=pd.DataFrame(data=poly_reg.fit_transform(X_test),columns=poly_reg.get_feature_names(X_test.columns))
    #poly_reg_rmae_mean, poly_reg_r2_mean = linear_regression(X_train, y_train, X_test, y_test,"Polynomial regression")


    regr = linear_model.LinearRegression()
    regr.fit(X_train_new, y_train)
    y_pred = regr.predict(X_test_new)

    poly_reg_rmae_mean_cv, poly_reg_rmae_std_cv, poly_reg_r2_mean_cv, poly_reg_r2_std_cv = cross_validation(regr, X_train_new, y_train)

    poly_reg_rmae_test = np.sqrt(mean_absolute_error(y_test, y_pred))  # mean absolute error (MAE) performed greatly better than
    poly_reg_r2_test = r2_score(y_test, y_pred)

    display_scores("Polynomial regression", poly_reg_rmae_mean_cv, poly_reg_rmae_std_cv, poly_reg_r2_mean_cv, poly_reg_r2_std_cv, poly_reg_rmae_test,
                   poly_reg_r2_test)
    residual_plots(y_test, y_pred, "Polynomial regression")
    #training_vs_test_performance_curve(regr, X_train, y_train, X_test, y_test, "Polynomial regression")
    #training vs test performance curve
    rmae_training = []
    rmae_test = []
    for i in range(1,X_train.shape[1]+1):
        selector = SelectKBest(f_regression, k=i)
        selector.fit(X_train, y_train)
        data_selected = pd.DataFrame(data=selector.transform(X_train),columns=X_train.columns[[np.where(selector.get_support())]][0])
        poly_reg.fit(data_selected)
        data_selected_transformed_to_poly=poly_reg.transform(data_selected)
        X_train_new = pd.DataFrame(data=data_selected_transformed_to_poly, columns=poly_reg.get_feature_names(data_selected.columns))

        test_data_selected = pd.DataFrame(data=selector.transform(X_test),columns=X_test.columns[[np.where(selector.get_support())]][0])
        test_data_selected_transformed_to_poly=poly_reg.transform(test_data_selected)

        X_test_new = pd.DataFrame(data=test_data_selected_transformed_to_poly, columns= poly_reg.get_feature_names(test_data_selected.columns))

        regr.fit(X_train_new, y_train)
        y_pred = regr.predict(X_test_new)

        poly_reg_rmae_mean_cv, poly_reg_rmae_std_cv, poly_reg_r2_mean_cv, poly_reg_r2_std_cv = cross_validation(regr, X_train_new, y_train)
        poly_reg_rmae_test = np.sqrt(mean_absolute_error(y_test, y_pred))


        #rmae_training.append(np.mean(rmae_training_j))
        #rmae_test.append(np.mean(rmae_test_j))
        rmae_training.append(poly_reg_rmae_mean_cv)
        rmae_test.append(poly_reg_rmae_test)


    plt.figure()
    plt.plot(rmae_training,label='training')
    plt.plot(rmae_test,label='test')
    plt.ylabel('Error (RMAE)')
    plt.xticks(np.arange(0,X_train.shape[1]+1))
    plt.xlabel('Model Complexity (# of features)')
    plt.xlim(1,X_train.shape[1])
    #plt.ylim(0,4)
    plt.legend()
    plt.title('test vs training - %s' % regressor_name)
    plt.show()



    return poly_reg_rmae_mean, poly_reg_r2_mean


def svr(X_train, y_train, X_test, y_test):
    from sklearn.svm import SVR
    from sklearn.model_selection import GridSearchCV
    svr=SVR(gamma='scale')
    reg_svr = GridSearchCV(svr, parameters, cv=n_split_cv)
    reg_svr.fit(X_train, y_train)
    #sorted(reg_svr.cv_results_)
    y_pred=reg_svr.predict(X_test)
    svr_rmae=mean_absolute_error(y_test, y_pred)
    svr_r2= r2_score(y_test, y_pred)
    #svr_rmae_mean, svr_rmae_std,svr_r2_mean, svr_r2_std = cross_validation(reg_svr, X_train, y_train)
    print('Support Vector Regression rmase: %.2f and r2: %.2f'% (svr_rmae, svr_r2))
    print('best SVR model:',reg_svr.best_estimator_)
    residual_plots(y_test,y_pred,'SVR')

    return  svr_rmae, svr_r2



if __name__ == '__main__':
    settings()
    dataset=reading_data()
    #visualization(dataset)
    dataset=fix_missing_values(dataset)
    dataset=data_transformation(dataset)
    #dataset=feature_univariate_selection(dataset)  #no features was dropped by the method
    #corr_analysis(dataset)
    #training_vs_test(dataset)

    X_train, y_train, X_test, y_test = dataset_splitting(dataset)
    X_train, y_train, X_test, y_test = fix_outliers_by_three_sigma(X_train, y_train, X_test, y_test)
    #X_train, y_train, X_test, y_test= fix_outliers_by_LocalOutlierFactor(X_train, y_train, X_test, y_test)
    X_train, y_train, X_test, y_test = data_scaling(X_train, y_train, X_test, y_test)
    #X_train, y_train, X_test, y_test = feature_selection(X_train, y_train, X_test, y_test)

    #LR_rmae, LR_r2 = linear_regression(X_train, y_train, X_test, y_test, "Linear regression")
    #LR_GD_rmae, LR_GD_r2 = LR_GD_modelling(X_train, y_train, X_test, y_test, eta) #dataset will be scaled in this module
    #decision_tree_reg_rmae, decision_tree_reg_r2=decision_tree_reg(X_train, y_train, X_test, y_test)
    #rf_reg_rmae,rf_reg_r2= random_forest_reg(X_train, y_train, X_test, y_test)
    #poly_reg_rmae, poly_reg_r2 = poly_reg(X_train, y_train, X_test, y_test)
    svr_rmae, svr_r2 = svr(X_train, y_train, X_test, y_test) #svr:support vector regression

