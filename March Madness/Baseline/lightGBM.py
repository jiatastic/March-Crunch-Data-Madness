import lightgbm as lgb

from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

from sklearn.model_selection import RandomizedSearchCV

def lightGBM(X_train, y_train):
    param_test = {'num_leaves': sp_randint(6, 255),
                  'min_child_samples': sp_randint(20, 100),
                  'min_child_weight': sp_randint(1, 10),
                  'subsample': sp_uniform(loc=0.5, scale=0.5),
                  'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
                  'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
                  'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}

    clf = lgb.LGBMClassifier(max_depth=100, random_state=1, n_jobs=4, n_estimators=1000)
    RS = RandomizedSearchCV(
        estimator=clf, param_distributions=param_test,
        n_iter= 500,
        scoring='neg_log_loss',
        cv=5,
        random_state=42)

    RS.fit(X_train, y_train)

    return RS