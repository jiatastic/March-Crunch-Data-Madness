from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV

def catboost(X_train,y_train):
    param_distributions = {'learning_rate': [0.01, 0.05, 0.1, 0.5, 1],
                           'depth': [3, 5, 7, 9],
                           'l2_leaf_reg': [1, 3, 5, 7, 9]}

    CB = CatBoostClassifier(random_state = 42, iterations=30)
    RS = RandomizedSearchCV(CB, param_distributions, n_iter=100, cv=5, scoring='neg_log_loss')

    RS.fit(X_train, y_train)

    return RS