from striprtf.striprtf import rtf_to_text
import json

with open('D:/Intern/Dendrite/algoparams_from_ui.json.rtf', 'r', encoding='utf-8') as file:
    rtf_content = file.read()

plain_text = rtf_to_text(rtf_content)
data = json.loads(plain_text)

target = data['design_state_data']['target']['target']
prediction_type = data['design_state_data']['target']['prediction_type']
import pandas as pd
df = pd.read_csv(data['design_state_data']['session_info']['dataset'])

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

class SpeciesEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = LabelEncoder()
        
    def fit(self, X, y=None):
        if 'species' in X.columns:
            self.encoder.fit(X['species'])
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        if 'species' in X_copy.columns:
            X_copy['species_encoded'] = self.encoder.transform(X_copy['species'])
            # Only change i have made from the old code without pipeline is this dropping the species column here
            X_copy.drop('species', axis=1, inplace=True)
        return X_copy
    
class TargetImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='mean', fill_value=None):
        self.strategy = strategy
        self.fill_value = fill_value
        self.imputer = SimpleImputer(strategy=self.strategy, fill_value=self.fill_value)
        
    def fit(self, X, y=None):
        self.imputer.fit(X[[target]])
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        X_copy[target] = self.imputer.transform(X_copy[[target]])
        return X_copy


from sklearn.pipeline import Pipeline

strategy = 'mean'
fill_value = None
if target in data['design_state_data']['feature_handling']:
    details = data['design_state_data']['feature_handling'][target]
    if details['is_selected']:
        if details['feature_details']['missing_values'] == "Impute":
            if details['feature_details']['impute_with'] == "Average of values":
                strategy = 'mean'
            elif details['feature_details']['impute_with'] == "custom":
                strategy = 'constant'
                fill_value = details['feature_details']['impute_value']


preprocessing = Pipeline([
    ('encoder', SpeciesEncoder()),
    ('imputer', TargetImputer(strategy=strategy, fill_value=fill_value))
])


df_transformed = preprocessing.fit_transform(df)
#print(df_transformed.columns)

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from scipy.stats import pearsonr
from sklearn.preprocessing import FunctionTransformer

class FeatureReductionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, json_data):
        self.json_data = json_data
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        feature_reduction = self.json_data['design_state_data']['feature_reduction']
        method = feature_reduction['feature_reduction_method']
        
        if method == "No Reduction":
            return df_transformed

        elif method == "Corr with Target":
            correlated_features = []
            for feature in X.columns:
                if feature != target:
                    correlation, _ = pearsonr(X[feature], X[target])
                    if abs(correlation) > 0.5:  
                        correlated_features.append(feature)
            correlated_features.append('species_encoded')
            return X[correlated_features]

        elif method == "Tree-based":

            if prediction_type == "Regression":
                model = RandomForestRegressor(n_estimators=int(feature_reduction['num_of_trees']), max_depth=int(feature_reduction['depth_of_trees']))
            else:
                model = RandomForestClassifier(n_estimators=int(feature_reduction['num_of_trees']), max_depth=int(feature_reduction['depth_of_trees']))
            model = model.fit(X.drop(target, axis=1), X[target]) 
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            top_features = [X.columns[i] for i in indices[:int(feature_reduction['num_of_features_to_keep'])]]
            top_features.append('species_encoded')
            return X[top_features]
        

        elif method == "PCA":
            pca = PCA(n_components=int(feature_reduction['num_of_features_to_keep']))
            reduced_data = pca.fit_transform(X.drop(target, axis=1)) 
            reduced_df = pd.DataFrame(reduced_data)
            reduced_df['species_encoded'] = X['species_encoded'].values
            return pd.DataFrame(reduced_data)

        else:
            raise ValueError(f"Unknown feature reduction method: {method}")

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor , GradientBoostingClassifier, GradientBoostingRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet, SGDClassifier, SGDRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor
import xgboost as xgb

models = {}
def create_model(algorithm, details):
        if algorithm == "RandomForestClassifier":
                min_trees = details['min_trees']
                max_trees = details['max_trees']
                min_depth = details['min_depth']
                max_depth = details['max_depth']
                min_samples_per_leaf_min_value = details['min_samples_per_leaf_min_value']
                min_samples_per_leaf_max_value = details['min_samples_per_leaf_max_value']
                parallelism = details['parallelism']
                if details['feature_sampling_statergy'] == "Default":
                     max_features = 'auto'
                model = RandomForestClassifier(
                        n_estimators=min_trees, 
                        max_depth=max_depth, 
                        min_samples_split=min_samples_per_leaf_min_value, 
                        min_samples_leaf=min_samples_per_leaf_max_value, 
                        n_jobs=details['parallelism'] if details['parallelism'] != 0 else None
                    )
                models[algorithm] = model
        elif algorithm == "RandomForestRegressor":
                min_trees = details['min_trees']
                #print(min_trees)
                max_trees = details['max_trees']
                min_depth = details['min_depth']
                max_depth = details['max_depth']
                min_samples_per_leaf_min_value = details['min_samples_per_leaf_min_value']
                min_samples_per_leaf_max_value = details['min_samples_per_leaf_max_value']
                parallelism = details['parallelism']
                if details['feature_sampling_statergy'] == "Default":
                     max_features = 'auto'
                model = RandomForestRegressor(
                        n_estimators=min_trees, 
                        max_depth=max_depth, 
                        min_samples_split=min_samples_per_leaf_min_value, 
                        min_samples_leaf=min_samples_per_leaf_max_value, 
                        n_jobs=details['parallelism'] if details['parallelism'] != 0 else None
                )
                models[algorithm] = model
        elif algorithm == "GBTClassifier":
                num_of_boosting_stages = details['num_of_BoostingStages']
                feature_sampling_strategy = details['feature_sampling_statergy']
                learning_rate = details['learningRate']
                use_deviance = details['use_deviance']
                use_exponential = details['use_exponential']
                fixed_number = details['fixed_number']
                min_subsample = details['min_subsample']
                max_subsample = details['max_subsample']
                min_stepsize = details['min_stepsize']
                max_stepsize = details['max_stepsize']
                min_iter = details['min_iter']
                max_iter = details['max_iter']
                min_depth = details['min_depth']
                max_depth = details['max_depth']

                model = GradientBoostingClassifier(
                        n_estimators=num_of_boosting_stages,
                        learning_rate=learning_rate,
                        subsample=[min_subsample, max_subsample],
                        min_samples_split=feature_sampling_strategy if feature_sampling_strategy == "Fixed number" else None,
                        min_samples_leaf=fixed_number if feature_sampling_strategy == "Fixed number" else None,
                        min_weight_fraction_leaf=min_subsample,
                        max_depth=[min_depth, max_depth],
                        min_impurity_decrease=min_stepsize,
                        min_impurity_split=max_stepsize,
                        init=None,
                        random_state=None,
                        max_features=None,
                        verbose=0,
                        max_leaf_nodes=None,
                        warm_start=False,
                        presort="auto",
                        validation_fraction=0.1,
                        n_iter_no_change=None,
                        tol=1e-4,
                        ccp_alpha=0.0
                    )
                models[algorithm] = model
        elif algorithm == "GBTRegressor":
                
                num_of_boosting_stages = details['num_of_BoostingStages']
                feature_sampling_strategy = details['feature_sampling_statergy']
                use_deviance = details['use_deviance']
                use_exponential = details['use_exponential']
                fixed_number = details['fixed_number']
                min_subsample = details['min_subsample']
                max_subsample = details['max_subsample']
                min_stepsize = details['min_stepsize']
                max_stepsize = details['max_stepsize']
                min_iter = details['min_iter']
                max_iter = details['max_iter']
                min_depth = details['min_depth']
                max_depth = details['max_depth']

                model = GradientBoostingRegressor(
                        n_estimators=num_of_boosting_stages,
                        learning_rate=learning_rate,
                        subsample=[min_subsample, max_subsample],
                        min_samples_split=feature_sampling_strategy if feature_sampling_strategy == "Fixed number" else None,
                        min_samples_leaf=fixed_number if feature_sampling_strategy == "Fixed number" else None,
                        min_weight_fraction_leaf=min_subsample,
                        max_depth=[min_depth, max_depth],
                        min_impurity_decrease=min_stepsize,
                        min_impurity_split=max_stepsize,
                        init=None,
                        random_state=None,
                        loss='ls', 
                        verbose=0,
                        max_leaf_nodes=None,
                        warm_start=False,
                        presort="auto",
                        validation_fraction=0.1,
                        n_iter_no_change=None,
                        tol=1e-4,
                        ccp_alpha=0.0
                    )
                models[algorithm] = model
                
        elif algorithm == "SVM":
                    kernel ='linear' if details['linear_kernel'] else \
                        'rbf' if details['rep_kernel'] else \
                        'poly' if details['polynomial_kernel'] else \
                        'sigmoid'
    
                    C_values = details['c_value']
                    auto = details['auto']
                    scale = details['scale']
                    custom_gamma_values = details['custom_gamma_values']
                    tolerance = details['tolerance']
                    max_iterations = details['max_iterations']
                    model = SVC(
                                C=max(C_values),
                                kernel=kernel,
                                degree=3, 
                                gamma='scale' if scale else 'auto' if auto else 'scale' if custom_gamma_values else 'auto',
                                coef0=0.0,  
                                shrinking=True, 
                                probability=False, 
                                tol=tolerance,
                                cache_size=200, 
                                class_weight=None, 
                                verbose=False,  
                                max_iter=max_iterations,
                                decision_function_shape='ovr',  
                                break_ties=False,  
                                random_state=0,  
                            )
                    models[algorithm] = model
                    
        elif algorithm == "LinearRegression":
                
                parallelism = details['parallelism']
                min_iter = details['min_iter']
                max_iter = details['max_iter']
                min_regparam = details['min_regparam']
                max_regparam = details['max_regparam']
                min_elasticnet = details['min_elasticnet']
                max_elasticnet = details['max_elasticnet']

                model = LinearRegression(
                        fit_intercept=True,
                        normalize=False,
                        copy_X=True,
                        n_jobs=parallelism,
                    )
                models[algorithm] = model
                        
        elif algorithm == "LogisticRegression":
                
                parallelism = details['parallelism']
                min_iter = details['min_iter']
                max_iter = details['max_iter']
                min_regparam = details['min_regparam']
                max_regparam = details['max_regparam']
                min_elasticnet = details['min_elasticnet']
                max_elasticnet = details['max_elasticnet']

                model = LogisticRegression(
                        penalty='elasticnet',
                        solver='saga',
                        fit_intercept=True,
                        max_iter=max_iter,
                        tol=1e-4,
                        random_state=0,
                        n_jobs=parallelism,
                    )
                models[algorithm] = model
                
        elif algorithm == "RidgeRegression":
                min_iter = details['min_iter']
                max_iter = details['max_iter']
                min_regparam = details['min_regparam']
                max_regparam = details['max_regparam']
                alpha = details['regularization_term']

                model = Ridge(
                        alpha=alpha,
                        solver='auto', 
                        max_iter=max_iter,
                        tol=1e-4,
                        random_state=0
                    )
                models[algorithm] = model
        
        elif algorithm == "LassoRegression":
                
                min_iter = details['min_iter']
                max_iter = details['max_iter']
                min_regparam = details['min_regparam']
                max_regparam = details['max_regparam']
                alpha = details['regularization_term']

                model = Lasso(
                        alpha=alpha,
                        max_iter=max_iter,
                        tol=1e-4,
                        random_state=0
                    )
                models[algorithm] = model
        
        elif algorithm == "ElasticNetRegression":
                min_iter = details['min_iter']
                max_iter = details['max_iter']
                min_regparam = details['min_regparam']
                max_regparam = details['max_regparam']
                min_elasticnet = details['min_elasticnet']
                max_elasticnet = details['max_elasticnet']
                alpha = details['regularization_term']

                model = ElasticNet(
                        max_iter=max_iter,
                        alpha=alpha,
                        l1_ratio=0.5,
                        random_state=0
                    )
                models[algorithm] = model

        elif algorithm == "xg_boost":
                
                booster = 'gbtree' if details['use_gradient_boosted_tree'] else 'dart'
                tree_method = details['tree_method'] if details['tree_method'] else 'auto'
                max_depth = max(details['max_depth_of_tree'])
                learning_rate = min(details['learningRate'])
                alpha = max(details['l1_regularization'])
                lambd = max(details['l2_regularization'])
                gamma = max(details['gamma'])
                min_child_weight = max(details['min_child_weight'])
                subsample = max(details['sub_sample'])
                colsample_bytree = max(details['col_sample_by_tree'])
                n_estimators = details['max_num_of_trees']
                early_stopping_rounds = details['early_stopping_rounds'] if details['early_stopping'] else None
                random_state = details['random_state']

                model = xgb.XGBClassifier(
                    booster=booster,
                    tree_method=tree_method,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    alpha=alpha,
                    lambd=lambd,
                    gamma=gamma,
                    min_child_weight=min_child_weight,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    n_estimators=n_estimators,
                    early_stopping_rounds=early_stopping_rounds,
                    random_state=random_state
                )
                models[algorithm] = model
        elif algorithm == "DecisionTreeRegressor":
                min_depth = details['min_depth']
                max_depth = details['max_depth']
                use_gini = details['use_gini']
                use_entropy = details['use_entropy']
                min_samples_per_leaf = max(details['min_samples_per_leaf'])
                use_best = details['use_best']
                use_random = details['use_random']
                model = DecisionTreeRegressor(
                        criterion='gini' if use_gini else 'entropy',
                        splitter='best' if use_best else 'random',
                        max_depth=max_depth,
                        min_samples_split=2,  
                        min_samples_leaf=min_samples_per_leaf,
                        min_weight_fraction_leaf=0.0,  
                        max_features=None, 
                        random_state=0,  
                        max_leaf_nodes=None,  
                        min_impurity_decrease=0.0,  
                        min_impurity_split=None,  
                        presort='auto', 
                    )
                models[algorithm] = model
        elif algorithm == "DecisionTreeClassifier":
                
                min_depth = details['min_depth']
                max_depth = details['max_depth']
                use_gini = details['use_gini']
                use_entropy = details['use_entropy']
                min_samples_per_leaf = max(details['min_samples_per_leaf'])
                use_best = details['use_best']
                use_random = details['use_random']

                model = DecisionTreeClassifier(
                        criterion='gini' if use_gini else 'entropy',
                        splitter='best' if use_best else 'random',
                        max_depth=max_depth,
                        min_samples_split=2,  
                        min_samples_leaf=min_samples_per_leaf,
                        min_weight_fraction_leaf=0.0,  
                        max_features=None,  
                        random_state=0,  
                        max_leaf_nodes=None,  
                        min_impurity_decrease=0.0,  
                        min_impurity_split=None,  
                        class_weight=None,  
                        presort='auto', 
                    )
                models[algorithm] = model
        
        elif algorithm == "SGD":
            loss = 'log' if details['use_logistics'] else \
           'modified_huber' if details['use_modified_hubber_loss'] else \
           'squared_loss'
    
            penalty = []
            if details['use_l1_regularization']:
                penalty.append('l1')
            if details['use_l2_regularization']:
                penalty.append('l2')
            if details['use_elastic_net_regularization']:
                penalty.append('elasticnet')
    
            alpha_values = details['alpha_value']
            max_iterations = details['max_iterations']
            tolerance = details['tolerance']
            parallelism = details['parallelism']
    
            if details['use_logistics']:
                model = SGDClassifier(
                    loss=loss,
                    penalty=penalty,
                    alpha=max(alpha_values),
                    l1_ratio=0.15, 
                    fit_intercept=True,  
                    max_iter=max_iterations,
                    tol=tolerance,
                    shuffle=True,  
                    verbose=0,  
                    epsilon=0.1,  
                    n_jobs=parallelism,
                    random_state=0,  
                    learning_rate='optimal', 
                    eta0=0.0,  
                    power_t=0.5, 
                    early_stopping=False,  
                    validation_fraction=0.1,  
                    n_iter_no_change=5,  
                    class_weight=None,  
                    warm_start=False,  
                    average=False,  
                )
                models[algorithm] = model
            else:
                model = SGDRegressor(
                    loss=loss,
                    penalty=penalty,
                    alpha=max(alpha_values),
                    l1_ratio=0.15,  
                    fit_intercept=True, 
                    max_iter=max_iterations,
                    tol=tolerance,
                    shuffle=True,  
                    verbose=0,  
                    epsilon=0.1,  
                    learning_rate='invscaling',  
                    eta0=0.01, 
                    power_t=0.25, 
                    early_stopping=False,  
                    validation_fraction=0.1,  
                    n_iter_no_change=5, 
                    average=False, 
                )

                models[algorithm] = model
        
        elif algorithm == "KNN":
            if details['distance_weighting']:
                weights = 'distance'
            else:
                weights = 'uniform'

            if details['neighbour_finding_algorithm'] == "Automatic":
                algorithm = 'auto'
            else:
                algorithm = details['neighbour_finding_algorithm']

            if prediction_type == "Regression":
                model = KNeighborsRegressor(
                    n_neighbors=details['k_value'][0],
                    weights=weights,
                    algorithm=algorithm,
                    leaf_size=30,  
                    p=details['p_value'],
                    metric='minkowski',  
                    metric_params=None,  
                    n_jobs=None, 
                )
                models[algorithm] = model
            else: 
                model = KNeighborsClassifier(
                    n_neighbors=details['k_value'][0],
                    weights=weights,
                    algorithm=algorithm,
                    leaf_size=30,  
                    p=details['p_value'],
                    metric='minkowski',  
                    metric_params=None,  
                    n_jobs=None, 
                )
                models[algorithm] = model
        
        elif algorithm == "extra_random_trees":

            if details['feature_sampling_statergy'] == "Square root":
                max_features = 'sqrt'
            elif details['feature_sampling_statergy'] == "Logarithm":
                max_features = 'log2'
            else:
                max_features = 'auto' 

            if prediction_type == "Regression":
                model = ExtraTreesRegressor(
                    n_estimators=max(details['num_of_trees']),
                    max_features=max_features,
                    max_depth=max(details['max_depth']),
                    min_samples_leaf=min(details['min_samples_per_leaf']),
                    n_jobs=details['parallelism']
                )
                models[algorithm] = model
            else: 
                model = ExtraTreesClassifier(
                    n_estimators=max(details['num_of_trees']),
                    max_features=max_features,
                    max_depth=max(details['max_depth']),
                    min_samples_leaf=min(details['min_samples_per_leaf']),
                    n_jobs=details['parallelism']
                )
                models[algorithm] = model
        
        elif algorithm == "neural_network":
            hidden_layer_sizes = tuple(details['hidden_layer_sizes'])
            activation = details['activation'] if details['activation'] else 'relu' 
            alpha = details['alpha_value']
            max_iter = details['max_iterations']
            tol = details['convergence_tolerance']
            early_stopping = details['early_stopping']
            solver = 'adam' if details['solver'].lower() == 'adam' else 'sgd'
            shuffle = details['shuffle_data']
            learning_rate_init = details['initial_learning_rate']
            beta_1 = details['beta_1']
            beta_2 = details['beta_2']
            epsilon = details['epsilon']
            power_t = details['power_t']
            momentum = details['momentum']
            nesterovs_momentum = details['use_nesterov_momentum']
            if prediction_type == "Regression":
                model = MLPRegressor(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation=activation,
                    alpha=alpha,
                    max_iter=max_iter,
                    tol=tol,
                    early_stopping=early_stopping,
                    solver=solver,
                    shuffle=shuffle,
                    learning_rate_init=learning_rate_init,
                    beta_1=beta_1,
                    beta_2=beta_2,
                    epsilon=epsilon,
                    power_t=power_t,
                    momentum=momentum,
                    nesterovs_momentum=nesterovs_momentum
                )
                models[algorithm] = model
            else: 
                model = MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation=activation,
                    alpha=alpha,
                    max_iter=max_iter,
                    tol=tol,
                    early_stopping=early_stopping,
                    solver=solver,
                    shuffle=shuffle,
                    learning_rate_init=learning_rate_init,
                    beta_1=beta_1,
                    beta_2=beta_2,
                    epsilon=epsilon,
                    power_t=power_t,
                    momentum=momentum,
                    nesterovs_momentum=nesterovs_momentum
                )
                models[algorithm] = model

        return model

algorithms = data['design_state_data']['algorithms']




from sklearn.model_selection import GridSearchCV , TimeSeriesSplit
from sklearn.model_selection import train_test_split
# testing whether reduction is done properly
partial_pipeline = Pipeline([
    ('feature_reduction', FeatureReductionTransformer(data))
])

reduced_df = partial_pipeline.transform(df_transformed)

print(reduced_df.head())



X = reduced_df.drop(target, axis=1) 
y = reduced_df[target]
train = data["design_state_data"]["train"]

train_ratio = 0.7 if train["train_ratio"] == 0 else train["train_ratio"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_ratio, random_state=train["random_seed"])

hyperparameters = data['design_state_data']['hyperparameters']

tscv = TimeSeriesSplit(n_splits=hyperparameters['num_of_folds'])

param_grids = {
    "RandomForestClassifier": {
        "n_estimators": list(range(10, 31, 10)),
        "max_depth": list(range(20, 31, 5)),
        "min_samples_leaf": list(range(5, 51, 5))
    },
    "RandomForestRegressor": {
        "n_estimators": list(range(10, 21, 10)),
        "max_depth": list(range(20, 26, 5)),
        "min_samples_leaf": list(range(5, 11, 5))
    },
    "GBTClassifier": {
        "n_estimators": [67, 89],
        "learning_rate": [0.1, 0.5],
        "max_depth": list(range(5, 8)),
        "subsample": [1, 2]
    },
    "GBTRegressor": {
        "n_estimators": [67, 89],
        "learning_rate": [0.1, 0.5], 
        "max_depth": list(range(5, 8)),
        "subsample": [1, 2]
    },
    "LinearRegression": {
        "max_iter": list(range(30, 51, 10))
    },
    "LogisticRegression": {
        "max_iter": list(range(30, 51, 10)),
        "C": [0.5, 0.8]
    },
    "RidgeRegression": {
        "alpha": [0.5, 0.8],
        "max_iter": list(range(30, 51, 10))
    },
    "LassoRegression": {
        "alpha": [0.5, 0.8],
        "max_iter": list(range(30, 51, 10))
    },
    "ElasticNetRegression": {
        "alpha": [0.5, 0.8],
        "l1_ratio": [0.5, 0.8],
        "max_iter": list(range(30, 51, 10))
    },
    "XGBoost": {
        "max_depth": [56, 89],
        "learning_rate": [89, 76], 
        "gamma": [68]
    },
    "DecisionTreeRegressor": {
        "max_depth": list(range(4, 8)),
        "min_samples_leaf": [12, 6]
    },
    "DecisionTreeClassifier": {
        "max_depth": list(range(4, 8)),
        "min_samples_leaf": [12, 6]
    },
    "SVM": {
        "C": [566, 79],
        "kernel": ["linear", "rbf", "poly", "sigmoid"]
    },
    "SGD": {
        "max_iter": [1000],
        "tol": [1e-3], 
        "penalty": ["l1", "l2", "elasticnet"]
    },
    "KNN": {
        "n_neighbors": [78],
        "weights": ["uniform", "distance"]
    },
    "ExtraTreesClassifier": {
        "n_estimators": [45, 489],
        "max_depth": [12, 45],
        "min_samples_leaf": [78, 56]
    }
}



from sklearn.model_selection import StratifiedKFold, KFold

def create_pipeline_with_grid_search(algorithm, details, param_grid, prediction_type):
    model = create_model(algorithm, details)
    if prediction_type == 'Classification':
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=hyperparameters['random_state'])
        scoring_metric = 'roc_auc'
    elif prediction_type == 'Regression':
        cv = KFold(n_splits=5, shuffle=True, random_state=hyperparameters['random_state'])
        scoring_metric = 'neg_mean_squared_error'
    else:
        raise ValueError("Invalid prediction_type. Expected 'classification' or 'regression'.")
    
    grid_search = GridSearchCV(
        model, 
        param_grid, 
        cv=cv, 
        n_jobs=hyperparameters['parallelism'],
        verbose=1,
        scoring=scoring_metric,  
        return_train_score=True,
    )
    
    return Pipeline([
        ('grid_search', grid_search)
    ])
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
from sklearn.metrics import accuracy_score


def evaluate_model(best_model, X_train, y_train, X_test, y_test, prediction_type, data):
    predictions = best_model.predict(X_test)
    y_probs = predictions
    print("Hyperparameters:")
    print(best_model.get_params)
    
    if prediction_type == 'Classification':
        train_predictions = best_model.predict(X_train)
        test_predictions = best_model.predict(X_test)

        train_accuracy = accuracy_score(y_train, train_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)

        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        auc = roc_auc_score(y_test, y_probs)
        print(f"AUC: {auc:.4f}")


        precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
        f1_scores = 2*(precision*recall)/(precision+recall)
        best_threshold = thresholds[np.argmax(f1_scores)]
        print(f"Best Threshold for F1 Score: {best_threshold:.4f}")

        y_pred = (y_probs > best_threshold).astype(int)
        cost_matrix = data['design_state_data']['metrics']
        cost = (
            cost_matrix['cost_matrix_gain_for_true_prediction_true_result'] * np.sum((y_pred == 1) & (y_test == 1)) +
            cost_matrix['cost_matrix_gain_for_true_prediction_false_result'] * np.sum((y_pred == 1) & (y_test == 0)) +
            cost_matrix['cost_matrix_gain_for_false_prediction_true_result'] * np.sum((y_pred == 0) & (y_test == 1)) +
            cost_matrix['cost_matrix_gain_for_false_prediction_false_result'] * np.sum((y_pred == 0) & (y_test == 0))
        )
        print(f"Total Cost: {cost}")
    else:
        mae = mean_absolute_error(y_test, y_probs)
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        
        mse = mean_squared_error(y_test, y_probs)
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        
        rmse = mean_squared_error(y_test, y_probs, squared=False)
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        
        r2 = r2_score(y_test, y_probs)
        print(f"R-squared: {r2:.2f}")



selected_model_name = next((key for key, value in algorithms.items() if value["is_selected"]), None)

pipelines = {}

for algorithm, details in algorithms.items():
    if details['is_selected']:
        pipelines[algorithm] = create_pipeline_with_grid_search(algorithm, details, param_grids[algorithm], prediction_type)

selected_model_name = next((key for key, value in algorithms.items() if value["is_selected"]), None)
selected_pipeline = pipelines[selected_model_name]
selected_pipeline.fit(X_train, y_train)


best_model = selected_pipeline.named_steps['grid_search'].best_estimator_
evaluate_model(best_model, X_train, y_train, X_test, y_test, prediction_type, data)


