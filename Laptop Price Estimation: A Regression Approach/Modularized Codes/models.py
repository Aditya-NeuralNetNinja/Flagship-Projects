# Train-test split
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=2)

    X_train

# Imports 
    from sklearn.linear_model import LinearRegression,Ridge,Lasso
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
    from sklearn.svm import SVR
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor

    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.metrics import r2_score, mean_absolute_error
    from sklearn.datasets import make_regression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, VotingRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from sklearn.svm import SVR
    from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Linear regression

    step1 = ColumnTransformer(transformers=[
        ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,11,12])
    ],remainder='passthrough')

    step2 = LinearRegression()

    pipe = Pipeline([
        ('step1',step1),
        ('step2',step2)
    ])

    pipe.fit(X_train,y_train)

    y_pred = pipe.predict(X_test)

    print('R2 score',r2_score(y_test,y_pred))
    print('MAE',mean_absolute_error(y_test,y_pred))

# Ridge Regression

    step1 = ColumnTransformer(transformers=[
        ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,11,12])
    ],remainder='passthrough')

    step2 = Ridge(alpha=9,solver='auto',random_state=0)

    pipe = Pipeline([
        ('step1',step1),
        ('step2',step2)
    ])

    pipe.fit(X_train,y_train)

    y_pred = pipe.predict(X_test)

    print('R2 score',r2_score(y_test,y_pred))
    print('MAE',mean_absolute_error(y_test,y_pred))

# Lasso Regression

    step1 = ColumnTransformer(transformers=[
        ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,11,12])
    ],remainder='passthrough')

    step2 = Lasso(alpha=0.00001,max_iter=100)

    pipe = Pipeline([
        ('step1',step1),
        ('step2',step2)
    ])

    pipe.fit(X_train,y_train)

    y_pred = pipe.predict(X_test)

    print('R2 score',r2_score(y_test,y_pred))
    print('MAE',mean_absolute_error(y_test,y_pred))

# KNN

    step1 = ColumnTransformer(transformers=[
        ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,11,12])
    ],remainder='passthrough')

    step2 = KNeighborsRegressor(n_neighbors=12,weights='distance',p=1)

    pipe = Pipeline([
        ('step1',step1),
        ('step2',step2)
    ])

    pipe.fit(X_train,y_train)

    y_pred = pipe.predict(X_test)

    print('R2 score',r2_score(y_test,y_pred))
    print('MAE',mean_absolute_error(y_test,y_pred))

# Decision Tree

    step1 = ColumnTransformer(transformers=[
        ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,11,12])
    ],remainder='passthrough')

    step2 = DecisionTreeRegressor(max_depth=20,
                                  splitter='best',
                                  min_samples_leaf=7,
                                  min_weight_fraction_leaf=0,
                                  max_leaf_nodes=120,
                                  ccp_alpha=0,
                                  random_state=0)

    pipe = Pipeline([
        ('step1',step1),
        ('step2',step2)
    ])

    pipe.fit(X_train,y_train)

    y_pred = pipe.predict(X_test)

    print('R2 score',r2_score(y_test,y_pred))
    print('MAE',mean_absolute_error(y_test,y_pred))

# SVM

    step1 = ColumnTransformer(transformers=[
        ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,11,12])
    ],remainder='passthrough')

    step2 = SVR(kernel='rbf',epsilon=0.01,C=1400,gamma='scale')

    pipe = Pipeline([
        ('step1',step1),
        ('step2',step2)
    ])

    pipe.fit(X_train,y_train)

    y_pred = pipe.predict(X_test)

    print('R2 score',r2_score(y_test,y_pred))
    print('MAE',mean_absolute_error(y_test,y_pred))

# Random Forest

    step1 = ColumnTransformer(transformers=[
        ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,11,12])
    ],remainder='passthrough')

    step2 = RandomForestRegressor(n_estimators=155,
                                  random_state=3,
                                  max_samples=0.5,
                                  max_features=0.15,
                                  max_depth=15)

    pipe = Pipeline([
        ('step1',step1),
        ('step2',step2)
    ])

    pipe.fit(X_train,y_train)

    y_pred = pipe.predict(X_test)

    print('R2 score',r2_score(y_test,y_pred))
    print('MAE',mean_absolute_error(y_test,y_pred))

# ExtraTrees

    step1 = ColumnTransformer(transformers=[
        ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,11,12])
    ],remainder='passthrough')

    step2 = ExtraTreesRegressor(n_estimators=95,
                                  random_state=3,
                                  max_features=0.6,
                                  max_depth=14)

    pipe = Pipeline([
        ('step1',step1),
        ('step2',step2)
    ])

    pipe.fit(X_train,y_train)

    y_pred = pipe.predict(X_test)

    print('R2 score',r2_score(y_test,y_pred))
    print('MAE',mean_absolute_error(y_test,y_pred))

# AdaBoost

    step1 = ColumnTransformer(transformers=[
        ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,11,12])
    ],remainder='passthrough')
    step2 = AdaBoostRegressor(n_estimators=100,learning_rate=0.2,random_state=0)

    pipe = Pipeline([
        ('step1',step1),
        ('step2',step2)
    ])

    pipe.fit(X_train,y_train)

    y_pred = pipe.predict(X_test)

    print('R2 score',r2_score(y_test,y_pred))
    print('MAE',mean_absolute_error(y_test,y_pred))

# Gradient Boost

    step1 = ColumnTransformer(transformers=[
        ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,11,12])
    ],remainder='passthrough')

    step2 = GradientBoostingRegressor(n_estimators=1000,
                                      max_depth=8,
                                      min_samples_leaf=100,
                                      random_state=0)

    pipe = Pipeline([
        ('step1',step1),
        ('step2',step2)
    ])

    pipe.fit(X_train,y_train)

    y_pred = pipe.predict(X_test)

    print('R2 score',r2_score(y_test,y_pred))
    print('MAE',mean_absolute_error(y_test,y_pred))

# XGBoost

    step1 = ColumnTransformer(transformers=[
        ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,11,12])
    ],remainder='passthrough')

    step2 = XGBRegressor(n_estimators=45,
                         max_depth=5,
                         learning_rate=0.5,
                         random_state=0)

    pipe = Pipeline([
        ('step1',step1),
        ('step2',step2)
    ])

    pipe.fit(X_train,y_train)

    y_pred = pipe.predict(X_test)

    print('R2 score',r2_score(y_test,y_pred))
    print('MAE',mean_absolute_error(y_test,y_pred))

# LightGBM

    step1 = ColumnTransformer(transformers=[(
    'ohe',OneHotEncoder(sparse=False,drop='first'),[0,1,7,11,12])],remainder='passthrough')

    step2 = LGBMRegressor(learning_rate=0.5,
                          n_estimators=100,
                          max_depth=6,
                          random_state=0)

    pipe = Pipeline([
        ('step1',step1),
        ('step2',step2) 
    ])

    pipe.fit(X_train,y_train)

    y_pred = pipe.predict(X_test)

    print(f'R2 score {r2_score(y_test,y_pred)}')
    print(f'MAE {mean_absolute_error(y_test,y_pred)}')

# Catboost

    step1 = ColumnTransformer(transformers=[(
    'ohe',OneHotEncoder(sparse=False,drop='first'),[0,1,7,11,12])],remainder='passthrough')

    step2 = CatBoostRegressor(iterations=200,
                              learning_rate=0.2,
                              depth=15,
                              random_state=0,
                              verbose=False)

    pipe = Pipeline([
        ('step1',step1),
        ('step2',step2)
    ])

    pipe.fit(X_train,y_train)

    y_pred = pipe.predict(X_test)

    print(f'R2 score {r2_score(y_test,y_pred)}')
    print(f'MAE {mean_absolute_error(y_test,y_pred)}')

# Voting Regressor
from sklearn.ensemble import VotingRegressor,StackingRegressor

    step1 = ColumnTransformer(
        transformers=[
            ('col_tnf', OneHotEncoder(sparse=False, drop='first'), [0,1,7,11,12])
        ],
        remainder='passthrough'
    )

    rf = RandomForestRegressor(
        n_estimators=155,
        random_state=3,
        max_samples=0.5,
        max_features=0.15,
        max_depth=15
    )

    lgb = LGBMRegressor(
        learning_rate=0.5,
        n_estimators=100,
        max_depth=6,
        random_state=0
    )

    gbdt = GradientBoostingRegressor(
        n_estimators=1000,
        max_depth=8,
        min_samples_leaf=100,
        random_state=0
    )

    xgb = XGBRegressor(
        n_estimators=45,
        max_depth=5,
        learning_rate=0.5,
        random_state=0
    )

    et = ExtraTreesRegressor(
        n_estimators=95,
        random_state=3,
        max_features=0.6,
        max_depth=14
    )

    catb = CatBoostRegressor(
        iterations=200,
        learning_rate=0.2,
        depth=15,
        random_state=0,
        verbose=False
    )

    step2 = VotingRegressor([
        ('rf', rf), ('lgb', lgb), ('gbdt', gbdt), ('xgb', xgb), ('et', et), ('catb', catb)
    ], weights=[1, 2, 1, 1, 1, 2])

    pipe = Pipeline([
        ('step1', step1),
        ('step2', step2)
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    print('R2 score', r2_score(y_test, y_pred))
    print('MAE', mean_absolute_error(y_test, y_pred))

# Stacking Regressor

    step1 = ColumnTransformer(transformers=[
        ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,11,12])
    ],remainder='passthrough')


    estimators = [
        ('rf', RandomForestRegressor(n_estimators=155,
                                     random_state=3,
                                     max_samples=0.5,
                                     max_features=0.15,
                                     max_depth=15)),
        
        ('et',ExtraTreesRegressor(n_estimators=95,
                                  random_state=3,
                                  max_features=0.6,
                                  max_depth=14)),
        
        ('lgb',LGBMRegressor(learning_rate=0.5,
                             n_estimators=100,
                             max_depth=6,
                             random_state=0)),
       
        ('gbdt',GradientBoostingRegressor(n_estimators=1000,
                                          max_depth=8,
                                          min_samples_leaf=100,
                                          random_state=0)),
        
        ('xgb', XGBRegressor(n_estimators=45,
                             max_depth=5,
                             learning_rate=0.5,
                             random_state=0)),
        
        ('catb', CatBoostRegressor(iterations=200,
                                   learning_rate=0.2,
                                   depth=15,
                                   random_state=0,
                                   verbose=False))
    ]

    step2 = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=0.2))

    pipe = Pipeline([
        ('step1',step1),
        ('step2',step2)
    ])

    pipe.fit(X_train,y_train)

    y_pred = pipe.predict(X_test)

    print('R2 score',r2_score(y_test,y_pred))
    print('MAE',mean_absolute_error(y_test,y_pred))

# Highest R2 score: Stacking Regressor Model
