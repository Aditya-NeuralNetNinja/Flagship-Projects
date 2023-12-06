

Imports

    import numpy as np 
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings('ignore')

Load data

    df = pd.read_csv('/kaggle/input/laptop/laptop.csv.txt')

EDA

    df.head()

    df.sample(5)

    df.shape

    df.info()

    df.duplicated().sum()

    df.isnull().sum()

    #drop column
    df.drop(columns=['Unnamed: 0'],inplace=True)

    df.head()

Feature Engineering

Removing units from Ram & Weight

    df['Ram']=df['Ram'].str.replace('GB','')
    df['Weight']=df['Weight'].str.replace('kg','')

    df.tail()

Typecasting

    #converting columns to suitable datatype

    df['Ram']=df['Ram'].astype('int32')
    df['Weight']=df['Weight'].astype('float32')

    df.info()

    #checking price distribution, target variable distribution looks normal

    sns.histplot(df['Price'],kde=True)

    df['Company'].value_counts()

    df['Company'].value_counts().plot(kind='bar')

    sns.countplot(x=df['Company'],order=df['Company'].value_counts().index)
    plt.xticks(rotation='90')
    plt.show()

Comparing Price & Companies

    #Bar is median, line is std. deviation
    sns.barplot(x=df['Company'],y=df['Price'],estimator='median',order=df['Company'].value_counts().index)
    plt.xticks(rotation='vertical')
    plt.show()

Checking which type are maximum laptops

    sns.countplot(x=df['TypeName'],order=df['TypeName'].value_counts().index)
    plt.xticks(rotation=90)

Checking laptop type price distribution

    sns.barplot(x=df['TypeName'],y=df['Price'], estimator="mean")
    plt.xticks(rotation='vertical')
    plt.show()

Checking screen size distribution

    sns.histplot(df['Inches'],kde=True)

Screen size vs. Price

    sns.scatterplot(x=df['Inches'],y=df['Price'])

Screen resolution count

    df['ScreenResolution'].value_counts()

Creating new feature Touchscreen from ScreenResolution

    df['Touchscreen']=df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)

    # View the newly created column df['Touchscreen']
    df.sample(5)

Touch vs Non-touchscreen laptops

    df['Touchscreen'].value_counts().plot(kind='bar')

Touchscreen laptops are costlier than non-touch

    sns.barplot(x=df['Touchscreen'],y=df['Price'])

Generating a new feature IPS display - yes or no

    df['Ips'] = df['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)

    df.head()

More non-IPS displays

    df['Ips'].value_counts().plot(kind='bar')

Ips display is costlier

    sns.barplot(x=df['Ips'],y=df['Price'])

    df["ScreenResolution"]

    new=df['ScreenResolution'].str.split('x',n=1,expand=True)

    new.head()

    df['X_res'] = new[0]
    df['Y_res'] = new[1]

    df.sample(5)

Extracting number from whole text

    df['X_res'] = df['X_res'].str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0])

    df.sample(5)

    df["X_res"]

Typecasting X_res, Y_res

    df['X_res'] = df['X_res'].astype('int')
    df['Y_res'] = df['Y_res'].astype('int')

    df.info()

Check input features' correlation with Price

    # Feature importance
    df.corr()['Price']

Making more efficient feature - Pixels Per Inch

PPI = (sqrt(X_res^2 + Y_res^2) / Inches)

    df['ppi'] = (((df['X_res']**2) + (df['Y_res']**2))**0.5/df['Inches']).astype('float')

    df.head()

    df.drop(columns=['ScreenResolution'],inplace=True)

    #Feature importance
    df.corr()['Price']

    #Dropping below features because we've created ppi

    df.drop(columns=['Inches','X_res','Y_res'],inplace=True)

    df.head()

Tackling feature Cpu

    df['Cpu'].value_counts()

    df['CPU Name']=df['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))

    df.head()

    df.info()

    def fetch_processor(text):
        if text=='Intel Core i3' or text=='Intel Core i5' or text=='Intel Core i7':
            return text
        else:
            if text.split()[0]=='Intel':
                return 'Intel'
            if text.split()[0]=='Samsung':
                return 'Samsung'
            else:
                return 'AMD'

    df['CPU brand']=df['CPU Name'].apply(fetch_processor)

    df.sample(5)

Barplot of various CPU brands

    df['CPU brand'].value_counts().plot(kind='bar')

    sns.barplot(x=df['CPU brand'],y=df['Price'])
    plt.xticks(rotation='vertical')
    plt.show()

    #extract ghz 
    df['GHz'] = df['Cpu'].apply(lambda x: x.split()[-1].replace('GHz',''))

More powerful the Cpu, higher is the price

    sns.barplot(x=df['GHz'],y=df['Price'])
    plt.xticks(rotation='vertical')
    plt.show()

    df.drop(columns =['CPU Name','Cpu'],inplace=True)

Barplot of RAM

    #8gb is the most selling one
    df['Ram'].value_counts().plot(kind='bar')

    df['Ram'].unique()

Higher the RAM, more is the Price

    sns.barplot(x=df['Ram'],y=df['Price'])
    plt.xticks(rotation='vertical')
    plt.show()

    df.head()

    df['Memory'].value_counts()

    #Regex code to eliminate decimals

    df['Memory'] = df['Memory'].astype(str)
    df['Memory'] = df["Memory"].str.replace('/.0', '', regex=True)

    df["Memory"].value_counts()

Making all units common to GB

    df["Memory"] = df["Memory"].str.replace('GB', '')
    df["Memory"] = df["Memory"].str.replace('TB', '000')

    df["Memory"].unique()

Splitting Memory column into 2 separate columns (SSD & HDD)

    new = df["Memory"].str.split("+", n = 1, expand=True)

    new

    df["first"]= new[0]
    df["first"]=df["first"].str.strip()

    df["second"]= new[1]

    df["first"]

    df["first"].unique()

    df["second"].unique()

Creating 4 new columns

    df['Layer1HDD']=df['first'].apply(lambda x:1 if 'HDD' in x else 0)
    df['Layer1SSD']=df['first'].apply(lambda x:1 if 'SSD' in x else 0)
    df['Layer1Hybrid']=df['first'].apply(lambda x:1 if 'Hybrid' in x else 0)
    df["Layer1Flash_Storage"] = df["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)

    df.head()

    df['first'] = df['first'].str.replace(r'\D', '', regex=True)

    df["second"].fillna("0", inplace = True)

Creating another 4 new columns

    df["Layer2HDD"] = df["second"].apply(lambda x: 1 if "HDD" in x else 0)
    df["Layer2SSD"] = df["second"].apply(lambda x: 1 if "SSD" in x else 0)
    df["Layer2Hybrid"] = df["second"].apply(lambda x: 1 if "Hybrid" in x else 0)
    df["Layer2Flash_Storage"] = df["second"].apply(lambda x: 1 if "Flash Storage" in x else 0)

    df['second'] = df['second'].str.replace(r'\D', '', regex=True)

    df["first"] = df["first"].astype(int)
    df["second"] = df["second"].astype(int)

    df["first"]

    df["second"]

    df["Layer1HDD"]

Creating the final columns for memory

    df["HDD"]=(df["first"]*df["Layer1HDD"]+df["second"]*df["Layer2HDD"])
    df["SSD"]=(df["first"]*df["Layer1SSD"]+df["second"]*df["Layer2SSD"])
    df["Hybrid"]=(df["first"]*df["Layer1Hybrid"]+df["second"]*df["Layer2Hybrid"])
    df["Flash_Storage"]=(df["first"]*df["Layer1Flash_Storage"]+df["second"]*df["Layer2Flash_Storage"])

    df.sample(5)

    #dropping the columns we made for working

    df.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid',
           'Layer1Flash_Storage', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid',
           'Layer2Flash_Storage'],inplace=True)

    df.drop(columns=['Memory'],inplace=True)

    df.corr()['Price']

    df['Hybrid'].unique()

    df['Hybrid'].value_counts()

    df['Flash_Storage'].value_counts()

    df.drop(columns=['Hybrid','Flash_Storage'],inplace=True)

    df.corr()['Price']

Tackling GPU column

    df['Gpu'].value_counts()

    df['Gpu brand'] = df['Gpu'].apply(lambda x:x.split()[0])

    df['Gpu brand'].value_counts()

Remove all rows where Gpu brand is ARM

    df = df[df["Gpu brand"].str.contains("ARM") == False]

    df["Gpu brand"]

    df['Gpu brand'].value_counts()

Gpu brand vs Price Barplot

    sns.barplot(x=df['Gpu brand'],y=df['Price'])
    plt.xticks(rotation='vertical')
    plt.show()

    df.drop(columns=['Gpu'],inplace=True)

Tackling OpSys feature

    df['OpSys'].value_counts()

    sns.barplot(x=df['OpSys'],y=df['Price'])
    plt.xticks(rotation='vertical')
    plt.show()

Creating popular OS categories

    def cat_os(inp):
        if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
            return 'Windows'
        elif inp == 'macOS' or inp == 'Mac OS X':
            return 'Mac'
        else:
            return 'Others/No OS/Linux'

    df['os'] = df['OpSys'].apply(cat_os)

    df.head()

    df.drop(columns=['OpSys'],inplace=True)

    sns.barplot(x=df['os'],y=df['Price'])
    plt.xticks(rotation='vertical')
    plt.show()

Tackling weight feature

    sns.histplot(df['Weight'],kde=True)

Weight vs Price

    sns.scatterplot(x=df['Weight'],y=df['Price'])

Pearson correlation coefficient

    df.corr()['Price'] 

    sns.heatmap(df.corr())

    sns.histplot(df["Price"],kde=True)

Applying log transformation to make Price distribution normal

    sns.histplot(np.log(df['Price']),kde=True)

Separating input features from labels

    X = df.drop(columns=['Price'])
    y = np.log(df['Price'])

    X

    y

Squeezing y values between 0 or 1

    y = y/10

    y

    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=2)

    X_train

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

Linear regression

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

Ridge Regression

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

Lasso Regression

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

KNN

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

Decision Tree

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

SVM

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

Random Forest

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

ExtraTrees

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

AdaBoost

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

Gradient Boost

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

XGBoost

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

LightGBM

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

Catboost

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

Voting Regressor

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

Stacking Regressor

    from sklearn.ensemble import VotingRegressor,StackingRegressor

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

Highest R2 score: Stacking Regressor Model

Exporting Stacking Regressor Model

    import pickle

    pickle.dump(df,open('df.pkl','wb'))
    pickle.dump(pipe,open('pipe.pkl','wb'))

    X_test.sample()

Testing model on input example

[Screenshot (2276).jpg]

    input_example = [["Apple","Ultrabook",8,1.37,0,1,227,"Intel Core i5",1.8,0,128,"Intel","Mac"]]

    y_pred = pipe.predict(input_example)

    # predicted price = e^(predicted price *10)
    # taking inverse transform of (log and division by 10)

    predicted_price = np.exp(y_pred*10)
    print("The predicted price for your laptop is: Rs.",int(predicted_price))
