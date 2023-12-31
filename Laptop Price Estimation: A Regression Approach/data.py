# Imports

    import numpy as np 
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings('ignore')

# Load data

    df = pd.read_csv('/kaggle/input/laptop/laptop.csv.txt')

# EDA

    df.head()

    df.sample(5)

    df.shape

    df.info()

    df.duplicated().sum()

    df.isnull().sum()

    df.drop(columns=['Unnamed: 0'],inplace=True)

    df.head()
