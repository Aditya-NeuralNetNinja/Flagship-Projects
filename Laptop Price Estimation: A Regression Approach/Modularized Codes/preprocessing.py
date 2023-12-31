# Feature Engineering

#Removing units from Ram & Weight

    df['Ram']=df['Ram'].str.replace('GB','')
    df['Weight']=df['Weight'].str.replace('kg','')

    df.tail()

#Typecasting

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

#Comparing Price & Companies

    #Bar is median, line is std. deviation
    sns.barplot(x=df['Company'],y=df['Price'],estimator='median',order=df['Company'].value_counts().index)
    plt.xticks(rotation='vertical')
    plt.show()

#Checking which type are maximum laptops

    sns.countplot(x=df['TypeName'],order=df['TypeName'].value_counts().index)
    plt.xticks(rotation=90)

#Checking laptop type price distribution

    sns.barplot(x=df['TypeName'],y=df['Price'], estimator="mean")
    plt.xticks(rotation='vertical')
    plt.show()

#Checking screen size distribution

    sns.histplot(df['Inches'],kde=True)

#Screen size vs. Price

    sns.scatterplot(x=df['Inches'],y=df['Price'])

#Screen resolution count

    df['ScreenResolution'].value_counts()

#Creating new feature Touchscreen from ScreenResolution

    df['Touchscreen']=df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)

    # View the newly created column df['Touchscreen']
    df.sample(5)

#Touch vs Non-touchscreen laptops

    df['Touchscreen'].value_counts().plot(kind='bar')

#Touchscreen laptops are costlier than non-touch

    sns.barplot(x=df['Touchscreen'],y=df['Price'])

#Generating a new feature IPS display - yes or no

    df['Ips'] = df['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)

    df.head()

#More non-IPS displays

    df['Ips'].value_counts().plot(kind='bar')

#Ips display is costlier

    sns.barplot(x=df['Ips'],y=df['Price'])

    df["ScreenResolution"]

    new=df['ScreenResolution'].str.split('x',n=1,expand=True)

    new.head()

    df['X_res'] = new[0]
    df['Y_res'] = new[1]

    df.sample(5)

#Extracting number from whole text

    df['X_res'] = df['X_res'].str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0])

    df.sample(5)

    df["X_res"]

#Typecasting X_res, Y_res

    df['X_res'] = df['X_res'].astype('int')
    df['Y_res'] = df['Y_res'].astype('int')

    df.info()

#Check input features' correlation with Price

    # Feature importance
    df.corr()['Price']

#Making more efficient feature - Pixels Per Inch

PPI = (sqrt(X_res^2 + Y_res^2) / Inches)

    df['ppi'] = (((df['X_res']**2) + (df['Y_res']**2))**0.5/df['Inches']).astype('float')

    df.head()

    df.drop(columns=['ScreenResolution'],inplace=True)

    #Feature importance
    df.corr()['Price']

    #Dropping below features because we've created ppi

    df.drop(columns=['Inches','X_res','Y_res'],inplace=True)

    df.head()

#Tackling feature Cpu

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

#Barplot of various CPU brands

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

#Barplot of RAM

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

#Making all units common to GB

    df["Memory"] = df["Memory"].str.replace('GB', '')
    df["Memory"] = df["Memory"].str.replace('TB', '000')

    df["Memory"].unique()

#Splitting Memory column into 2 separate columns (SSD & HDD)

    new = df["Memory"].str.split("+", n = 1, expand=True)

    new

    df["first"]= new[0]
    df["first"]=df["first"].str.strip()

    df["second"]= new[1]

    df["first"]

    df["first"].unique()

    df["second"].unique()

#Creating 4 new columns

    df['Layer1HDD']=df['first'].apply(lambda x:1 if 'HDD' in x else 0)
    df['Layer1SSD']=df['first'].apply(lambda x:1 if 'SSD' in x else 0)
    df['Layer1Hybrid']=df['first'].apply(lambda x:1 if 'Hybrid' in x else 0)
    df["Layer1Flash_Storage"] = df["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)

    df.head()

    df['first'] = df['first'].str.replace(r'\D', '', regex=True)

    df["second"].fillna("0", inplace = True)

#Creating another 4 new columns

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

#Creating the final columns for memory

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

#Tackling GPU column

    df['Gpu'].value_counts()

    df['Gpu brand'] = df['Gpu'].apply(lambda x:x.split()[0])

    df['Gpu brand'].value_counts()

#Remove all rows where Gpu brand is ARM

    df = df[df["Gpu brand"].str.contains("ARM") == False]

    df["Gpu brand"]

    df['Gpu brand'].value_counts()

#Gpu brand vs Price Barplot

    sns.barplot(x=df['Gpu brand'],y=df['Price'])
    plt.xticks(rotation='vertical')
    plt.show()

    df.drop(columns=['Gpu'],inplace=True)

#Tackling OpSys feature

    df['OpSys'].value_counts()

    sns.barplot(x=df['OpSys'],y=df['Price'])
    plt.xticks(rotation='vertical')
    plt.show()

#Creating popular OS categories

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

#Tackling weight feature

    sns.histplot(df['Weight'],kde=True)

#Weight vs Price

    sns.scatterplot(x=df['Weight'],y=df['Price'])

#Pearson correlation coefficient

    df.corr()['Price'] 

    sns.heatmap(df.corr())

    sns.histplot(df["Price"],kde=True)

#Applying log transformation to make Price distribution normal

    sns.histplot(np.log(df['Price']),kde=True)

#Separating input features from labels

    X = df.drop(columns=['Price'])
    y = np.log(df['Price'])

    X

    y

#Squeezing y values between 0 or 1

    y = y/10

    y
