import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
# matplotlib.rcParams["figur.figsize"] = (20,10)

df1 = pd.read_csv("./bengaluru_house_prices.csv")
print(df1.head())

print("The initial shape of data : ",df1.shape)


print(df1['area_type'].unique())

#COUNT OF COLUMNS
# print(df1['area_type'].value_counts)
#COUNT OF COLUMNS BY GROUPING
print(df1.groupby('area_type')['area_type'].agg('count'))

#DROPING UNNESECCARY COLUMNS
print("Columns : ",df1.columns)

df2 = df1.drop(['area_type','society','balcony','availability'],axis='columns')
print(df2.head())
print("The shape of df2 : ",df2.shape)
print('\n')
#DATA CLEANING
print("DATA CLEANING")
#CHECKING IF THERE IS ANY NULL VALUE
print(df2.isnull().sum())

print('\n')

# DROPPING NA VALUES BECAUSE THEY ARE SMALL IN NUMBER AS DATA HAVE MUCH MORE ROWS
df3 = df2.dropna()
print(df3.isnull().sum())
print("The shape of df3 : ",df3.shape)

# ADDING NEW FEATURES (INTEGER) FOR BHK
print(df3['size'].unique())

df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
print(df3.head())
print(df3['bhk'].unique())


# EXPLORING TOTAL_SQFT FEATURE

print("\n")
print("Total square feet")
print(df3.total_sqft.unique())

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

sqft = df3[~df3['total_sqft'].apply(is_float)].head()
print(sqft)

#taking average of total_sqft
def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens)==2:
        return(float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None

print("\n")
df4 = df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)
print(df4.head())

# ADDING NEW FEATURE CALLED PRICE PER SQUARE FEET

df5 = df4.copy()
df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft']
print(df5.head())

df5_stats = df5['price_per_sqft'].describe()
print("STATS OF DF5 : \n",df5_stats)

#EXPLORING LOCATION
print(len(df5.location.unique()))

df5.location = df5.location.apply(lambda x:x.strip())
location_stats = df5['location'].value_counts(ascending=False)
print(location_stats)

print("LOCATION : ",location_stats.values.sum())
print("TOTAL LENGTH OF LOCATION STATS : ",len(location_stats))
print("LENGHT OF LOCATION LESS THAN 10 DATAPOINT : ",len(location_stats[location_stats<=10]))

#DIMENSIONALITY REDUCTION (any location having less than 10 datapoints should be tagged as other location)
location_stats_less_than_10 = location_stats[location_stats<=10]
print("LOCATION HAVING LESS THAN 10 DATA POINT :\n",location_stats_less_than_10)

print("TOTAL NUMBER OF LOCATION IN DATA : ",len(df5.location.unique()))

df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
print("OTHER LOCATION : ",len(df5.location.unique()))

# OUTLIER DETECTION
#ONE WAY OF REMOVING OULIER
print("BEFORE OUTLIER",df5.shape)
df6 = df5[~(df5.total_sqft/df5.bhk<300)]
print("AFTER REMOVING OUTLIER",df6.shape)

#Outlier Removal Using Standard Deviation and Mean
print("\n")
print(df6.price_per_sqft.describe())

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key , subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out

df7 = remove_pps_outliers(df6)
print("THE SHAPE OF DF7 : ",df7.shape)

# Let's check if for a given location how does the 2 BHK and 3 BHK property prices look like
def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] = (5,5)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df7,"Rajaji Nagar")
plt.show()

# Now we can remove those 2 BHK apartments whose price_per_sqft is less than mean price_per_sqft of 1 BHK apartment
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8 = remove_bhk_outliers(df7)
# df8 = df7.copy()
print(df8.shape)
plot_scatter_chart(df8,"Rajaji Nagar")
plt.show()
matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df8.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")
plt.show()

# Outlier Removal Using Bathrooms Feature
print(df8.bath.unique())
plt.hist(df8.bath,rwidth=0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")
plt.show()
print(df8[df8.bath>10])

print(df8[df8.bath>df8.bhk+2])

df9 = df8[df8.bath<df8.bhk+2]
print(df9.shape)
print(df9.head())

#DROPPING SIZE AND PRICE_PER_SQFT
df10 = df9.drop(['size','price_per_sqft'],axis='columns')
df10.head(3)
print("\n")

#Use One Hot Encoding For Location
# CATEGORICAL VALUES IN NUMERIC VALUES CALLED AS ONE HOT ENCODING
dummies = pd.get_dummies(df10.location)
print("DUMMIES :\n",dummies.head())

#concatinate df10 with dummies
df11= pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
print("DF11 : ",df11.head())

#DROP LOCATION
df12 = df11.drop('location',axis='columns')
print(df12.head())

# Build a Model
print("\n")
print("BUILDING MODEL")
# X is independent variable
X = df12.drop(['price'],axis='columns')
print(X.head())
print("shape of x : ",X.shape)

y = df12.price
print("Y :\n ",y.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
score = model.score(X_test,y_test)
print("Accuracy : ",score*100)

# Use K Fold cross validation to measure accuracy of our LinearRegression model\
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val = cross_val_score(LinearRegression(), X, y, cv=cv)
print("K-cross validation : ",cross_val)

#Test the model for few properties
def predict_price(location,sqft,bath,bhk):
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index>=0:
        x[loc_index] = 1
    return model.predict([x])[0]

print("Testing model for 1st Phase JP Nagar ",predict_price('1st Phase JP Nagar',1000, 2, 2))
print("Testing model for 1st Phase JP Nagar with 3bhk ",predict_price('1st Phase JP Nagar',1000, 3, 3))
print("Testing model for Indira nagar ",predict_price('Indira Nagar',1000, 3, 3))

#Export the tested model to a pickle file
import pickle
with open('banglore_home_prices_model.pickle','wb') as f:
    pickle.dump(model,f)

#Export location and column information to a file that will be useful later on in our prediction application
import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))