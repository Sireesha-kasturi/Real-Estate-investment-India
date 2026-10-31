# %% [markdown]
# ## Real Estate Investment Advisor
# ## Predicting Property Profitability & Future Value
# ## Intern name - Sireesha

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import streamlit as st
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# %%
data=pd.read_csv(r"C:\Users\DELL\Desktop\india_housing_prices (1).csv")
data


# %%
data.shape

# %%
data.info()

# %%
data.head()



# %%
data.tail()

# %%
data.describe()


# %%
data.columns

# %%
data.isnull().sum()

# %%
data.duplicated().sum()

# %%
data.duplicated()

# %%
cat_col=data.select_dtypes(include="str")
cat_col

# %%
num_data=data.select_dtypes(include="number")
num_data

# %%
#Drop the ID column because it has all unique values we do not need it for further analysis
num_data=num_data.drop("ID",axis=1)
num_data

# %% [markdown]
# ## Feature scaling

# %%
#Apply standard scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# %%

from sklearn.preprocessing import StandardScaler

Scalar_Standard = StandardScaler()

Scaled_standard = Scalar_Standard.fit_transform(num_data)

Scaled_standard

# %%
df_statdard_scaled=pd.DataFrame(Scaled_standard,columns=num_data.columns)
df_statdard_scaled

# %%

Scalar_MinMax = MinMaxScaler()

Scaled_minmax = Scalar_MinMax.fit_transform(num_data)

Scaled_minmax = pd.DataFrame(
    Scaled_minmax,
    columns=num_data.columns,
    index=num_data.index
)

Scaled_minmax.head()

# %%
#checki for correlation between variables
num_data.corr()

# %%

sn.heatmap(num_data.corr(),cmap='YlOrBr')
plt.title('Correlation Heatmap')
plt.show()

# %% [markdown]
# ## Encoding Categorical variable

# %% [markdown]
# ### OHE-one hot encoding for categorical columns

# %%
#fetching for categorical columns
x=cat_col
x

# %%
cat_col.isnull().sum()

# %%
#apply ohe using pandas
x_encoded=pd.get_dummies(x)
x_encoded


# %%
#apply OHE usinh sklearn
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(
    handle_unknown='ignore',
    sparse_output=False
)

encoded_data = encoder.fit_transform(data[['City']])

encoded_df = pd.DataFrame(
    encoded_data,
    columns=encoder.get_feature_names_out(['City']),
    index=data.index
)

encoded_df.head()


# %%
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

data['City_Encoded'] = encoder.fit_transform(data['City'])

data[['City', 'City_Encoded']].head()


# %%
cat_col

# %%
from sklearn.preprocessing import LabelEncoder

data2 = data.copy()

categorical_cols = [
    'Property_Type',
    'Furnished_Status',
    'Facing',
    'Owner_Type',
    'State',
    'Locality',
    'Amenities',
    'Parking_Space'
]

for col in cat_col:
    encoder = LabelEncoder()
    data2[col] = encoder.fit_transform(data2[col].astype(str))

data2.head()

# %% [markdown]
# ## Exploratory Data Analysis (EDA)

# %%
#propertys in states and cities
data[['State', 'City']].value_counts()

# %%
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))

sns.countplot(
    data=data,
    y='State',
    order=data['State'].value_counts().index
)

plt.title('Number of Properties by State')
plt.xlabel('Number of Properties')
plt.ylabel('State')
plt.tight_layout()
plt.show()

# %%
#1.	What is the distribution of property type?
property_counts = data['Property_Type'].value_counts()

plt.figure(figsize=(8, 8))

plt.pie(
    property_counts,
    labels=property_counts.index,
    autopct='%1.1f%%',
    startangle=90,
    counterclock=False
)

plt.title('Distribution of Properties by Property Type')
plt.axis('equal')
plt.show()

# %%
#1.	What is the distribution of property prices?
sum_price = data.groupby('Property_Type')['Price_in_Lakhs'].sum()
print(sum_price)

# %%
avg_price = data.groupby('Property_Type')['Price_in_Lakhs'].mean()
print(avg_price)

# %%
#2.	What is the distribution of property sizes?
sns.histplot(data['Size_in_SqFt'],
             bins=30,
             kde=True,
             color='skyblue')

plt.title('Distribution of Property Sizes')
plt.xlabel('Property Size (SqFt)')
plt.ylabel('Number of Properties')

plt.show()

# %%
#3.	How does price per sq ft vary by property type?
avg_price_sft = data.groupby('Property_Type')['Price_per_SqFt'].mean()
plt.figure(figsize=(10,4))

sns.boxplot(x=data['Size_in_SqFt'])

plt.title('Outliers in Property Size')
plt.xlabel('Size in SqFt')

plt.show()
print(avg_price_sft)

# %%
plt.figure(figsize=(10,5))

sns.barplot(
    data=data,
    x='Property_Type',
    y='Price_per_SqFt',
    estimator='mean'
)

plt.title('Average Price per SqFt by Property Type')
plt.xlabel('Property Type')
plt.ylabel('Average Price per SqFt')

plt.show()

# %%
#5.	Are there any outliers in price per sq ft or property size?
plt.figure(figsize=(10,4))

sns.boxplot(x=data['Size_in_SqFt'])

plt.title('Outliers in Property Size')
plt.xlabel('Size in SqFt')

plt.show()

# %%
plt.figure(figsize=(10,4))

sns.boxplot(x=data['Price_per_SqFt'])

plt.title('Outliers in Price per SqFt')
plt.xlabel('Price per SqFt')

plt.show()

# %%
sns.catplot(
    data=data,
    x='Property_Type',
    y='Price_in_Lakhs',
    hue='Parking_Space',
    col='Security',
    kind='bar',
    errorbar=None,
    height=5,
    aspect=1.2
)

plt.show()

# %%
property_size = data.groupby('Property_Type')['Size_in_SqFt'].agg(
    Maximum_Size='max',
    Minimum_Size='min',
    Avg_Size='mean'
)

print(property_size)

# %%
property_size.plot(kind='bar', figsize=(10,5))

plt.title('Maximum and Minimum Property Size by Property Type')
plt.xlabel('Property Type')
plt.ylabel('Size (Sq. Ft.)')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# %%
df=data
df

# %%
def build_good_investment_label(df):
    df = df.copy()

    city_median_price = df.groupby('City')['Price_in_Lakhs'].transform('median')
    dataset_median_ppsf = df['Price_per_SqFt'].median()

    # Explicitly ensure these columns are numeric before comparison
    # Coerce errors to NaN, then fill with a sensible value like 0 or median
    df['Nearby_Schools'] = pd.to_numeric(df['Nearby_Schools'], errors='coerce').fillna(0).astype(int)
    df['Public_Transport_Accessibility'] = pd.to_numeric(df['Public_Transport_Accessibility'], errors='coerce').fillna(0).astype(int)


    c1 = df['Price_in_Lakhs'] <= city_median_price
    c2 = df['Price_per_SqFt'] <= dataset_median_ppsf
    c3 = df['Nearby_Schools'] >= 2
    c4 = df['Public_Transport_Accessibility'] >= 2

    conditions = c1.astype(int) + c2.astype(int) + c3.astype(int) + c4.astype(int)
    df['Good_Investment'] = (conditions >= 2).astype(int)

    print("Good_Investment Value Counts:")
    print(df['Good_Investment'].value_counts())
    print(f"\nClass Balance: {df['Good_Investment'].mean():.2%} positive class")

    return df

df = build_good_investment_label(df)
     
     

# %%
#6.	What is the average price per sq ft by state?
plt.figure(figsize=(12,6))

sns.barplot(
    data=data,
    x='State',
    y='Price_per_SqFt',
    estimator='mean',
    errorbar=None
)

plt.title('Average Price per SqFt by State')
plt.xlabel('State')
plt.ylabel('Average Price per SqFt')
plt.xticks(rotation=45)

plt.show()

# %%
data.columns

# %%
#7.	What is the average property price by city?
plt.figure(figsize=(12,6))

sns.barplot(
    data=data,
    x='City',
    y='Price_in_Lakhs',
    estimator='mean',
    errorbar=None
)

plt.title('Average Price by city')
plt.xlabel('City')
plt.ylabel('Average Price')
plt.xticks(rotation=45)

plt.show()

# %%
#8. What is the median age of properties by locality?
median_age = data.groupby('Locality')['Age_of_Property'].median()
print(median_age)


# %%
#9.	How is BHK distributed across cities?
bhk_distribution = data.groupby(
    ['City', 'BHK']
).size().unstack(fill_value=0)

print(bhk_distribution)

# %%
#10.	What are the price trends for the top 5 most expensive localities?
top_localities = (
    data.groupby('Locality')['Price_in_Lakhs']
    .mean()
    .sort_values(ascending=False)
    .head(5)
)

print(top_localities)

# %%
top5_data = data[data['Locality'].isin(top_localities.index)]

plt.figure(figsize=(12,6))

sns.barplot(
    data=top5_data,
    x='Locality',
    y='Price_in_Lakhs',
    
    errorbar=None,
    order=top_localities.index
)

plt.title('Top 5 Most Expensive Localities')
plt.xlabel('Locality')
plt.ylabel('Property Price')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# %%
#11. How are numeric features correlated with each other?
sn.heatmap(num_data.corr(),cmap='YlOrBr')
plt.title('Correlation Heatmap')
plt.show()


# %%
#14.How does price vary by furnished status?
plt.figure(figsize=(9,5))

sns.barplot(
    data=data,
    x='Furnished_Status',
    y='Price_in_Lakhs',
    estimator='mean',
    errorbar=None
)

plt.title('Average Property Price by Furnished Status')
plt.xlabel('Furnished Status')
plt.ylabel('Average Property Price')

plt.show()


# %%
#15.How does price per sq ft vary by property facing direction?
plt.figure(figsize=(9,5))

sns.barplot(
    data=data,
    x='Facing',
    y='Price_per_SqFt',
    estimator='mean',
    errorbar=None
)

plt.title('Average Property Price_sqft by Facing')
plt.xlabel('Facing')
plt.ylabel('Average price_sqft')

plt.show()

# %%
plt.figure(figsize=(9,5))

sns.countplot(
    data=data,
    x='Owner_Type',
    order=data['Owner_Type'].value_counts().index
)

plt.title('Number of Properties by Owner Type')
plt.xlabel('Owner Type')
plt.ylabel('Number of Properties')

plt.show()

# %%
#18.	Does parking space affect property price?

plt.figure(figsize=(9,5))

sns.boxplot(
    data=data,
    x='Parking_Space',
    y='Price_in_Lakhs'
)

plt.title('Property Price Distribution by Parking Space')
plt.show()




# %%



