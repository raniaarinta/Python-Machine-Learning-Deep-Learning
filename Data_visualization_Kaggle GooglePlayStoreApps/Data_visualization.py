import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

data = pd.read_csv("googleplaystore.csv")
data.head(10)
data.info()
data['Price'] = data['Price'].str.slice(1, 9)
data['Price'] = pd.to_numeric(data['Price'], errors='coerce')



print(data.describe())

print(data['Category'].value_counts(normalize=True))

top_10_user_rating = (
    data.groupby('App', as_index=False)
    .agg({'Rating':'sum'})
    .sort_values('Rating', ascending=False)[:10]
)
top_10_reviews=(
    data.groupby('App', as_index=False)
    .agg({'Reviews':'sum'})
    .sort_values('Reviews', ascending=False)[:10]
)
top_5_install=(
    data.groupby('App', as_index=False)
        .agg({'Installs':'max'})
        .sort_values('Installs', ascending=False)[:5]
)

print(top_10_user_rating)
print("\n----------------------------\n")
print(top_10_reviews)
print("\n----------------------------\n")
print(top_5_install)

plt.figure(figsize=(5,5))
plt.title('Rating')
sn.distplot(data['Rating'],color='red',kde=True)
plt.show()
data['Category'].value_counts().plot.pie(figsize=(30,30))
plt.show()

data.groupby(['Category'])['Rating'].sum().plot(kind='barh')
plt.show()

data.groupby(['Content Rating'])['Rating'].sum().plot(kind='barh')
plt.show()

data.groupby(['Category'])['Price'].max().plot(kind='barh')
plt.show()

data.groupby(['Content Rating'])['Price'].sum().plot(kind='barh')
plt.show()

data['Reviews'] = pd.to_numeric(data['Reviews'], errors='coerce')

data.groupby(['Category'])['Reviews'].sum().plot(kind='barh',figsize=(100,100))
plt.show()

data.groupby(['Type'])['Reviews'].sum().plot(kind='barh',figsize=(100,100))
plt.show()

data.groupby(['Type'])['Rating'].sum().plot(kind='barh',figsize=(100,100))
plt.show()
