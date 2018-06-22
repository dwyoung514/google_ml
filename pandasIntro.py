import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
print (pd.__version__)

#import csv file
california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")
print(california_housing_dataframe.describe())

print(california_housing_dataframe.head())

print(california_housing_dataframe.hist('housing_median_age'))

print(california_housing_dataframe.population)

plt.show()

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])

pd.DataFrame({ 'City name': city_names, 'Population': population })

cities = pd.DataFrame({ 'City name': city_names, 'Population': population })
print (type(cities['City name']))
print(cities['City name'])
print(population)
print(population*2)
print(population/1000)

print(population.apply(lambda val: val > 1000000))


print(cities)
cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['Population'] / cities['Area square miles']
print(cities)

cities['Boolean'] = (cities['Area square miles'] > 50) & cities['City name'].apply(lambda name: name.startswith('San'))
print(cities)
print(city_names.index)
print(cities.index)
print(cities.reindex([2,0,1]))

print(cities.reindex(np.random.permutation(cities.index)))


print(cities.reindex([4,8,10,1]))
#population = california_housing_dataframe.population

