# Data Preprocessing for Tortilla Prices Analysis in México

## Authors:
- **Julio Cesar De Aquino Castellanos**
- **Lorena Danae Perez Lopez**

**Username:** `juliodeaquino`  
**Key:** Private

---

## Installing Dependencies not included in Colab:
```bash
! pip install opendatasets
```

---

## Libraries Importation:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os  # For Kernel manipulation in Colab
import opendatasets as od  # For Kaggle's API importation
from scipy.stats import linregress  # For linear correlation analysis
```

---

## Kaggle API Importation:
```python
link = 'https://www.kaggle.com/datasets/richave/tortilla-prices-in-mexico'
od.download(link)
```

---

## Data Import:
```python
data_dir = '/content/tortilla-prices-in-mexico'
file_name = 'tortilla-prices.csv'
file_path = os.path.join(data_dir, file_name)
df = pd.read_csv(file_path)
```

---

## Data Overview:
```python
df.head()
df.tail()
print(f"There are {len(df)} rows!")
df.info()
df.describe()
```

---

## Handling Missing Data:
```python
df.isnull().sum()
mean_price = df['Price per kilogram'].mean()
df['Price per kilogram'].fillna(mean_price, inplace=True)
```

---

## Data Type Conversion:
```python
df['Price per kilogram'] = df['Price per kilogram'].astype(np.float32)
df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
```

---

## Outlier Detection and Filtration:
```python
Q1 = df['Price per kilogram'].quantile(0.25)
Q3 = df['Price per kilogram'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_without_outliers = df[(df['Price per kilogram'] >= lower_bound) & 
                         (df['Price per kilogram'] <= upper_bound)]
```

---

## Exploratory Data Analysis (EDA):
```python
sns.histplot(df_without_outliers['Price per kilogram'])
sns.boxplot(df_without_outliers['Price per kilogram'])
sns.countplot(df_without_outliers['State'])
```

---

## Bivariate Analysis:
Transforming `Date` into a continuous variable:
```python
df['Date-Continuos'] = df['Date'].astype('int64') // 10**9
sample = df_without_outliers.iloc[::1330]
```

Scatter plot with regression line:
```python
sns.scatterplot(x='Date', y='Price per kilogram', data=sample)
sns.regplot(data=sample, x='Date-Continuos', y='Price per kilogram', marker='o')
```

Regression metrics:
```python
slope, intercept, r_value, p_value, std_err = linregress(sample['Year'], sample['Price per kilogram'])
print(f"Slope: {slope}")
print(f"Intercept: {intercept}")
print(f"R-squared: {r_value**2}")
print(f"P-value: {p_value}")
print(f"Standard Error: {std_err}")
```

Bar plot:
```python
sns.barplot(data=sample, x='Year', y='Price per kilogram')
plt.xticks(rotation=45)
plt.show()
```

Heatmap of correlations:
```python
sns.heatmap(df[['Year', 'Month', 'Price per kilogram']].corr())
```

---

## Bibliography:
- [Tortilla Prices in Mexico Dataset](https://www.kaggle.com/datasets/richave/tortilla-prices-in-mexico)
