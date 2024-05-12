import pandas as pd
import numpy as np
import re
import statsmodels.api as sm

# Sample data based on the provided table structure
# data = {
#     "countries": ["Afghanistan", "Albania", "Angola", "Antigua and Barbuda", "Argentina"],
#     "HIV prevalence": ["<0.1", "<0.1", "0.8", "...", "<0.1"],
#     "Knowledge on HIV": ["1,75", "31,72", "32,25", "85,67", "..."],
#     "laws and policies": ["No", "PB", "PB", "PB", "PB"],
#     "Curriculum": ["CU", "CU", "CB", "...", "CB"],
#     "Curriculum use": ["MB", "MU", "MB", "...", "..."],
#     "Coverage of primary schools8": ["0-25%", "76-100%", "76-100%", "76-100%", "..."],
#     "Coverage of secondary schools8": ["0-25%", "76-100%", "76-100%", "76-100%", "..."],
#     "Teacher Training Policy/ Programme/ Curriculum": ["No", "Yes", "Yes", "Yes", "Yes"],
#     "Monitoring and Evaluation": ["...", "M&E in place", "Included in EMIS", "...", "..."],
#     "Resistence in society/CSE opposition1": ["No", "Yes", "No", "...", "..."]
# }

# Convert dictionary to DataFrame
df = pd.read_csv('data analysis.csv', dtype=str)
columns_to_drop = [column for column in df.columns if 'unnamed' in column.lower() or 'monitoring and evaluation' in column.lower()]
# print("Columns to drop:", columns_to_drop)
df.drop(columns=columns_to_drop, inplace=True)

def get_nan_values(val):
    if type(val) is not str and np.isnan(val):
        return np.nan
    val = val.strip()
    if '...' in val or '…' in val or val == '':
        return np.nan
    return val
    
def clean_small_values(val):
    if type(val) is not str and np.isnan(val):
        return np.nan
    if '<0.1' in val.replace(' ', ''):
        return 0.1
    return val

def remove_range(val):
    if type(val) is not str:
        if np.isnan(val):
            return np.nan
        else:
            return val
    return re.sub(r'\s*\[.*?\]', '', val).strip()

def replace(val):
    if type(val) is str:
        return val.replace(',', '.')
    return val

# Data cleaning
df = df.map(get_nan_values)
# Replace "<0.1" with 0.1 and ranges with their averages
df['HIV prevalence'] = df['HIV prevalence'].apply(clean_small_values).apply(remove_range).astype(float)
df['Knowledge on HIV'] = df['Knowledge on HIV'].apply(replace).astype(float)

# Convert percentage ranges to their averages
def average_percentage(range_str):
    if '-' in range_str:
        low, high = map(float, range_str[:-1].split('-'))
        return (low + high) / 2
    return float(range_str[:-1])

def unify_values(val):
    if type(val) is not str:
        if np.isnan(val):
            return np.nan
    val = val.strip()
    if 'M&E' in val:
        return 'M&E'
    if 'EMIS' in val:
        return 'EMIS'
    return clean_small_values

df['Coverage of primary schools8'] = df['Coverage of primary schools8'].apply(lambda x: average_percentage(x) if pd.notna(x) else x).astype(float)
df['Coverage of secondary schools8'] = df['Coverage of secondary schools8'].apply(lambda x: average_percentage(x) if pd.notna(x) else x).astype(float)
# df['Monitoring and Evaluation'] = df['Monitoring and Evaluation'].replace(unify_values)

categorical_columns = ['laws and policies', 'Curriculum', 'Curriculum use', 'Teacher Training Policy/ Programme/ Curriculum', 'Resistence in society/CSE opposition1']
# Convert categorical to numerical (simple binary encoding for demonstration)
# drop={'laws and policies': 'No', 'Curriculum': 'No',  'Curriculum use': 'OU', 'Teacher Training Policy/ Programme/ Curriculum': 'No', 'Resistence in society/CSE opposition1': 'No'}
dummies = pd.get_dummies(df[categorical_columns])

# Manually drop the 'no' columns
columns_to_drop = [col for col in dummies.columns if col.lower().endswith('_no')]
dummies = dummies.drop(columns=columns_to_drop)

# Convert dummies to float dtype
dummies = dummies.astype(float)
df = df.drop(categorical_columns, axis=1)
df = pd.concat([df, dummies], axis=1)

# df.dropna(inplace=True)
critical_columns = ['HIV prevalence', 'Knowledge on HIV']
df = df.dropna(subset=critical_columns)

# Splitting the data into features and target for the first model (HIV prevalence)
X = df.drop(columns=['countries', 'HIV prevalence', 'Knowledge on HIV'])
X = X.apply(lambda x: x.fillna(x.mean()), axis=0)
y_hiv_prevalence = df['HIV prevalence'].astype(float)
y_knowledge_on_hiv = df['Knowledge on HIV'].astype(float)

# # Creating and training the linear regression model
# model_hiv_prevalence = LinearRegression()
# model_hiv_prevalence.fit(X, y_hiv_prevalence)

# r_squared = model_hiv_prevalence.score(X, y_hiv_prevalence)
# print(f"R-squared: {r_squared}")

# # Get the coefficients and the intercept
# coefficients = model_hiv_prevalence.coef_
# intercept = model_hiv_prevalence.intercept_

# # Display coefficients
# print("Coefficients:", coefficients)
# print("Intercept:", intercept)

# Add constant to feature matrix X
X_with_const = sm.add_constant(X)

model_hiv_prevalence = sm.OLS(y_hiv_prevalence, X_with_const).fit()
print(model_hiv_prevalence.summary())

model_knowledge_on_hiv = sm.OLS(y_knowledge_on_hiv, X_with_const).fit()
print(model_knowledge_on_hiv.summary())

# # R-squared
# print(f"R-squared: {model_hiv_prevalence.rsquared}")

# # Coefficients
# coefficients = model_hiv_prevalence.params
# print("Coefficients:")
# print(coefficients)

# # You can access the intercept with `model_hiv_prevalence.params[0]`
# # if the constant was added first in the X matrix.
# print("Intercept:", coefficients[0])


