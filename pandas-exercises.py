import pandas as pd

insurance = pd.read_csv('data/insurance.csv')

print(insurance)

print(insurance.columns)
print('---this is the output from to_string - print the whole data set----')
print(insurance.to_string())

print('---this is the output from dtype----')
print(insurance.dtypes)

print('---this is the output from shape----')
print(insurance.shape)

print('---this is the output from info()----')
print(insurance.info())

print('---this is the output from describe()----')
print(insurance.describe())

print('---this is the output by selecting Age----')
print(insurance['age'])

print('---this is the output by selecting Age, Children, Charges - used [[ ]] because passing a list ----')
print(insurance[['age', 'children', 'charges']])

print('---this is the output by selecting first 5 rows of Age, Children, Charges----')
print(insurance.loc[[0,1,2,3,4], ['age', 'children', 'charges']])
print('-------- another solution -----------')
print(insurance[['age', 'children', 'charges']].head(5))

print('---this is the output by selecting min of Charges----')
print(insurance['charges'].min())

print('---this is the output by selecting max of Charges----')
print(insurance['charges'].max())

print('---this is the output by selecting Average of Charges----')
print(insurance['charges'].mean())

print('---this is the output by selecting Age, Sex, Charges, where paid: 10797.3362----')
print(insurance.loc[insurance['charges'] == 10797.3362, ['age', 'sex', 'charges']])
print(insurance.loc[insurance['charges'] == 10797.3362, ['age', 'sex', 'charges', 'smoker']])


print('---this is the output by selecting the Age of person who paid Max----')
max_val = insurance['charges'].max()
print(insurance.loc[insurance['charges'] == max_val, ['age']])

print('---this is the output of how many insured by region ----')
print(insurance['region'].value_counts())

print('---this is the output of how many insured are Children----')
print(insurance['children'].sum())
print('---real solution is - this is the output of how many insured are Children----')
print(insurance[insurance['age'] < 18])

print('---this is the output of how many insured by region - to_string helps to show all the data not ... ----')
# I though younger paid more, Family with kids paid less, but looks like Im wrong.
# age  and charges are correlated
print(insurance.corr().to_string())