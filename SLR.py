import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
​
data = pd.DataFrame({
    'Jam' : [2, 4, 5, 6, 7, 8, 9, 9, 12, 10],
    'Skor' : [87, 36, 79, 90, 110, 98, 89, 82, 97, 76]
})
​
data
​
plt.scatter(data.Jam, data.Skor)
plt.title('Jam Ujian vs Hasil Ujian')
plt.xlabel('Jam')
plt.ylabel('Skor')
plt.show()
​
data.boxplot(column=['Skor'])
​
y = data['Skor']
​
x = data[['Jam']]
​
x = sm.add_constant(x)
​
model = sm.OLS(y, x).fit()
​
print(model.summary())
​
​
fig = plt.figure(figsize=(12,8))
​
fig = sm.graphics.plot_regress_exog(model, 'Jam', fig=fig)
​
res = model.resid
​
fig = sm.qqplot(res, fit=True, line="45")
plt.show() 
