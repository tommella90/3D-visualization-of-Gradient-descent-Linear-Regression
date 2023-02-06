# CREATE RANDOM DATA

import numpy as np
from sklearn.linear_model import LinearRegression
import random 

# choose parameters
n_sample = op('parameters').par.value0
noise = op('parameters').par.value1
slope_interval = op('parameters').par.value2

# create x, y, intercept and slope
x = np.arange(n_sample)/n_sample
slope = random.randint(10, slope_interval)
bias = np.random.uniform(-10, noise, size=(n_sample,))
y = (slope * x + bias)/100

# random initial parameters
b, m = random.randint(-150,150)/100, random.randint(-100,100)/100

# errors
y_pred = b + m*x
errors = (y - y_pred)**2


# transfer data to touchdesigner
op('data/data_storage_x').unstore('*')
op('data/data_storage_x').store('num_vars', x)

op('data/data_storage_y').unstore('*')
op('data/data_storage_y').store('num_vars', y)

op('intercept').unstore('*')
op('intercept').store('intercept', b)

op('coefficient').unstore('*')
op('coefficient').store('slope', m)

op('y_pred').unstore('*')
op('y_pred').store('y_pred', y_pred)

op('ssr_').unstore('*')
op('ssr_').store('errors', errors)


# crate error 3d space to visualize local minima 
intercept, coefficient, errors = [], [], []
my_range = 100
steps = 1
for intercept_i in range(-my_range+steps, my_range, steps+steps): 
	intercept_i = float(intercept_i)/100

	for coefficient_i in range(-my_range+steps, my_range, steps+steps): 
		coefficient_i = float(coefficient_i)/100

		intercept.append(intercept_i)
		coefficient.append(coefficient_i)

		y_pred = []
		for instance_x in x:
			y_pred.append(intercept_i + coefficient_i*instance_x)
		errors.append(((y_pred - y)**2).sum())


errors = np.array(errors)
ssr = errors / np.linalg.norm(errors)
ssr = errors.sum() / x.shape[0] 


# transfer 3d data to touchdesigner
op('xy_ssr').clear()

op('coeff_3d').unstore('*')
op('coeff_3d').store('coefficient', coefficient)

op('interc_3d').unstore('*')
op('interc_3d').store('interc_3d', intercept)

op('ssr_3d').unstore('*')
op('ssr_3d').store('ssr', errors)

epoch=0
op('epoch').clear()
op('epoch').appendRow(epoch)

op('total_errors').clear()
op('total_errors').appendRow(ssr)


# plot parameters from sklearn to compare 
x = x.reshape(-1,1)
reg = LinearRegression().fit(x, y)

m_sklearn = reg.coef_[0]
b_sklearn =  reg.intercept_
print(m_sklearn, b_sklearn)

op('row_global_minima').par.rowindexstart = m_sklearn
op('row_global_minima').par.rowindexend = b_sklearn

op('ols_sklearn').clear()
op('ols_sklearn').appendRow([m_sklearn, b_sklearn])


op('info').par.value0 = m
op('info').par.value1 = m_sklearn
op('info').par.value2 = b
op('info').par.value3 = b_sklearn
op('info').par.value4 = ssr
op('info').par.value5 = 0
