# UPDATE INFO THROUGH THE EPOCHS

import numpy as np

# Retreive the data from touchdesigner
x, y = [], []
for i in range(1, op('xy').numRows):
	inst_x = float(op('xy')[i, 'x'])
	x.append(inst_x)
	inst_y = float(op('xy')[i, 'y'])
	y.append(inst_y)

x, y = np.array(x), np.array(y)
b, m = float(op('bias_')[0,0]), float(op('coeff_')[0,0])
learning_rate = .1
N = x.shape[0]

# calculate the gradient for b and m
# y = mx + b
der_m, der_b = [], []
for x_i, y_i in zip(x, y):
	y_pred = x_i*m + b
	der_m.append((y_i - y_pred) * x_i)
	der_b.append(y_i - y_pred)

der_m = (-2/N)*(np.array(der_m).sum())
der_b = (-2/N)*(np.array(der_b).sum())

new_m = m - (learning_rate*der_m)
new_b = b - (learning_rate*der_b)

y_pred = new_b + new_m*x
errors = (y_pred - y)**2
errors = np.array(errors)
ssr = errors.sum() / x.shape[0]

print("true coeff=", op('ols_')[1,0], 
	"\nactual coeff=", new_m,
	"\ntrue intercept=:", op('ols_')[1,1],
	"\nactual intercept=", new_b,
	"\nssr=", ssr, 
	"\nepoch=", op('epoch')[op('epoch').numRows-1, 0])


# transfer data to touchdesigner
op('info').par.value0 = op('ols_')[1,0]
op('info').par.value1 = new_m
op('info').par.value2 = op('ols_')[1,1]
op('info').par.value3 = new_b
op('info').par.value4 = ssr
op('info').par.value5 = op('epoch')[op('epoch').numRows-1, 0]

# weights and biases
op('intercept').unstore('*')
op('intercept').store('intercept', new_b)

op('coefficient').unstore('*')
op('coefficient').store('coefficient', new_m)

op('y_pred').unstore('*')
op('y_pred').store('y_pred', y_pred)

op('ssr_').unstore('*')
op('ssr_').store('ssr', errors)

# update epoche and errors
n_rows = op('epoch').numRows
epoch = op('epoch')[n_rows-1, 0]
epoch+=1
op('epoch').appendRow(epoch)
op('total_errors').appendRow(ssr)

