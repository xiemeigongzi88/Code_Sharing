import numpy as np 
from matplotlib import pyplot as plt

    
########################################################
# import data 

def get_train_test_set(file):
    
    data = np.loadtxt(file, skiprows=19)
    # Delete the first column
    data = np.delete(data, 0, 1)
    
    x = data[:, :-1]
    add_list = []
    for i in range(x.shape[1]):
        col_value = x[:,i]
        
        max_col = np.max(col_value)
        min_col = np.min(col_value)
        new_value = (col_value - min_col) /(max_col - min_col)
        add_list.append(new_value)
          
    norm_x = np.array(add_list).T
    add_one_col = np.ones(norm_x.shape[0])
    data_x = np.c_[norm_x, add_one_col]  
    
    y = data[:, -1]
    max_y = np.max(y) # 1113.156 
    min_y = np.min(y) # 790.733
    norm_y = (y-min_y)/(max_y - min_y)
    
    
    x_train = data_x[:int(0.8*data_x.shape[0])]
    y_train = norm_y[:int(0.8*data_x.shape[0])]
    x_test = data_x[int(0.8*data_x.shape[0]):]
    y_test = norm_y[int(0.8*data_x.shape[0]):]
    
    return x_train, x_test, y_train, y_test
    
file = "data.txt"

x_train, x_test, y_train, y_test = get_train_test_set(file)

#######################################################################

lambda_ridge = 2
d = 15

epsilon = 0.001 
lr = 0.01

def mse (x_test, y_test , w):
    
    diff = np.sum(w * x_test, axis = 1) - y_test
    
    mse = np.sum(diff **2) / (2* y_test.shape[0])
    
    return mse 


def ridge_regression(x_train, y_train,lambda_ridge,x_test, y_test, epsilon, lr):

    num_train = x_train.shape[0]
    w = np.zeros(x_train.shape[1])
    loss_bef = 1e-6
    
    loss_ridge = []
    mse_curve = []
    while  True: 
        
        y_hat = np.dot(w, x_train.T)
        
        linear_loss = np.sum((y_hat-y_train)**2)/(2*num_train)
        ridge_loss = lambda_ridge * np.sum(w**2)/ (2*(d+1)) 
        loss = linear_loss + ridge_loss
        loss_ridge.append(loss)
        
        grad = np.dot(x_train.T, (y_hat-y_train))/num_train + lambda_ridge * w /(d)
        w -= lr * grad 
        
        # mse 

        mse_val = mse(x_test, y_test, w)
        mse_curve.append(mse_val)
        
        if np.abs(loss-loss_bef)*100/loss_bef < epsilon:
            break
        else:
            loss_bef = loss

    return w, loss_ridge, mse_curve

w_ridge, loss_ridge, mse_ridge = ridge_regression(x_train, y_train,lambda_ridge,x_test, y_test, epsilon, lr)
print(sum(np.abs(w_ridge)<0.01)) # 2

# square loss on test data 
print("Squre Error of Ridge: ", mse(x_test, y_test, w_ridge)*x_test.shape[0]) # 0.1532659286282437 


################################################################################
# lasso 

def sign(x):
    if x >= 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0
    
lambda_lasso = 0.3
d = 15

epsilon = 0.001 
lr = 0.01


def lasso_regression(x_train, y_train,lambda_ridge,x_test, y_test, epsilon, lr, sign):
    
    num_train = x_train.shape[0]
    w = np.zeros(x_train.shape[1])
    loss_bef = 1e-6
    vec_sign = np.vectorize(sign)
    loss_lasso = []
    mse_curve = []
    
    while True:
        y_hat = np.dot(w, x_train.T)
        linear_loss = np.sum((y_hat-y_train)**2)/(2*num_train)
        lasso_loss = lambda_lasso * np.sum(abs(w))/ (2*(d)) 
        
        loss = linear_loss + lasso_loss
        
        grad = np.dot(x_train.T, (y_hat-y_train))/num_train + lambda_lasso * vec_sign(w)/(2*d)
        
        w -= lr * grad 
        
        mse_val = mse(x_test, y_test, w)
        mse_curve.append(mse_val)
        
        loss_lasso.append(loss)
        
        if np.abs(loss-loss_bef)*100/loss_bef < epsilon:
            break
        else:
            loss_bef = loss
        
    return w, loss_lasso, mse_curve

w_lasso, loss_lasso, mse_lasso = lasso_regression(x_train, y_train,lambda_ridge,x_test, y_test, epsilon, lr, sign)

print(sum(np.abs(w_lasso)<0.01)) # 6
print("Squre Error of Lasso: ", mse(x_test, y_test, w_lasso)*x_test.shape[0])# 0.1553511078190968 

################################################################################
def plot_diag(loss, mse,label_loss):
    plt.plot(loss_ridge , color = 'blue', label = label_loss)
    plt.plot(mse, color = 'red', label ='MSE')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

plot_diag(loss_ridge, mse_ridge, "Ridge Regression")

plot_diag(loss_lasso, mse_lasso, "Lasso Regression")
