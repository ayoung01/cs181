import util
import numpy as np
import scipy.sparse as ss
import scipy.sparse.linalg as ssl
import pickle
from matplotlib import pyplot as plt
from sklearn import linear_model

train_filename = 'ratings-train.csv'
test_filename  = 'ratings-test.csv'
user_filename  = 'users.csv'
book_filename  = 'books.csv'

train_valid  = util.load_train(train_filename)
test_queries   = util.load_test(test_filename)
user_list      = util.load_users(user_filename)
book_list      = util.load_books(book_filename)

i_id = []; id_i = {}
i = 0
for entry in user_list:
    user = entry['user']
    i_id.append(user)
    id_i[user] = i
    i += 1
for entry in book_list:
    isbn = entry['isbn']
    i_id.append(isbn)
    id_i[isbn] = i
    i += 1
    
global_mean = float(sum(map(lambda x: x['rating'], train_valid)))/len(train_valid)

def initialize(data):
    # mapping index to user name and book isbn   
    n = len(data); p = len(user_list) + len(book_list)
    x = ss.lil_matrix((n,p), dtype = np.uint16)
    y = np.zeros((n,1), dtype = np.float)
    
    row = 0
    counter = 0
    for entry in data:
        counter += 1
        if counter % 10000 == 0:
            print int(counter/n * 100)
        user = entry['user']; isbn = entry['isbn']; rating = entry['rating']
        user_i = id_i[user]; isbn_i = id_i[isbn]
    
        x[row, user_i] = np.uint16(1)
        x[row, isbn_i] = np.uint16(1)
        y[row] = np.float(rating)
        row += 1
    

    y -= global_mean    
    x = ss.csc_matrix(x)
    #xtx = x.transpose().dot(x)
    #xty = x.transpose().dot(y)
    
    return x, y

#print("Ridge(alpha=1.0)")
#clf = Ridge(alpha=1.0)
#clf.fit(xtx, xty) 

train = train_valid[40000:]
valid = train_valid[:40000]

train_x, train_y = initialize(train)
valid_x, valid_y = initialize(valid)


alpha = 1e-6
clf = linear_model.Lasso(alpha=alpha, fit_intercept=False, normalize=False)
clf.fit(train_x, train_y)
b = clf.coef_; b = b.reshape((len(b), 1))
print(np.sum(b))
#RMSE = []; TRAIN_RMSE = [];
#lamb = 4.8
#sol = ssl.lsqr(x, y, damp = np.sqrt(lamb), iter_lim = 100000)
#b_origin = np.array(sol[0]).reshape((len(sol[0]), 1))

#b = np.copy(b_origin)
#delta = 0.02
#for i, bi in enumerate(b):
#    if np.abs(bi) < delta:
#        b[i] = 0


y_predict = train_x.dot(b)
res = y_predict - train_y
train_rmse = float(np.sqrt(np.dot(res.T, res)/len(res)))

y_predict = valid_x.dot(b)
res = y_predict - valid_y
valid_rmse = float(np.sqrt(np.dot(res.T, res)/len(res)))

print (alpha, train_rmse, valid_rmse)
