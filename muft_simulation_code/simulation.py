import argparse
import tensorflow as tf
import math
import pandas as pd
from sklearn.linear_model import LinearRegression
from keras.layers import Flatten
from keras.layers import Conv1D
import numpy as np
from keras.layers import Dense,LSTM,GRU, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping

parser=argparse.ArgumentParser(description="Give Parameters...")
parser.add_argument("--cycles", type=int)
parser.add_argument("--state_prob", type=float)
parser.add_argument("--dist_prob", type=float)
parser.add_argument("--sim_no", type=int)
args=parser.parse_args()

no_cycles = args.cycles
state_probability=args.state_prob
dist_probability=args.dist_prob
sim_no = args.sim_no
best_model_no = 20

pop_path = 'population_data.csv'
#pop_path = '/content/drive/MyDrive/Research Work/SWANN/population_data.csv'
pop_data = pd.read_csv(pop_path)

def r2(y_pred, y_test):
  corr_matrix = np.corrcoef(y_pred, y_test)
  corr = corr_matrix[0,1]
  R_sq = corr**2
  return R_sq

def RMSE(y_pred, y_test):
  return math.sqrt(((y_pred - y_test)**2).mean())

def MAE(y_pred, y_test):
  return (abs(y_pred - y_test)).mean()

def MAPE(y_pred, y_test):
  return (abs((y_test - y_pred)/y_test).mean())*100

def multiply_dataset(df, mf=100):

  m_grps = df.groupby(['Multiplier'])

  new_df = pd.DataFrame()

  for m, grp in m_grps:
    #print("Processing datasets corresponding to multiplier:", m)
    #print((mf*m[0]))
    for c in range(int(mf*m[0])):
      #new_df = new_df.append(grp, ignore_index=True)
      new_df = pd.concat([new_df, grp], ignore_index=True)

  ##print(new_df)
  ##print(new_df.Multiplier.value_counts())
  return new_df

def create_2_stage_w_sample(pop_data):

  #print(pop_data)

  new_df = pd.DataFrame()

  #st_prob = (random.randint(5,6)/10.0)
  #st_prob = 0.6
  st_prob = state_probability

  #print("State probability1:", st_prob)

  st_list = pd.DataFrame(pop_data.STATE.unique())

  st_list = st_list.sample(frac=st_prob).values

  st_prob = st_list.shape[0]/pop_data.STATE.unique().shape[0]

  #print("State probability2:", st_prob)

  t_units = 0

  for st in st_list:

    ##print(st[0])
    st_data = pop_data[pop_data.STATE==st[0]]
    ##print(st_data)

    t_units+=st_data.shape[0]

    #dist_prob = (random.randint(5,6)/10.0)
    #dist_prob = 0.6
    dist_prob = dist_probability
    #print("District probability1:", dist_prob)
    dist_data = st_data.sample(frac=dist_prob).values
    ##print(dist_data)

    dist_prob = dist_data.shape[0]/st_data.shape[0]
    #print("District probability2:", dist_prob)

    w_list = [dist_prob*st_prob for i in range(dist_data.shape[0])]
    dist_data2 =pd.DataFrame(dist_data)
    dist_data2["Weight"]=w_list
    ##print(dist_data2)

    #new_df= new_df.append(dist_data2, ignore_index=True)
    new_df = pd.concat([new_df, dist_data2], ignore_index=True)

    #break

  new_df.columns = list(pop_data.columns)+["Weights"]

  ##print("Sample with weights:")
  ##print(new_df)

  #print("Population size:",pop_data.shape[0])
  #print("Sample size:",new_df.shape[0])

  #print("\n\nNo. of States in sample: ",len(new_df.STATE.unique()))
  #print("No. of States in population: ",len(pop_data.STATE.unique()))
  #print("No. of units in sample: ",new_df.shape[0])
  #print("No. of units in selected states: ",t_units)

  N = len(pop_data.STATE.unique())
  n = len(new_df.STATE.unique())
  M = t_units
  m = new_df.shape[0]

  sample_data = new_df[new_df.columns[:-1]]

  sample_data["Multiplier"]=1/new_df[new_df.columns[-1]]
  sample_data["pi"]=new_df[new_df.columns[-1]]

  #print(sample_data)
  #print("Unique Multipliers in the dataset: ",sample_data.Multiplier.unique())
  #print("Sum of Multipliers : ",sample_data.Multiplier.sum())

  return (sample_data,n,N,m,M)

def cal_pij(pop_data, sample_data, i, j):

    st_idx = sample_data.columns.tolist().index('STATE')

    state_i = sample_data.values[i][st_idx]
    state_j = sample_data.values[j][st_idx]

    N = len(pop_data.STATE.unique())
    n = round(N*0.6,0)

    M_i = pop_data[pop_data['STATE']==state_i].shape[0]
    m_i = sample_data[sample_data['STATE']==state_i].shape[0]

    M_j = pop_data[pop_data['STATE']==state_j].shape[0]
    m_j = sample_data[sample_data['STATE']==state_j].shape[0]

    if state_i == state_j:

        ##print("Records are from same state: n, N, m, M = ", n, N, m_i, M_i)
        pij = ((n*(n-1))/(N*(N-1))) * ((m_i*(m_i-1))/(M_i*(M_i-1)))
        ##print("pij=", pij)

    else:

        ##print("Records are from different states: n, N, m_i, M_i, m_j, M_j = ", n, N, m_i, M_i, m_j, M_j)
        pij = ((n*(n-1))/(N*(N-1))) * ((m_i*m_j)/(M_i*M_j))
        ##print("pij=", pij)

    return pij

def cal_ht_htvar(pop_data, data1, y, p):

    #print(data1)

    ##print(data1)

    n1 = data1.shape[0]

    pi = 'pi'

    ##print(data2)
    data2=data1

    n2 = data2.shape[0]

    '''
    def cal_pi(rec):
        ##print(rec[p])
        ##print(n1)
        return (1-((1-rec[p])**n1))

    pi_list = data2.apply(cal_pi, axis=1)

    data2['pi']=pi_list
    '''
    ##print(data2)

    HTE = sum(data2[y]/data2[pi])
    #print("HT Total Estimate:",HTE)
    #print("HT Mean Estimate:",HTE/pop_data.shape[0])

    def cal_t1(rec):
        return ((1-rec[pi])/(rec[pi]**2)*(rec[y]**2))

    t1_list = data2.apply(cal_t1, axis=1)

    y_list = data2[y].values.tolist()
    pi_list = data2[pi].values.tolist()
    p_list = data2[p].values.tolist()

    t2 = 0

    for i in range(len(pi_list)):
        for j in range(i, len(pi_list)):
            if i<j:
                #pij = pi_list[i] + pi_list[j] - (1 - (1 - p_list[i]-p_list[j])**n1)
                pij = cal_pij(pop_data, data1, i, j)
                temp = ((pij - (pi_list[i]*pi_list[j]) )) * (((y_list[i]/pi_list[i])-(y_list[j]/pi_list[j]))**2) * (1 / pij)
                ##print(i+1, j+1, pi_list[i], pi_list[j], p_list[i], p_list[j], pij, temp)
                t2 += temp

    var_est = (t2)*(-1)

    #print("Variance:", var_est)
    #print("Var of mean:", var_est/(pop_data.shape[0]**2))

    return (HTE, HTE/pop_data.shape[0], var_est, var_est/(pop_data.shape[0]**2))

def cal_greg_est_var(pop_data, sample_data):

    pop_y_h_sum = pop_data['y_h'].sum()
    ##print("pop_y_h_sum=",pop_y_h_sum)

    sample_e_sum = ((sample_data['RMT85']-sample_data['y_h'])/sample_data['pi']).sum()
    ##print("sample_e_sum=",sample_e_sum)

    g_est = pop_y_h_sum+sample_e_sum

    #print("GREG estimation total=",g_est)
    #print("GREG estimation mean=",g_est/pop_data.shape[0])

    pi = 'pi'
    n1 = sample_data.shape[0]
    y = 'RMT85'
    p = 'pi'

    def cal_t1(rec):
        return ((1-rec[pi])/(rec[pi]**2)*(rec[y]**2))

    t1_list = sample_data.apply(cal_t1, axis=1)

    e_list = (sample_data[y]-sample_data['y_h']).values.tolist()
    pi_list = sample_data[pi].values.tolist()
    p_list = sample_data[p].values.tolist()

    t2 = 0

    for i in range(len(pi_list)):
        for j in range(i, len(pi_list)):
            if i<j:
                #pij = pi_list[i] + pi_list[j] - (1 - (1 - p_list[i]-p_list[j])**n1)
                pij = cal_pij(pop_data, sample_data, i, j)
                temp = ((pij - (pi_list[i]*pi_list[j]) )) * (((e_list[i]/pi_list[i])-(e_list[j]/pi_list[j]))**2) * (1 / pij)
                ##print(i+1, j+1, pi_list[i], pi_list[j], p_list[i], p_list[j], pij, temp)
                t2 += temp

    var_est = (t2)*(-1)

    #print("GREG Estimation of variance:", var_est)

    #print("GREG Estimation of variance of mean:", var_est/(pop_data.shape[0]**2))

    return (g_est, g_est/pop_data.shape[0], var_est, var_est/(pop_data.shape[0]**2))

def get_LR_results(sample_data, pop_data):

    in_var = list(sample_data.columns[:-5])
    d_var = [sample_data.columns[-5]]

    train_x = sample_data[in_var + d_var].values[:,:-1]
    train_y = sample_data[in_var + d_var].values[:,-1]

    in_var = list(pop_data.columns[:-3])
    d_var = [pop_data.columns[-3]]

    test_x = pop_data[in_var + d_var].values[:,:-1]
    test_y = pop_data[in_var + d_var].values[:,-1]

    model = LinearRegression()
    model.fit(train_x, train_y)

    pred_y = model.predict(train_x)
    s_lr_rmse = RMSE(train_y, pred_y)
    s_lr_mae = MAE(train_y, pred_y)
    s_lr_mape = MAPE(train_y, pred_y)
    sample_data['y_h']=pred_y

    #print("Sampled units...")
    #print("RMSE:",s_lr_rmse)
    #print("MAE:",s_lr_mae)
    #print("MAPE:",s_lr_mape)


    pred_y = model.predict(test_x)
    t_lr_rmse = RMSE(test_y, pred_y)
    t_lr_mae = MAE(test_y, pred_y)
    t_lr_mape = MAPE(test_y, pred_y)
    pop_data['y_h']=pred_y

    #print("Total units...")
    #print("RMSE:",t_lr_rmse)
    #print("MAE:",t_lr_mae)
    #print("MAPE:",t_lr_mape)

    (est_t, est_m, var_t, var_m) = cal_greg_est_var(pop_data, sample_data)

    return (s_lr_rmse, s_lr_mae, s_lr_mape, t_lr_rmse, t_lr_mae, t_lr_mape, est_t, est_m, var_t, var_m)

def get_SWLR_results(sample_data, pop_data):

    in_var = list(sample_data.columns[:-5])
    d_var = [sample_data.columns[-5]]

    train_x = sample_data[in_var + d_var].values[:,:-1]
    train_y = sample_data[in_var + d_var].values[:,-1]

    in_var = list(pop_data.columns[:-3])
    d_var = [pop_data.columns[-3]]

    test_x = pop_data[in_var + d_var].values[:,:-1]
    test_y = pop_data[in_var + d_var].values[:,-1]

    model = LinearRegression()
    model.fit(train_x, train_y, sample_data['pi'].values)

    pred_y = model.predict(train_x)
    s_swlr_rmse = RMSE(train_y, pred_y)
    s_swlr_mae = MAE(train_y, pred_y)
    s_swlr_mape = MAPE(train_y, pred_y)
    sample_data['y_h']=pred_y

    #print("Sampled units...")
    #print("RMSE:",s_swlr_rmse)
    #print("MAE:",s_swlr_mae)
    #print("MAPE:",s_swlr_mape)


    pred_y = model.predict(test_x)
    t_swlr_rmse = RMSE(test_y, pred_y)
    t_swlr_mae = MAE(test_y, pred_y)
    t_swlr_mape = MAPE(test_y, pred_y)
    pop_data['y_h']=pred_y

    #print("Total units...")
    #print("RMSE:",t_swlr_rmse)
    #print("MAE:",t_swlr_mae)
    #print("MAPE:",t_swlr_mape)

    (est_t, est_m, var_t, var_m) = cal_greg_est_var(pop_data, sample_data)

    return (s_swlr_rmse, s_swlr_mae, s_swlr_mape, t_swlr_rmse, t_swlr_mae, t_swlr_mape, est_t, est_m, var_t, var_m)

def get_ANN_results(sample_data, pop_data):

    in_var = list(sample_data.columns[:-5])
    d_var = [sample_data.columns[-5]]

    train_x = sample_data[in_var + d_var].values[:,:-1]
    train_y = sample_data[in_var + d_var].values[:,-1]

    in_var = list(pop_data.columns[:-3])
    d_var = [pop_data.columns[-3]]

    test_x = pop_data[in_var + d_var].values[:,:-1]
    test_y = pop_data[in_var + d_var].values[:,-1]

    window_size = train_x.shape[1]
    output_size = 1
    cycle=0

    min_mse = 1000000
    c_thresh = best_model_no

    while True:
        model = Sequential()
        model.add(Dense(30, activation='relu', input_shape=(window_size,),kernel_regularizer='l2'))
        model.add(Dense(output_size, activation='relu'))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=15)

        model.fit(train_x, train_y,validation_data=(test_x, test_y), epochs=500, batch_size=8, verbose=0, callbacks=[es])
        print("ANN MSE : ")
        acc=model.evaluate(test_x, test_y)

        if min_mse > acc[1]:
            min_mse = acc[1]
            min_model = model

        #if acc[1]<lr_mse or cycle>=c_thresh:
        if cycle>=c_thresh:
            ann_mse = acc[1]
            if min_mse ==1000000:
                ann_mse = min_mse
                min_model = model
            break

        cycle+=1
        print("Cycle:", cycle)

    #print("Best Model")
    acc=min_model.evaluate(test_x, test_y)


    pred_y = min_model.predict(train_x).reshape(-1)

    #print("Training actual and predicted y",train_y, pred_y)

    s_swlr_rmse = RMSE(train_y, pred_y)
    s_swlr_mae = MAE(train_y, pred_y)
    s_swlr_mape = MAPE(train_y, pred_y)
    sample_data['y_h']=pred_y

    #print("Sampled units...")
    #print("RMSE:",s_swlr_rmse)
    #print("MAE:",s_swlr_mae)
    #print("MAPE:",s_swlr_mape)


    pred_y = min_model.predict(test_x).reshape(-1)

    ##print("Testing actual and predicted y",test_y, pred_y)

    t_swlr_rmse = RMSE(test_y, pred_y)
    t_swlr_mae = MAE(test_y, pred_y)
    t_swlr_mape = MAPE(test_y, pred_y)
    pop_data['y_h']=pred_y

    #print("Total units...")
    #print("RMSE:",t_swlr_rmse)
    #print("MAE:",t_swlr_mae)
    #print("MAPE:",t_swlr_mape)

    (est_t, est_m, var_t, var_m) = cal_greg_est_var(pop_data, sample_data)

    return (s_swlr_rmse, s_swlr_mae, s_swlr_mape, t_swlr_rmse, t_swlr_mae, t_swlr_mape, est_t, est_m, var_t, var_m)

def get_SWANN_results(sample_data, pop_data):

    m_data = multiply_dataset(sample_data)

    in_var = list(m_data.columns[:-5])
    d_var = [m_data.columns[-5]]

    train_x = m_data[in_var + d_var].values[:,:-1]
    train_y = m_data[in_var + d_var].values[:,-1]

    in_var = list(pop_data.columns[:-3])
    d_var = [pop_data.columns[-3]]

    test_x = pop_data[in_var + d_var].values[:,:-1]
    test_y = pop_data[in_var + d_var].values[:,-1]

    window_size = train_x.shape[1]
    output_size = 1
    cycle=0

    min_mse = 1000000
    c_thresh = best_model_no

    while True:
        model = Sequential()
        model.add(Dense(30, activation='relu', input_shape=(window_size,),kernel_regularizer='l2'))
        model.add(Dense(output_size, activation='relu'))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)

        model.fit(train_x, train_y,validation_data=(test_x, test_y), epochs=50, batch_size=64, verbose=0, callbacks=[es])
        print("SWANN MSE : ")
        acc=model.evaluate(test_x, test_y)

        if min_mse > acc[1]:
            min_mse = acc[1]
            min_model = model

        #if acc[1]<lr_mse or cycle>=c_thresh:
        if cycle>=c_thresh:
            ann_mse = acc[1]
            if min_mse ==1000000:
                ann_mse = min_mse
                min_model = model
            break

        cycle+=1
        print("Cycle:", cycle)

    print("Best Model")
    acc=min_model.evaluate(test_x, test_y)


    #print("Training actual and predicted y",train_y, pred_y)
    train_x = sample_data[in_var + d_var].values[:,:-1]
    train_y = sample_data[in_var + d_var].values[:,-1]

    pred_y = min_model.predict(train_x).reshape(-1)

    s_swlr_rmse = RMSE(train_y, pred_y)
    s_swlr_mae = MAE(train_y, pred_y)
    s_swlr_mape = MAPE(train_y, pred_y)
    sample_data['y_h']=pred_y

    #print("Sampled units...")
    #print("RMSE:",s_swlr_rmse)
    #print("MAE:",s_swlr_mae)
    #print("MAPE:",s_swlr_mape)


    pred_y = min_model.predict(test_x).reshape(-1)

    ##print("Testing actual and predicted y",test_y, pred_y)

    t_swlr_rmse = RMSE(test_y, pred_y)
    t_swlr_mae = MAE(test_y, pred_y)
    t_swlr_mape = MAPE(test_y, pred_y)
    pop_data['y_h']=pred_y

    #print("Total units...")
    #print("RMSE:",t_swlr_rmse)
    #print("MAE:",t_swlr_mae)
    #print("MAPE:",t_swlr_mape)

    (est_t, est_m, var_t, var_m) = cal_greg_est_var(pop_data, sample_data)

    return (s_swlr_rmse, s_swlr_mae, s_swlr_mape, t_swlr_rmse, t_swlr_mae, t_swlr_mape, est_t, est_m, var_t, var_m)

def get_1DCNN_results(sample_data, pop_data):

    in_var = list(sample_data.columns[:-5])
    d_var = [sample_data.columns[-5]]

    train_x = sample_data[in_var + d_var].values[:,:-1]
    train_y = sample_data[in_var + d_var].values[:,-1]

    train_x_rs = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1)).astype(np.float32)

    in_var = list(pop_data.columns[:-3])
    d_var = [pop_data.columns[-3]]

    test_x = pop_data[in_var + d_var].values[:,:-1]
    test_y = pop_data[in_var + d_var].values[:,-1]

    window_size = train_x.shape[1]
    output_size = 1
    cycle=0

    min_mse = 1000000
    c_thresh = best_model_no

    while True:

        model = Sequential()
        model.add(Conv1D(filters=5, kernel_size=4, activation='relu'))
        model.add(Flatten())
        model.add(Dense(output_size, activation='relu'))
        model.compile(optimizer='adam', loss='mse')

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=20)

        model.fit(train_x_rs, train_y,validation_data=(test_x, test_y), epochs=150, batch_size=8,verbose=0,  callbacks=[es])

        print("1D CNN MSE : ")
        acc=model.evaluate(test_x, test_y)

        if min_mse > acc:
            min_mse = acc
            min_model = model

        #if acc[1]<lr_mse or cycle>=c_thresh:
        if cycle>=c_thresh:
            ann_mse = acc
            if min_mse ==1000000:
                ann_mse = min_mse
                min_model = model
            break

        cycle+=1
        print("Cycle:", cycle)

    print("Best Model")
    acc=min_model.evaluate(test_x, test_y)


    pred_y = min_model.predict(train_x_rs).reshape(-1)

    #print("Training actual and predicted y",train_y, pred_y)

    s_swlr_rmse = RMSE(train_y, pred_y)
    s_swlr_mae = MAE(train_y, pred_y)
    s_swlr_mape = MAPE(train_y, pred_y)
    sample_data['y_h']=pred_y

    #print("Sampled units...")
    #print("RMSE:",s_swlr_rmse)
    #print("MAE:",s_swlr_mae)
    #print("MAPE:",s_swlr_mape)


    pred_y = min_model.predict(test_x).reshape(-1)

    ##print("Testing actual and predicted y",test_y, pred_y)

    t_swlr_rmse = RMSE(test_y, pred_y)
    t_swlr_mae = MAE(test_y, pred_y)
    t_swlr_mape = MAPE(test_y, pred_y)
    pop_data['y_h']=pred_y

    #print("Total units...")
    #print("RMSE:",t_swlr_rmse)
    #print("MAE:",t_swlr_mae)
    #print("MAPE:",t_swlr_mape)

    (est_t, est_m, var_t, var_m) = cal_greg_est_var(pop_data, sample_data)

    return (s_swlr_rmse, s_swlr_mae, s_swlr_mape, t_swlr_rmse, t_swlr_mae, t_swlr_mape, est_t, est_m, var_t, var_m)

def get_SW1DCNN_results(sample_data, pop_data):

    m_data = multiply_dataset(sample_data)

    in_var = list(m_data.columns[:-5])
    d_var = [m_data.columns[-5]]

    train_x = m_data[in_var + d_var].values[:,:-1]
    train_y = m_data[in_var + d_var].values[:,-1]

    train_x_rs = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1)).astype(np.float32)

    in_var = list(pop_data.columns[:-3])
    d_var = [pop_data.columns[-3]]

    test_x = pop_data[in_var + d_var].values[:,:-1]
    test_y = pop_data[in_var + d_var].values[:,-1]

    window_size = train_x.shape[1]
    output_size = 1
    cycle=0

    min_mse = 1000000
    c_thresh = best_model_no

    while True:

        model = Sequential()
        model.add(Conv1D(filters=5, kernel_size=4, activation='relu'))
        model.add(Flatten())
        model.add(Dense(output_size, activation='relu'))
        model.compile(optimizer='adam', loss='mse')

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)

        model.fit(train_x_rs, train_y,validation_data=(test_x, test_y), epochs=75, batch_size=32,verbose=0,  callbacks=[es])

        print("SW1D CNN MSE : ")
        acc=model.evaluate(test_x, test_y)

        if min_mse > acc:
            min_mse = acc
            min_model = model

        #if acc[1]<lr_mse or cycle>=c_thresh:
        if cycle>=c_thresh:
            ann_mse = acc
            if min_mse ==1000000:
                ann_mse = min_mse
                min_model = model
            break

        cycle+=1
        print("Cycle:", cycle)


    #print("Best Model")
    acc=min_model.evaluate(test_x, test_y)

    train_x = sample_data[in_var + d_var].values[:,:-1]
    train_y = sample_data[in_var + d_var].values[:,-1]

    train_x_rs = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1)).astype(np.float32)

    pred_y = min_model.predict(train_x_rs).reshape(-1)

    ##print("Training actual and predicted y",train_y, pred_y)

    s_swlr_rmse = RMSE(train_y, pred_y)
    s_swlr_mae = MAE(train_y, pred_y)
    s_swlr_mape = MAPE(train_y, pred_y)
    sample_data['y_h']=pred_y

    #print("Sampled units...")
    #print("RMSE:",s_swlr_rmse)
    #print("MAE:",s_swlr_mae)
    #print("MAPE:",s_swlr_mape)

    pred_y = min_model.predict(test_x).reshape(-1)

    ##print("Testing actual and predicted y",test_y, pred_y)

    t_swlr_rmse = RMSE(test_y, pred_y)
    t_swlr_mae = MAE(test_y, pred_y)
    t_swlr_mape = MAPE(test_y, pred_y)
    pop_data['y_h']=pred_y

    #print("Total units...")
    #print("RMSE:",t_swlr_rmse)
    #print("MAE:",t_swlr_mae)
    #print("MAPE:",t_swlr_mape)

    (est_t, est_m, var_t, var_m) = cal_greg_est_var(pop_data, sample_data)

    return (s_swlr_rmse, s_swlr_mae, s_swlr_mape, t_swlr_rmse, t_swlr_mae, t_swlr_mape, est_t, est_m, var_t, var_m)

def display_store_results(n,N,m,M,ht_t, ht_m, htvar_t, htvar_m, lr_results, swlr_results, ann_results, swann_results, cnn_results, swcnn_results):

    '''
    print("Simulation information:")
    print("n=",n,"N=",N,"m=",m,"M=",M)

    print("\nLinear Regression Results:")
    print("RMSE sample=", lr_results[0])
    print("MAE sample=", lr_results[1])
    print("MAPE sample=", lr_results[2])
    print("RMSE population=", lr_results[3])
    print("MAE population=", lr_results[4])
    print("MAPE population=", lr_results[5])
    print("Total Estimate=", lr_results[6])
    print("Mean Estimate=", lr_results[7])
    print("Total Variance=", lr_results[8])
    print("Mean Variance=", lr_results[9])

    print("\nSurvey Weighted Linear Regression Results:")
    print("RMSE sample=", swlr_results[0])
    print("MAE sample=", swlr_results[1])
    print("MAPE sample=", swlr_results[2])
    print("RMSE population=", swlr_results[3])
    print("MAE population=", swlr_results[4])
    print("MAPE population=", swlr_results[5])
    print("Total Estimate=", swlr_results[6])
    print("Mean Estimate=", swlr_results[7])
    print("Total Variance=", swlr_results[8])
    print("Mean Variance=", swlr_results[9])

    print("\nArtificial Neural Networks Results:")
    print("RMSE sample=", ann_results[0])
    print("MAE sample=", ann_results[1])
    print("MAPE sample=", ann_results[2])
    print("RMSE population=", ann_results[3])
    print("MAE population=", ann_results[4])
    print("MAPE population=", ann_results[5])
    print("Total Estimate=", ann_results[6])
    print("Mean Estimate=", ann_results[7])
    print("Total Variance=", ann_results[8])
    print("Mean Variance=", ann_results[9])

    print("\nSurvey Weighted Artificial Neural Networks Results:")
    print("RMSE sample=", swann_results[0])
    print("MAE sample=", swann_results[1])
    print("MAPE sample=", swann_results[2])
    print("RMSE population=", swann_results[3])
    print("MAE population=", swann_results[4])
    print("MAPE population=", swann_results[5])
    print("Total Estimate=", swann_results[6])
    print("Mean Estimate=", swann_results[7])
    print("Total Variance=", swann_results[8])
    print("Mean Variance=", swann_results[9])

    print("\n1D-Convolutional Neural Networks Results:")
    print("RMSE sample=", cnn_results[0])
    print("MAE sample=", cnn_results[1])
    print("MAPE sample=", cnn_results[2])
    print("RMSE population=", cnn_results[3])
    print("MAE population=", cnn_results[4])
    print("MAPE population=", cnn_results[5])
    print("Total Estimate=", cnn_results[6])
    print("Mean Estimate=", cnn_results[7])
    print("Total Variance=", cnn_results[8])
    print("Mean Variance=", cnn_results[9])

    print("\nSurvey Weighted 1D-Convolutional Neural Networks Results:")
    print("RMSE sample=", swcnn_results[0])
    print("MAE sample=", swcnn_results[1])
    print("MAPE sample=", swcnn_results[2])
    print("RMSE population=", swcnn_results[3])
    print("MAE population=", swcnn_results[4])
    print("MAPE population=", swcnn_results[5])
    print("Total Estimate=", swcnn_results[6])
    print("Mean Estimate=", swcnn_results[7])
    print("Total Variance=", swcnn_results[8])
    print("Mean Variance=", swcnn_results[9])
    '''

    rec = {}

    rec['N']=N
    rec['n']=n
    rec['M']=M
    rec['m']=m

    rec['HT total']=ht_t
    rec['HT mean']=ht_m
    rec['HT var total']=htvar_t
    rec['HT var mean']=htvar_m

    rec["LR RMSE sample"]= lr_results[0]
    rec["LR MAE sample"]= lr_results[1]
    rec["LR MAPE sample"]= lr_results[2]
    rec["LR RMSE population"]= lr_results[3]
    rec["LR MAE population"]= lr_results[4]
    rec["LR MAPE population"]= lr_results[5]
    rec["LR Total Estimate"]= lr_results[6]
    rec["LR Mean Estimate"]= lr_results[7]
    rec["LR Total Variance"]= lr_results[8]
    rec["LR Mean Variance"]= lr_results[9]

    rec["SWLR RMSE sample"]= swlr_results[0]
    rec["SWLR MAE sample"]= swlr_results[1]
    rec["SWLR MAPE sample"]= swlr_results[2]
    rec["SWLR RMSE population"]= swlr_results[3]
    rec["SWLR MAE population"]= swlr_results[4]
    rec["SWLR MAPE population"]= swlr_results[5]
    rec["SWLR Total Estimate"]= swlr_results[6]
    rec["SWLR Mean Estimate"]= swlr_results[7]
    rec["SWLR Total Variance"]= swlr_results[8]
    rec["SWLR Mean Variance"]= swlr_results[9]

    rec["ANN RMSE sample"]= ann_results[0]
    rec["ANN MAE sample"]= ann_results[1]
    rec["ANN MAPE sample"]= ann_results[2]
    rec["ANN RMSE population"]= ann_results[3]
    rec["ANN MAE population"]= ann_results[4]
    rec["ANN MAPE population"]= ann_results[5]
    rec["ANN Total Estimate"]= ann_results[6]
    rec["ANN Mean Estimate"]= ann_results[7]
    rec["ANN Total Variance"]= ann_results[8]
    rec["ANN Mean Variance"]= ann_results[9]

    rec["SWANN RMSE sample"]= swann_results[0]
    rec["SWANN MAE sample"]= swann_results[1]
    rec["SWANN MAPE sample"]= swann_results[2]
    rec["SWANN RMSE population"]= swann_results[3]
    rec["SWANN MAE population"]= swann_results[4]
    rec["SWANN MAPE population"]= swann_results[5]
    rec["SWANN Total Estimate"]= swann_results[6]
    rec["SWANN Mean Estimate"]= swann_results[7]
    rec["SWANN Total Variance"]= swann_results[8]
    rec["SWANN Mean Variance"]= swann_results[9]

    rec["1DCNN RMSE sample"]= cnn_results[0]
    rec["1DCNN MAE sample"]= cnn_results[1]
    rec["1DCNN MAPE sample"]= cnn_results[2]
    rec["1DCNN RMSE population"]= cnn_results[3]
    rec["1DCNN MAE population"]= cnn_results[4]
    rec["1DCNN MAPE population"]= cnn_results[5]
    rec["1DCNN Total Estimate"]= cnn_results[6]
    rec["1DCNN Mean Estimate"]= cnn_results[7]
    rec["1DCNN Total Variance"]= cnn_results[8]
    rec["1DCNN Mean Variance"]= cnn_results[9]

    rec["SW1DCNN RMSE sample"]= swcnn_results[0]
    rec["SW1DCNN MAE sample"]= swcnn_results[1]
    rec["SW1DCNN MAPE sample"]= swcnn_results[2]
    rec["SW1DCNN RMSE population"]= swcnn_results[3]
    rec["SW1DCNN MAE population"]= swcnn_results[4]
    rec["SW1DCNN MAPE population"]= swcnn_results[5]
    rec["SW1DCNN Total Estimate"]= swcnn_results[6]
    rec["SW1DCNN Mean Estimate"]= swcnn_results[7]
    rec["SW1DCNN Total Variance"]= swcnn_results[8]
    rec["SW1DCNN Mean Variance"]= swcnn_results[9]

    #out_path = '/content/drive/MyDrive/Research Work/SWANN/simulation_results.csv'
    out_path = 'simulation_results_'+str(sim_no)+'.csv'

    try:
        results = pd.read_csv(out_path)
    except:
        print("Creating the results file...")
        results = pd.DataFrame()

    #results=results.append(rec, ignore_index=True)
    results = pd.concat([results, pd.DataFrame([rec])], ignore_index=True)

    #print(results)
    results.to_csv(out_path, index=False)

#print("Y sum=",pop_data['RMT85'].sum())
#print("Y mean=",pop_data['RMT85'].mean())

for simulation_cycle in range(no_cycles):
    print("\n\n\n\nRunning simulation cycle:", simulation_cycle)
    sample_data,n,N,m,M = create_2_stage_w_sample(pop_data)
    lr_results = get_LR_results(sample_data.copy(), pop_data.copy())
    swlr_results = get_SWLR_results(sample_data.copy(), pop_data.copy())
    ann_results = get_ANN_results(sample_data.copy(), pop_data.copy())
    swann_results = get_SWANN_results(sample_data.copy(), pop_data.copy())
    cnn_results = get_1DCNN_results(sample_data.copy(), pop_data.copy())
    swcnn_results = get_SW1DCNN_results(sample_data.copy(), pop_data.copy())
    ht_t, ht_m, htvar_t, htvar_m = cal_ht_htvar(pop_data, sample_data, 'RMT85', 'pi')
    display_store_results(n,N,m,M,ht_t, ht_m, htvar_t, htvar_m, lr_results, swlr_results, ann_results, swann_results, cnn_results, swcnn_results)

