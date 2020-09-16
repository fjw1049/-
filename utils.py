from keras.layers import Input, Dense, LSTM, merge,Conv1D,Dropout,Bidirectional,Multiply
from keras.models import Model
import keras.backend as K
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
import matplotlib.pyplot as plt
import  pandas as pd
import  numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error

def del_col(data):
    """
    去除数据集中包含的str类型数据，全部转化为float类型数据
    @data_new: 返回float类型数据集
    """
    data_new=pd.DataFrame()
    columns=data.columns.values
    for col in columns:
        data_new[col]=pd.to_numeric(data[col],errors='coerce')
    data_new=data_new.dropna()
    return data_new

def calculate_variance(dps, moving_average):
    """
    计算协方差确定样本的偏离程度

    """
    variance = 0
    flag_list = moving_average.isnull()
    count = 0
    for index in range(len(dps)):
        if flag_list[index]:
            count += 1
            continue
        variance += (dps[index] - moving_average[index]) ** 2
    variance /= (len(dps) - count)

    return variance

def extract_outliers(error):
    """
    通过样本分布策略ewma,确定异常点位置信息

    """
    error_index=list()
    dps = pd.Series(error)
    ewma_line = pd.DataFrame.ewm(dps,span=4).mean()
    ewma_var = calculate_variance(dps, ewma_line)
    print(ewma_var)
    for index in ewma_line.index:
        if not (ewma_line[index] - 20.0*ewma_var <= dps[index] <= ewma_line[index] + 20.0*ewma_var):
            #print("出现的异常点", dps[index])
            error_index.append(index)
    return error_index


def attention_3d_block2(inputs, single_attention_vector=False):

    #实现注意力机制
    # 如果上一层是LSTM，需要return_sequences=True
    # inputs.shape = (batch_size, time_steps, input_dim)
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1))(a)
    # 乘上了attention权重，但是并没有求和，好像影响不大
    # 如果分类任务，进行Flatten展开就可以了
    # element-wise
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul



def create_dataset(dataset, look_back):
    '''
    对数据按照满足LSTM格式进行处理
    '''
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back),:]
        dataX.append(a)
        dataY.append(dataset[i + look_back,:])
    TrainX = np.array(dataX)
    Train_Y = np.array(dataY)

    return TrainX, Train_Y


def NormalizeMult(data):
    """

    @rtype: 多维归一化  返回数据和最大最小值
    """
    #normalize 用于反归一化
    data = np.array(data)
    normalize = np.arange(2*data.shape[1],dtype='float64')

    normalize = normalize.reshape(data.shape[1],2)
    #print(normalize.shape)
    for i in range(0,data.shape[1]):
        #第i列
        list = data[:,i]
        listlow,listhigh =  np.percentile(list, [0, 100])
        # print(i)
        normalize[i,0] = listlow
        normalize[i,1] = listhigh
        delta = listhigh - listlow
        if delta != 0:
            #第j行
            for j in range(0,data.shape[0]):
                data[j,i]  =  (data[j,i] - listlow)/delta
    #np.save("./normalize.npy",normalize)
    return  data,normalize




def attention_model():
    """

    @return: 构建注意力机制+LSTM的回归预测模型
    """
    INPUT_DIMS = 3
    TIME_STEPS = 1
    lstm_units = 30
    inputs = Input(shape=(TIME_STEPS, INPUT_DIMS))

    x = Conv1D(filters = 64, kernel_size = 1, activation = 'relu')(inputs)  #, padding = 'same'
    x = Dropout(0.3)(x)
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    lstm_out = Dropout(0.3)(lstm_out)
    attention_mul = attention_3d_block2(lstm_out)
    attention_mul = Flatten()(attention_mul)

    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model


def plot_picture(model,recommend=True):
    """

    @return: 训练集与测试集的训练及测试误差图
    """
    if recommend:
        plt.figure()
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.plot(model.history['loss'], label='train')
        plt.plot(model.history['val_loss'], label='test')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.title("误差图")


def prediction(model, test_x, test_y):
    """

    @return: 返回预测的RMSE和数组(预测值,真实值）
    """
    y_precdict = model.predict(test_x)
    RMSE = sqrt(mean_squared_error(y_precdict, test_y))
    y_precdict = y_precdict.reshape(-1, 1)
    test_y = test_y.reshape(-1, 1)
    data = np.hstack([y_precdict, test_y])
    data_df = pd.DataFrame(data, columns=['Precdict', 'True'])

    return RMSE, data_df


def FNormalizeMult(data,normalize):
    """
    @data: 预测值与真实值的DataFrame
    @rtype: 将预测和真实值进行反归一化
    """
    data = np.array(data)
    for i in range(0,data.shape[1]):
        listlow =  normalize[0,0]
        listhigh = normalize[0,1]
        delta = listhigh - listlow
        if delta != 0:
            #第j行
            for j in range(0,data.shape[0]):
                data[j,i]  =  data[j,i]*delta + listlow

    data_df = pd.DataFrame(data, columns=['Precdict', 'True'])

    return data_df