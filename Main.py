import  pandas as pd
import  numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils import *
from attention_utils import *



#加载数据
if __name__ == '__main__':

    data = pd.read_csv(r'C:\Users\fangjianwen\Desktop\dataset.csv',sep=';')
    data = del_col(data)  #将数据集上有的数据为字母的行进行删除
    #print(data.columns)
    #print(data.shape)

    INPUT_DIMS = 3
    TIME_STEPS = 1
    lstm_units = 50

    #归一化
    data,normalize = NormalizeMult(data)
    pollut_data = data[:,0].reshape(len(data),1)

    train_X, _ = create_dataset(data,TIME_STEPS)
    _, train_Y = create_dataset(pollut_data,TIME_STEPS)
    print(train_X.shape,train_Y.shape)

    #划分数据集，基于模型进行训练
    SINGLE_ATTENTION_VECTOR = False
    train_x,test_x,train_y,test_y=train_test_split(train_X,train_Y,test_size=0.3,random_state=0)
    model= attention_model()
    model.summary()
    model.compile(optimizer='adam', loss='mse')
    history=model.fit([train_x], train_y, epochs=20, batch_size=64, validation_data=([test_x], test_y),verbose=2, shuffle=False)

    #绘制训练与测试误差图,保存在Img文件中
    plot_picture(history,recommend=True)
    plt.savefig(r".\Img\train_test_error.png")

    #模型预测和模型的反归一化得到预测值与真实值
    rmse,dataframe=prediction(model,test_x,test_y)
    print("验证集的RMSE：{name}".format(name=rmse))
    normal_data=FNormalizeMult(dataframe, normalize)
    #print(normal_data)

    #绘制预测值与真实值的残差图
    normal_data['Pre_Error']=normal_data['Precdict']-normal_data['True']
    plt.figure()
    normal_data['Pre_Error'].plot()
    plt.title("残差图")
    plt.savefig(r".\Img\residual.png")

    #通过ewma确认判定异常点的阈值
    residual=np.array(normal_data['Pre_Error'])
    index=extract_outliers(residual)
    print(len(index))