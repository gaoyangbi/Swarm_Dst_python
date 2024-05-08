import torch
import sys
import numpy as np
import pandas as pd
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import time
from datetime import timedelta
import torch.utils
import torch.utils.data


#-------------------------定义一个数据格式的类，继承自torch.utils.data.Dataset
class Dataset_(torch.utils.data.Dataset) :
    def __init__(self, data_df) :   #-----------此处由于父类并没有init，因此不必使用super关键字
        self.label   = torch.from_numpy(data_df['Dst'].values).to(torch.float32)
        self.data    = torch.from_numpy(data_df[data_df.columns[:-1]].values).to(torch.float32)

    # --------复写父类的__getitem__
    def __getitem__(self, index) :
        batch_data  = self.get_batch_data(index)
        batch_label = self.get_batch_label(index)
        return   batch_data, batch_label        # 此处必须严格按顺序，先特征值，后标签值

    def classes(self) :
        return self.label
    
    def __len__(self) :
        return self.data.size(0)
    
    def get_batch_label(self,index) :
        return self.label[index]
    
    def get_batch_data(self, index) :
        return self.data[index]


#----------------------------保存数据，加载数据
class Config_finished() : #-----------------已经划分好训练集，验证集，测试集
    '''
    针对本次实验，由于在之前的步骤中，已将数据分为10组
    此次可直接用于封装已经划分好训练集，验证集，测试集的数据
    '''
    def __init__(self, data_dir_path, name, train_list, valid_list, test_list, batch_size, learning_rate, epoch) -> None:
        """
        data_dir_path     : string 数据文件所在文件夹路径
        name              : string 模型名字
        train_list        : lsit 训练集的编号  7个
        valid_list        : list 验证集的编号  1个
        test_list         : list 测试集合的编号  2个
        batch_size        : int 多少条数据组成一个batch
        learning_rate     : float 学习率
        epoch             : int 学习轮数
        train_loader, valid_loader, test_loader  : 训练数据、验证数据、测试数据
        """

        self.name           = name
        self.data_dir_path  = data_dir_path
        self.train_list     = train_list
        self.valid_list     = valid_list
        self.test_list      = test_list
        self.batch_size     = batch_size
        self.learning_rate  = learning_rate
        self.epoch          = epoch
        self.train_loader, self.valid_loader, self.test_loader = self.load_tdt()

    #------------------将读取的numpy文件转化为dataframe
    def transform(self, path_name) :
        data_npy   = np.load(path_name)
        data_npy   = np.delete(data_npy, 0, axis=1)  # 删除了第一行 也就是Spacecraft
        data_npy   = data_npy.astype(np.float32)
        df         = pd.DataFrame(data_npy)
        Config_finished.if_nan(df)
        df.columns = ['Timestamp', 'Latitude', 'Longitude', 'Radius', 'B_N', 'B_E', 'B_C',
                    'B_N_CHAOS-internal', 'B_E_CHAOS-internal', 'B_C_CHAOS-internal', 'Dst', 'QDLat', 'QDLon']
        df         = df.reindex(columns=['Timestamp', 'Latitude', 'Longitude', 'Radius', 'B_N', 'B_E', 'B_C',
                    'B_N_CHAOS-internal', 'B_E_CHAOS-internal', 'B_C_CHAOS-internal', 'QDLat', 'QDLon', 'Dst'])
        return df
    

    #------------------生成数据集集并封装成Dataloader类，需要读入数据集的选项
    def dataset_create(self, list) :
        '''
        list输入你想要生成总数据集的子数据集编号，在此次实验中是1-10
        '''
        i =  list[0]
        path_name           = self.data_dir_path + "database" + str(i).zfill(2) + ".npy"
        data_frame_finished = self.transform(path_name) 
        for i in list[1:] :
            path_name           = self.data_dir_path + "database" + str(i).zfill(2) + ".npy"
            data_frame          = self.transform(path_name) 
            data_frame_finished = pd.concat([data_frame_finished, data_frame])

        dataset  =  Dataset_(data_frame_finished)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
    
    #-------------------生成训练集、验证集、测试集
    def load_tdt(self) :
        train_loader = self.dataset_create(self.train_list)
        valid_loader = self.dataset_create(self.valid_list)
        test_loader  = self.dataset_create(self.test_list)
        return train_loader, valid_loader, test_loader

    #-------------------判断数据集是否有空数值 (输入数据格式为pd.dataframe) 
    #-------------------该方法为静态方法，可以不用通过实例化来使用
    @staticmethod
    def if_nan(dataframe) :
        if dataframe.isnull().any().any():
            emp = dataframe.isnull().any()
            print(emp[emp].index)
            print("Empty data exists")
            sys.exit(0) #---------程序正常退出，并进行变量清理等等