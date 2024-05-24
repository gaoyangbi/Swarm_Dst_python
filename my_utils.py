import torch
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import time
from datetime import timedelta
import os


#-------------------------定义一个数据格式的类，继承自torch.utils.data.Dataset
class Dataset_(torch.utils.data.Dataset) :
    def __init__(self, data_df) :   #-----------此处由于父类并没有init，因此不必使用super关键字
        self.label   = torch.from_numpy(data_df['Dst'].values).to(torch.float)
        self.data    = torch.from_numpy(data_df[data_df.columns[:-1]].values).to(torch.float)

    # --------复写父类的__getitem__
    def __getitem__(self, index) :
        batch_data  = self.get_batch_data(index)
        batch_label = self.get_batch_label(index)
        return   batch_data, batch_label        # 此处必须严格与后面for index,(trains, labels) in enumerate(dataloader)中的括号内容对应
                                                # 先特征值，后标签值

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
        epoch             : int 输入学习轮数，在输入模型训练后，输出为实际训练的轮数
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
        df['B_N_minus'] = df['B_N'] - df['B_N_CHAOS-internal']
        df['B_E_minus'] = df['B_E'] - df['B_E_CHAOS-internal']
        df['B_C_minus'] = df['B_C'] - df['B_C_CHAOS-internal']
        df.drop(['Timestamp','B_N','B_E','B_C','B_N_CHAOS-internal','B_E_CHAOS-internal','B_C_CHAOS-internal',
                 'Radius','Latitude', 'Longitude', 'QDLat', 'QDLon'], axis=1, inplace=True)
        df         = df.reindex(columns=['B_N_minus', 'B_E_minus', 'B_C_minus', 'Dst'])
        
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



#--------------------------------回归模型的训练、测试和评估
class REG_model() :
    '''
    针对本次实验，对实验数据进行训练、测试和评估等等
    并可直接进行可视化操作
    '''

    def __init__(self, model, config) -> None:
        self.model = model
        self.config = config


    def run(self) :
        self.train_(self.model)

    def train_(self, model) :
        dev_best_loss = float('inf')
        strat_time = time.time()
        #-------------------------将模型切换为训练模型
        model.train()
        #------------------------定义优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        acc_list = [[], []]
        loss_list = [[], []]
        #-------------------------记录损失不下降的epoch数，到达20之后就直接退出 => 训练无效，再训练下去可能过拟合
        break_epoch = 0

        for epoch in range(self.config.epoch) :
            print('Epoch [{}/{}]'.format(epoch+1,self.config.epoch))
            for index, (trains, labels) in enumerate(self.config.train_loader) :
                # 归零
                model.zero_grad()      #---------------进行梯度归零
                # 得到预测结果，进行正向解算
                outputs = model(trains)
                # 计算MSELOSS函数
                loss_      = torch.nn.MSELoss() #-----------实例化一个对象
                loss_mean  = loss_(outputs, labels)
                # 反向传播loss
                loss_mean.backward()
                # 参数优化与参数更新
                optimizer.step()
                # 每迭代100次或跑完一个epoch，进行一次验证
                if (index % 100 == 0 and index != 0) or index == (len(self.config.train_loader) - 1) :
                    true = labels.detach().cpu().numpy()
                    # 预测数据
                    predict = outputs.detach().cpu().numpy()
                    # 计算训练集的准确度 决定系数R2   基于sklearn库 需要转化成nump格式
                    # 注意torch库有自带的计算决定系数的函数，输入值与sklearn库有所不同，需要注意
                    train_acc = r2_score(true, predict)
                    # 计算验证集的准确度 决定系数R2    注意事项同上
                    [dev_acc, dev_loss, dev_mse] = self.evaluate(model)
                    # 验证loss函数是否进步
                    if dev_loss < dev_best_loss :
                        dev_best_loss = dev_loss
                        improve = '*'
                        break_epoch = 0
                    else :
                        improve = ''
                        break_epoch += 1
                    # 计算消耗时间
                    time_dif = self.get_time_dif(start_time=strat_time)

                    # 输出阶段性成果 .item() 方法表示的是  将单元素tensor量 转化为float
                    msg = 'Iter:{0:>6},  Train Loss: {1:>5.3},  Train R2: {2:>6.3},  Val Loss: {3:>5.3},  Val R2: {4:>6.3},  Val Mse: {5:>6.3},  Time: {6} {7}'
                    print(msg.format(index, loss_mean.item(), train_acc, dev_loss, dev_acc, dev_mse, time_dif, improve))
                    # 每当跑完一个epoch，记录画图数据
                    if index == (len(self.config.train_loader) - 1) :
                        acc_list[0].append(train_acc)
                        acc_list[1].append(dev_acc)
                        loss_list[0].append(loss_mean.item())
                        loss_list[1].append(dev_loss)

                    # 转化为训练模式
                    model.train()
            # 设定早退，防止过拟合，如果20次验证，损失函数没有减小，直接退出训练
            if break_epoch > 20 :
                self.config.epoch = epoch + 1
                break
        # 测试
        self.test(model)
        # 画图 图片默认的保存地址是src文件夹上一级的images文件夹
        self.draw_curve(acc_list, loss_list, self.config.epoch)

                    
    def test(self, model) :
        start_time = time.time()
        # 测试集准确度R2，损失函数值，MSE
        [test_acc, test_loss, test_mse] = self.evaluate(model, test=True)
        msg = 'Test R2: {0:>5.3},  Test loss: {1:>6.3},  Test MSE: {2:>6.3}'
        print(msg.format(test_acc, test_loss, test_mse))
        time_dif = self.get_time_dif(start_time=start_time)
        print("Time usage:", time_dif)

    
    def evaluate(self, model, test=False) :
        '''
        test=False 使用验证集
        test=True  使用测试集
        '''
        # 转变模型模式
        model.eval()
        loss_total  = 0
        predict_all = np.array([], dtype=float)
        labels_all  = np.array([], dtype=float)

        if test :
            with torch.no_grad() :
                for index, (valids, labels) in enumerate(self.config.test_loader) :
                    outputs     = model(valids)
                    loss_       = torch.nn.MSELoss() #-----------实例化一个对象
                    loss_mean   = loss_(outputs, labels)
                    loss_total  += loss_mean
                    labels      = labels.detach().cpu().numpy()
                    predict     = outputs.detach().cpu().numpy()
                    labels_all  = np.append(labels_all, labels)
                    predict_all = np.append(predict_all, predict)

        else :
            with torch.no_grad() :
                for index, (valids, labels) in enumerate(self.config.valid_loader) :
                    outputs     = model(valids)
                    loss_       = torch.nn.MSELoss() #-----------实例化一个对象
                    loss_mean   = loss_(outputs, labels)
                    loss_total  += loss_mean
                    labels      = labels.detach().cpu().numpy()
                    predict     = outputs.detach().cpu().numpy()
                    labels_all  = np.append(labels_all, labels)
                    predict_all = np.append(predict_all, predict)

        dev_acc = r2_score(labels_all, predict_all)
        dev_mse = mean_squared_error(labels_all, predict_all)
        #-----------------注意：loss_total / len(self.config.test_loader) ->>>>   dev_loss
        #-----------------表示的是损失函数和的均值 
        if test :
            return dev_acc, loss_total / len(self.config.test_loader), dev_mse
        else :
            return dev_acc, loss_total / len(self.config.valid_loader), dev_mse


    # 计算时间损耗
    def get_time_dif(self, start_time) :
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))
    
    # 可视化输出
    def draw_curve(self, acc_list, loss_list, epochs) :
        #-----------------------创建保存输出文件的文件夹
        if not os.path.exists('../images/'):
            os.makedirs('../images/')
        
        x = range(0, epochs)
        y1 = loss_list[0]
        y2 = loss_list[1]
        y3 = acc_list[0]
        y4 = acc_list[1]
        plt.figure(figsize=(13, 13))
        plt.subplot(2, 1, 1)
        plt.plot(x, y1, color="blue", label="train_loss", linewidth=2)
        plt.plot(x, y2, color="orange", label="val_loss", linewidth=2)
        plt.title("Loss_curve", fontsize=20)
        plt.xlabel(xlabel="Epochs", fontsize=15)
        plt.ylabel(ylabel="Loss", fontsize=15)
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(x, y3, color="blue", label="train_acc", linewidth=2)
        plt.plot(x, y4, color="orange", label="val_acc", linewidth=2)
        plt.title("Acc_curve", fontsize=20)
        plt.xlabel(xlabel="Epochs", fontsize=15)
        plt.ylabel(ylabel="Accuracy", fontsize=15)
        plt.legend()
        plt.savefig("../images/"+self.config.name+"_Loss&acc.png")

