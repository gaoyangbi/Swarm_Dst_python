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
    此次直接使用整体数据集
    '''
    def __init__(self, data_path, name, batch_size, learning_rate, epoch) -> None:
        """
        data_path     : string 数据文件所在文件夹路径
        name              : string 模型名字
        batch_size        : int 多少条数据组成一个batch
        learning_rate     : float 学习率
        epoch             : int 输入学习轮数，在输入模型训练后，输出为实际训练的轮数
        train_loader, valid_loader, test_loader  : 训练数据、验证数据、测试数据
        """

        self.name           = name
        self.data_path  = data_path
        self.batch_size     = batch_size
        self.learning_rate  = learning_rate
        self.epoch          = epoch
        self.train_loader, self.valid_loader, self.test_loader = self.load_tdt()
        self.input_col, self.output_class = self.get_class()


    #-------------------生成训练集、验证集、测试集,并把数据封装成DataLoader类
    def load_tdt(self) :
        file_ = self.read_file()
        train_dev_test = self.cut_data(file_)
        tdt_loader     = [self.load_data(i) for i in train_dev_test]
        return tdt_loader[0], tdt_loader[1], tdt_loader[2]


    #------------------读文件
    def read_file(self) :
        file_ = pd.read_csv(self.data_path, encoding="utf-8-sig", index_col=None)
        file_.columns.values[-1] = "Dst"
        self.if_nan(file_)
        return file_

    #---------------------切割数据7：1：2 训练集，验证机，测试集
    def cut_data(self, data_df) :
        train_df, test_dev_df = train_test_split(data_df,test_size=0.3,random_state=1100)
        dev_df, test_df = train_test_split(test_dev_df, test_size=0.66, random_state=1100)
        return train_df,dev_df, test_df

    #------------------生成数据集集并封装成Dataloader类
    def load_data(self, data_df) :
        '''
        对dataframe进行封装
        '''
        dataset  =  Dataset_(data_df)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
    


    #-------------------判断数据集是否有空数值 (输入数据格式为pd.dataframe) 
    #-------------------该方法为静态方法，可以不用通过实例化来使用
    @staticmethod
    def if_nan(dataframe) :
        if dataframe.isnull().any().any():
            emp = dataframe.isnull().any()
            print(emp[emp].index)
            print("Empty data exists")
            sys.exit(0) #---------程序正常退出，并进行变量清理等等


    def get_class(self) :
        file_ = self.read_file()
        label = file_[file_.columns[-1]]
        label = list(set(list(label)))
        return file_.columns[:-1], label





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

