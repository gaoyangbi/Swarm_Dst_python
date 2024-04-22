globals().clear()
import cdflib
import os
from matplotlib.dates import YearLocator
import numpy as np
import myfunction
import logging


#----------------------确定筛选参数
belt_width_lat = 0.2
lat_cen        = [-40,-30,-20,20,30,40]  # 单位：度

belt_width_lon = 360.0
lon_cen        = [0]

Swarm_dir      = "../Swarm_Data/"
Dst_index      = "../Dst_index/index.dat"
QDday          = "../QDday/QDday.txt"
select_dir     = "../Swarm_select/"

#----------------------读取QDdays文件并保存到相应数据矩阵中

QDday_data = myfunction.QDread(QDday)
QDday_data = np.delete(QDday_data,range(0,11),axis=0)  # 将QDday数据的时间轴与Swarm数据对齐

#--------------------- Create a output logging
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)
logger_name = 'my_log.txt'
fh = logging.FileHandler(logger_name)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

#-----------------------创建保存输出文件的文件夹
if not os.path.exists(select_dir):
    os.makedirs(select_dir)


#----------------------获取所有数据的文件名
files = sorted(os.listdir(Swarm_dir))   # sorted函数用于排序


#------------------------文件戳 记录第几个文件
file_epoch = 1

for file_name in files:  
    file_path = Swarm_dir + file_name
    cdf_file = cdflib.CDF(file_path)
    var = cdf_file.cdf_info()
    #----------------------针对每一个文件中的数据进行筛选 
    data_mat_finish = np.ones((1,14))   # 初始化数据保存矩阵，对数据文件中的每一行进行筛选判断，若满足条件则加入其中 第一行是无意义的1，后续需要删除

    #-----------------------------

    raw_data = cdf_file.varget(var.zVariables[0])   #读入数据文件中的所有初始数据
    for i in ['Timestamp', 'Latitude', 'Longitude', 'Radius', 'B_NEC', 'B_NEC_CHAOS-internal', 'Dst', 'QDLat', 'QDLon']:
        data0 = cdf_file.varget(i)
        raw_data = np.column_stack((raw_data,data0))

    for i in range(0,raw_data.shape[0]):           # 进行数据筛选
        raw_data_test   = raw_data[i,:].reshape(1,-1)
        lat             = float(raw_data_test[0,2])
        lon             = float(raw_data_test[0,3])
        time_           = cdflib.cdfepoch.to_datetime(float(raw_data_test[0,1]))
        year            = int(str(time_[0])[0:4])
        mon             = int(str(time_[0])[5:7])
        day             = int(str(time_[0])[8:10])


        QDday_data_test = QDday_data[file_epoch-1,:].reshape(1,-1)
        QD_year         = QDday_data_test[0,0]
        QD_mon          = QDday_data_test[0,1]
        
        if (year != QD_year) or (mon != QD_mon) :   # 判断时间轴是否对齐，否则跳出循环
            print(f"{year}-{mon}Time is wrong!!!")
            break
        
        if not (day in QDday_data_test[0,2:17]):
            continue
        elif not myfunction.location_pd(lat,belt_width_lat,lat_cen):
            continue
        elif not myfunction.location_pd(lon,belt_width_lon,lon_cen):
            continue
        
        data_mat_finish = np.row_stack((data_mat_finish,raw_data_test))

    log_message    =  file_name+"has been calculated."
    logger.debug(log_message)
    #-----------------------------------------------
    file_epoch += 1
        
    #---------------------------------------------------save data
    data_mat_finish = np.delete(data_mat_finish,0,axis=0)   # 第一行需要删除
    save_file       = select_dir + 'Swarm_data_' + str(year).zfill(4) + str(mon).zfill(2) + '_finishedtest.npy'
    np.save(save_file,data_mat_finish)     
