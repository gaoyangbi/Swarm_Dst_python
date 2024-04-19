import numpy as np


def location_pd(point,belt,cen):
    '''
    用于判断输入点的经度或纬度是否在所筛选的数据带
    '''
    result = False
    for i in range(0,len(cen)):
        if point >= cen[i]-belt/2.0 and point <= cen[i]+belt/2.0:
            result = True
            break   
    return result


#--------------------------------------------------
def QDread(path_):
    '''
    用于读取QDday文件中的数据，数据格式按照世界地磁数据中心发布的标准格式
    !!需要import numpy as np
    '''

    f = open(path_)
    line = f.readline()
    data_mat  =  np.ones((1,17),dtype=int)
    data_line  =  np.ones((1,17),dtype=int) 

    while line:
        line = f.readline()
        if line == '':
            break
        data_line[0,0] = int(line[0:4])
        data_line[0,1] = int(line[5:7])

        data_line[0,2] = int(line[8:10])
        data_line[0,3] = int(line[10:12])
        data_line[0,4] = int(line[12:14])
        data_line[0,5] = int(line[14:16])
        data_line[0,6] = int(line[16:18])

        data_line[0,7] = int(line[19:21])
        data_line[0,8] = int(line[21:23])
        data_line[0,9] = int(line[23:25])
        data_line[0,10] = int(line[25:27])
        data_line[0,11] = int(line[27:29])

        data_line[0,12] = int(line[30:32])
        data_line[0,13] = int(line[32:34])
        data_line[0,14] = int(line[34:36])
        data_line[0,15] = int(line[36:38])
        data_line[0,16] = int(line[38:40])
        data_mat = np.row_stack((data_mat,data_line))

    f.close()
    data_mat = np.delete(data_mat,0,axis=0)
    return data_mat









