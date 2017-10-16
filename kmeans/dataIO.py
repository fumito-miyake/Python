# -*- coding: utf-8 -*-
#%matplotlib notebook
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import time as tm
import os
import PIL as pl

MAX_XY = [10, 10]
MIN_XY = [0, 0]
NUM=200

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

def make_data( size_l, loc_l, scale_l ):

#    size_l = [10, 10, 10]
#    loc_l = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
#    scale_l = [[0.35, 0.35], [0.35, 0.35], [0.35, 0.35]]

    size = 0
    for s in size_l:
        size += s

    data_x = [ np.empty(shape=[sz], dtype=np.float) for sz in size_l ]
    data_y = [ np.empty(shape=[sz], dtype=np.float) for sz in size_l ]

    for i, lc, sc, sz in zip( range(len(size_l)), loc_l, scale_l, size_l ):
        data_x[i] = np.random.normal(loc = lc[0], scale = sc[0], size = sz)
        data_y[i] = np.random.normal(loc = lc[1], scale = sc[1], size = sz)

    data_x_all = []
    data_y_all = []

    for dt_x, dt_y in zip(data_x, data_y):
        data_x_all += list( dt_x )
        data_y_all += list( dt_y )

    data_x_np = np.array(data_x_all)
    data_y_np = np.array(data_y_all)

    data_all = np.array([data_x_np, data_y_np]).T

    return data_all

def make_data_unif( max_x=MAX_XY[0], min_x=MIN_XY[0], max_y=MAX_XY[1], min_y=MIN_XY[1], num=NUM ):
    x_l = np.random.uniform( low = min_x, high = max_x, size = num )
    y_l = np.random.uniform( low = min_y, high = max_y, size = num )

    data = np.stack( (x_l, y_l) )
    return data.T

def make_a_data_ball( dim, rad=1.0 ):
    x = np.random.randn( dim )
    r = np.linalg.norm( x )
    if r != 0.:
        x /= r
    x *= rad
    reg_r = np.power( np.random.random(), 1. / (dim) )
    return x * reg_r

def make_data_ball( dim, num, rad=1.0 ):
    data_l = [ make_a_data_ball( dim, rad ) for i in range(num) ]
    return np.array( data_l )

def make_data_mul_normal( mu_l, cv_l, num=100, seed=5 ):
    np.random.seed(seed)
    data = np.random.multivariate_normal( mean = mu_l, cov = cv_l, size=num )
    return data

def cal_diameter( data ):
    num = len(data)
    max_dis = 0.0
    for i in range(num):
        for j in range(num):
            if i == j:
                continue
            dis = np.linalg.norm( data[i] - data[j] )
            if  dis > max_dis:
                max_dis = dis
    return max_dis

def plot_data( data, size_l ):
    p_color = [ "b", "g", "r", "c", "m", "y", "k" ]*( (len(data) // 7) + 1 )

    # 新規のウィンドウを描画
    fig = plt.figure(figsize=(10,10))
    # サブプロットを追加
    #ax1 = fig.add_subplot(1,1,1)

    data_T = data.T

    #print(data_T)
    p0 = 0
    for sz, cl in zip( size_l, p_color ):
        plt.plot( data_T[0][p0:p0 + sz], data_T[1][p0:p0 + sz], cl + "o")
        p0 += sz

    plt.grid()
    plt.show()

def plot_3D_point( X, Y, Z ):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter3D( X, Y, Z, c="y" )

def plot_3D_surf( x_range, y_range, func, title="" ): # x_range->[ x_min, x_max, interval ]
    print("now drowing...")

    X = np.arange( x_range[0], x_range[1], x_range[2] )
    Y = np.arange( y_range[0], y_range[1], y_range[2] )
    x_num = len( X )
    y_num = len( Y )

    X, Y = np.meshgrid(X, Y)
    Z = [ [ 0 for j in range( y_num ) ] for i in range( x_num ) ]

    for i in range( x_num ):
        for j in range( y_num ):
            Z[i][j] = func( X[i][j], Y[i][j], i, j )

    fig = plt.figure()
#    plt.tight_layout()
    fig.suptitle( title, fontsize=20)
    plt.subplots_adjust(top=1.1)
    ax = Axes3D(fig)
    ax.plot_surface( X, Y, Z, rstride=1, cstride=1, cmap='hot', vmin=0 )

def csv_w(path="", file_n="", mode="a", trmn="\n", date=True):
    path = str(path)
    file_n = str(file_n)

    if os.path.isdir( path ) == False:
        return None, None

    while os.path.isfile( path + file_n ) == True:
        file_n += "0"

    if date == True:
        file = open( path + file_n + tm.strftime("%m.%d-%H.%M") + ".csv", mode )
    else:
        file = open( path + file_n + ".csv", mode )

    writer = csv.writer( file, lineterminator=trmn )
    return file, writer


'''!
  @struct		PgmImage
  @brief		PGM画像を格納するための構造体
'''
class PgmImage():

    '''
	  @fn		__init__(int _width, int _height)
	  @brief	コンストラクタ
	  @param	[in] _width		画像の幅
	  @param	[in] _height	画像の高さ
    '''
    def __init__(self, _width, _height):
        self.width = _width
        self.height = _height
        self.label = -1

        self.pixel = np.empty( [self.height, self.width] )

'''!
  @fn		int LoadPgmImage(string file_n, PgmImage pgm, int label)
  @brief	PGM画像を読み込む
  @param	[in] file_n 	ファイル名
  @param	[in] image		読み込んだ画像を格納するためのPgmImageインスタンス
  @param	[in] label		読み込む画像のクラスラベル（デフォルトではラベル無し(label=-1)）
  @return	1:画像の読み込みに成功
  @n		0:画像の読み込みに失敗
'''
def LoadPgmImage( file_n, pgm, label = -1):

    try:
        img = pl.Image.open(file_n)
        pgm.pixel[:] = np.asarray( img )
        pgm.label = label
    except:
        print("The file \"{}\" was not opened.".format( file_n ) )
        return 0

    return 1

def LoadTemplateImages( tmp, file_n, image_num, class_num ):

    img_no = 0

    each_class_num = image_num // class_num
    for label in range( class_num ):
        for sample in range( each_class_num ):
            filename = file_n.format( label, sample )
            if sample == 0:
                print("\rLoading the file {}\n".format(filename) )
            if  LoadPgmImage(filename, tmp[img_no], label) == 0:
                return 0
            img_no += 1
    return 1

'''!
  @fn		int SavePgmImage(char file_n, PgmImage image)
  @brief	PGM画像として保存する
  @param	[in] filename	ファイル名
  @param	[in] image		保存する画像を格納したインスタンス
  @return	1:画像の保存に成功
  @n		0:画像の保存に失敗
'''
def SavePgmImage( filename, Pgm ):

    try:
        img = pl.Image.fromarray(np.uint8(Pgm.pixel))
        img.save( filename, 'pgm', quality = 100, optimize = True)

    except:
        print("The file \"{}\" was not opened.".format( filename ) )
        return 0

    return 1
