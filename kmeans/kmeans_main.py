# -*- coding: utf-8 -*-

import numpy as np
#import matplotlib as plt
import dataIO as dt
import KmeansUtil as km
import config1 as cf
import math as mt
import pandas as pd
import sys
import csv
import time as tm
import os

sys.path.append('C:/Users/Dell/.spyder2-py3/MLUtil/')
import opt

INF = float("inf")

time_str_f = tm.strftime("%m.%d-%H.%M")
time_str_ex = [ tm.strftime("%m/%d"), tm.strftime("%H:%M") ]

path_t = cf.save_path + time_str_f + "(MNIST)/"

count = 2
while os.path.isdir(path_t) == True:
    path_l = path_t.split("_")
    if len( path_l ) <= 1:
        path_t = path_t[:-1] + "_1/"
    elif len( path_l ) == 2:
        path_t = path_l[0] + "_" + str(count) + "/"
        count += 1
    else:
        print("Invalid file name.")
        sys.exit()

os.makedirs(path_t)


image_w = image_h = 28
image_num = 10

cls_n = 10
dim = image_w * image_h

tmp = [ dt.PgmImage( image_w, image_h ) for i in range(image_num) ]
dt.LoadTemplateImages( tmp=tmp, file_n=cf.data_path, image_num=image_num, class_num=cls_n )

data = np.array( [ np.reshape( tmp[i].pixel, dim ) for i in range(image_num) ] )

dim_K = dt.cal_diameter( data )
print("\n------INFO------")
print("dim(K)={}".format(dim_K) )

mu_l, size_l, cv_l = cf.init_set2( cls_n=cls_n, data_NUM=image_num, dim=dim )

MLU = opt.MLUtil()
x_0 = MLU.make_data_ball( dim=dim, num=cls_n, center=mu_l, rad=min(size_l) )
x_0 = np.reshape( x_0, (cls_n * dim) )

K_ext = { "center": x_0, "radius": dim_K*2 }
tar_ep = 1e-5

##########  GD  ##########################################
GD_writer = { "path": path_t, "name": "GD" }

print("\n---------Normal_GD---------") # Smoothingなしk-means法でクラスタリングする
KM_GD = km.k_means_opt( data, cls_n=cls_n )
pred, loss_gd = KM_GD.GradOpt( mode="GD", x_0=x_0, T=50, ep=tar_ep, pr=0.5, K_ext=K_ext, epoch=10, delta=1.0, prt=True, write_info=GD_writer )


##########  SGD  ##########################################
SGD_writer = { "path": path_t, "name": "SGD" }

print("\n--------Smoothing_GD-------") # Smoothingありk-means法でクラスタリングする
delta_sgd = dim_K / 10
#ep_SGD = (delta **2) / 32
epoch_sgd = int( np.floor( np.log2( 4 * dim_K / tar_ep ) ) )

KM_SGD = km.k_means_opt( data, cls_n=cls_n )
pred, loss_sgd = KM_SGD.GradOpt( mode="SGD", x_0=x_0, T=100, pr=0.5, K_ext=K_ext, epoch=epoch_sgd, ave_T=4, delta=delta_sgd, prt=True, write_info=SGD_writer )


##########  MM  ###########################################
MM_writer = { "path": path_t, "name": "MM" }

print("\n---------Normal_MM---------") # Smoothingなしk-means法でクラスタリングする
KM_MM = km.k_means_opt( data, cls_n=cls_n )
pred, loss_mm = KM_MM.GradOpt( mode="MM", x_0=x_0, T=50, ep=tar_ep, pr=0.5, K_ext=K_ext, epoch=10, delta=1.0, prt=True, write_info=MM_writer )


##########  SMM  ##########################################
SMM_writer = { "path": path_t, "name": "SMM" }

print("\n--------Smoothing_MM-------") # Smoothingありk-means法でクラスタリングする
delta_smm = dim_K / 10
epoch_smm = int( np.floor( np.log2( 16 * dim_K / tar_ep) ) )

KM_SMM = km.k_means_opt( data, cls_n=cls_n )
pred, loss_smm = KM_SMM.GradOpt( mode="SMM", x_0=x_0, T=50, pr=0.5, K_ext=K_ext, epoch=epoch_smm, ave_T=10, delta=delta_smm, prt=True, write_info=SMM_writer )


##########  RESULT  #######################################
print("\ndim(K)={}".format(dim_K) )
print("loss_GD( Normal - Smoothing ) : {}".format( (loss_gd -loss_sgd) ))
print("loss_MM( Normal - Smoothing ) : {}".format( loss_mm - loss_smm) )

print("\n---------GD vs MM----------")
print("loss_Normal( GD - MM ) : {}".format( (loss_gd -loss_mm) ))
print("loss_Smoothing( GD - MM ) : {}".format( loss_sgd - loss_smm) )

# file書き込み
file, writer = dt.csv_w( cf.save_path, "RESULT(MNIST)", date=False )
log_l = []
log_l += [ time_str_ex[0], time_str_ex[1] ]
log_l += [ dim_K ]
log_l += [ " " ]
log_l += [ loss_gd ]
log_l += [ loss_sgd ]
log_l += [ delta_sgd ]
log_l += [ epoch_sgd ]
log_l += [ (loss_gd -loss_sgd) ]
log_l += [ " " ]
log_l += [ loss_mm ]
log_l += [ loss_smm ]
log_l += [ delta_smm ]
log_l += [ epoch_smm ]
log_l += [ (loss_mm - loss_smm) ]
log_l += [ " " ]
log_l += [ (loss_gd -loss_mm) ]
log_l += [ (loss_sgd - loss_smm) ]
writer.writerow( log_l )
file.close()

print( "\nwrote to ", path_t )