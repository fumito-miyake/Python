# -*- coding: utf-8 -*-

cls_n0 = 10
data_NUM0 = 1000
dim0 = 100

save_path = "C:/Users/Dell/Desktop/csv/"
data_path = "C:/Users/Dell/.spyder2-py3/pattern_rec/Images/TestSamples/{}-{:04d}.pgm"

def init_set2( cls_n=cls_n0, data_NUM=data_NUM0, dim=dim0 ):
    mu_l = [ 0 for d in range(dim) ]
    cv_l = [ [ 0 for d in range(dim) ] for d in range(dim) ]
    size_l = [ int(data_NUM // cls_n) for i in range(cls_n) ]
    dig = [ ( 10*d ) % 100 + 1 for d in range(dim) ]

    for d in range(dim):
            cv_l[d][d] = dig[d]

    if data_NUM % cls_n > 0:
        size_l[-1] +=  data_NUM % cls_n

    return mu_l, size_l, cv_l