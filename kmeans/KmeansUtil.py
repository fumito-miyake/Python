# -*- coding: utf-8 -*-
import numpy as np
import dataIO as dt
import time as tm
import sys

sys.path.append('C:/Users/Dell/.spyder2-py3/MLUtil/')
import opt


INF = float("inf")

class k_means_opt( opt.OPT ):

    def __init__( self, data, cls_n, seed=tm.time(), prt=False ):
        self.data = np.array( data )
        self.cls_n = cls_n
        self.dim = len(data[0])
        self.size = len(data)
        self.prt = prt
        self.center = np.array( [ [1.0 for d in range(self.dim)] for c in range(self.cls_n) ] )
        self.MU = opt.MLUtil()

        super().__init__( x_dim=self.dim * self.cls_n, seed=seed )

    def enter_data( self, data ):
        self.data = np.array( data )
        self.dim = len(data[0])
        self.size = len(data)
        super().x_dim = self.dim * self.cls_n

    def init_set(self):
        self.len_d = [ self.size / self.cls_n for c in range(self.cls_n)]

        self.sum_dvec = [ self.data[c*self.dim + 1: c*(self.dim + 1) + 1] for c in range(self.cls_n) ]

        self.sum_dnrm = [ self.MU.norm2(self.data[c*self.dim + 1: c*(self.dim + 1) + 1]) for c in range(self.cls_n) ]


    def cal_center( self, pred ): # data:(size, dim)
        '''
        受け取ったデータ点(data)とその推定結果(pred)から，各クラスの中心点を再計算する
        '''
        center_keep = np.zeros_like( self.center ) # 新中心座標格納用のゼロ行列を生成(cls_n, dim)
        lb_n = [ 0 for c in range(self.cls_n) ] # 各クラスに属するデータ点カウント用のゼロベクトルを生成(clS_n)
        for i in range( self.size ): # 同じクラスと推定しているデータ点の各座標を足し合わせ，また各クラスに属するデータ点の個数をカウントする
            center_keep[ pred[i] ] += self.data[i]
            lb_n[ pred[i] ] += 1

        for c in range( self.cls_n ):
            for d in range( self.dim ):
                if lb_n[c] == 0: # クラスに属するデータ点がゼロ個の場合，そのクラスの前回の中心座標を採用する
                    center_keep[c][d] = self.center[c][d]
                else: # クラスに属するデータ点がゼロ個以上の場合，そのクラスに属するデータ点の重心をそのクラスの中心座標として採用する
                    center_keep[c][d] /= lb_n[c]

        for c in range( self.cls_n ):
            if lb_n[c] == 0: # クラスに属するデータ点がゼロ個の場合，そのクラスの前回の中心座標を採用する
                center_keep[c] = self.center[c].copy()
            else: # クラスに属するデータ点がゼロ個以上の場合，そのクラスに属するデータ点の重心をそのクラスの中心座標として採用する
                center_keep[c] /= lb_n[c]

        self.center = center_keep.copy()

    def count_menber( self, pred ):
        '''
        与えられた推定結果(pred)から，各クラスに何個ずつ要素が入っているかを数え，リストとして返す
        '''
        counter = [ 0 for c in range( self.cls_n ) ]
        for i in range( self.size ):
            counter[ pred[i] ] += 1

        return counter

    def relabel( self ): # data:(size, dim)
        '''
        与えられたデータ点(data)から，各データがどのクラスに属するかを推定して返す
        '''
        pred = [ 0 for i in range( self.size ) ] # 推定結果格納用のゼロベクトルを生成(size)
        for i in range( self.size ): # 各データ点について全クラス中心との距離を計算し，最も近い中心点にそのデータ点を所属させる
            min_dis = INF
            for c in range(self.cls_n):
                norm = np.linalg.norm( np.array(self.center[c]) - self.data[i] )
#                print(norm, np.array(self.center[c]) )
                if  norm < min_dis:
                    min_dis = norm
                    new_lb = c

            pred[i] = new_lb

        return pred

    def cal_loss( self, pred=None ): # data:(size, dim)
        '''
        データ点の属するクラスの中心点とそのデータ点の距離について，全データについて足し合わせたものを損失として返す
        '''
        if type(pred) == type(None): # 推定結果が渡されなかった場合は中心点の推定を行う
            pred = self.relabel()
        loss = 0.0
        for i in range( self.size ):
            loss += np.linalg.norm( self.center[ pred[i] ] - self.data[i] ) **2 # 2-normの2乗

        return loss

    def loss_grad( self, pred=None ):
        '''
        損失関数(cal_loss)を現在の中心点で微分した値を返す
        '''
        if type(pred) == type(None):
            pred = self.relabel()
        grad = np.zeros_like( self.center )

        for i in range( self.size ):
            grad[ pred[i] ] += 2 * ( self.center[ pred[i] ] - self.data[i] )

        return grad

    def best_init( self, f_T=10 ):
        '''
        f_T回初期値の生成を繰り返し，最も損失が小さかったものを初期値として返す
        '''
        best_loss = INF
        np.random.seed(10)
        for i in range( f_T ): # f_T回初期値の生成を繰り返し，最も損失が小さかったものを初期値として採用する
            pred = np.random.randint( low = 0, high = self.cls_n, size = self.size ) # 0~cls_nのいずれかの整数を要素として持つベクトルを生成(size)
            self.cal_center( pred ) # 推定結果に対する損失を計算
            loss = self.cal_loss( pred )

            if loss < best_loss:
                best_pred = pred
                best_loss = loss

        print("best :{}".format(best_loss))
        return best_pred

    def sep_vec( self, x=None ):
        '''
        parm array x: 重心の座標を(cls_n * dim)のかたちで格納したベクトル
        retun array center: (cls_n, dim)のかたちに分割した重心の行列
        '''
        return np.reshape( x, (self.cls_n, self.dim) )

    def mrg_mat( self, m=None ):
        '''
        parm array m: 重心の座標を(cls_n, dim)のかたちで格納した行列
        retun array x_center: (cls_n * dim)のかたちに結合した重心のベクトル
        '''
        return np.reshape( m, (self.cls_n * self.dim) )

    ###### Override ######

    def srg_min( self, x_org, info=None ):
        '''
        parm array x_org: 現在の予測点
        parm list info: surrogate関数を作成するために必要なパラメータ等(eg. L-lipschitz)
        parm int seed: 乱数を用いてsurrogate関数を作成するときのシード値 [if 0: 乱数を用いない]
        return array argmin_x: 作成したsurrogate関数の最小解
        '''
        self.center = self.sep_vec( x_org )
        pred = self.relabel()
        self.cal_center( pred )
        return self.mrg_mat( self.center )

    def smooth_srg_min( self, x_org, info=None, delta=1.0, ave_T=10 ):
        '''
        parm array x_org: 現在の予測点
        parm list info: surrogate関数を作成するために必要なパラメータ等(eg. L-lipschitz)
        parm float delta: smoothingの荒さを決めるパラメータ
        parm int ave_T: smoothingの際，ave_T回不偏推定を求め，その平均をsmoothing後の勾配とする
        return array argmin_x: 作成したsurrogate関数の最小解
        '''
        def data_sum( pred ):
            '''
            与えられた推定結果(pred)から，それぞれのクラスに属するデータ点の座標の総和を計算してarrayとして返す
            '''
            sum_l = np.array( [ [ 0.0 for d in range( self.dim ) ] for c in range( self.cls_n ) ] )
            for i in range( self.size ):
                sum_l[ pred[i] ] += self.data[i]

            return np.array( sum_l )

        # 初期値設定
        if info[ "t" ] == 1:
            info[ "a_scl0" ] = [ info[ "rho" ] / 2 for c in range( self.cls_n ) ]
            info[ "a_vec0" ] = 2 * x_org.copy()

        zero_vec = np.zeros_like( x_org, np.float )
        center_keep = self.center.copy()
        a_scl = [ 0.0 for c in range( self.cls_n ) ]
        a_vec = [ [ 0.0 for d in range( self.dim ) ] for c in range( self.cls_n ) ]
        a_scl0 = info[ "a_scl0" ].copy()
        a_vec0 = info[ "a_vec0" ].copy()

        # 乱数を使ってave_T回の試行を行う
        for t in range(ave_T):
            u = self.MU.make_a_data_ball( dim=self.x_dim, center=zero_vec, rad=delta )
            sep_u = self.sep_vec( u )
            self.center += sep_u
            pred = self.relabel()
            counter = np.array( self.count_menber( pred ) )
            a_scl += counter
            sum_l = data_sum( pred )
            a_vec += sum_l - counter[:, np.newaxis] * sep_u

            self.center = center_keep.copy()

        # ave_T回の結果を元に今回の解を求める
        a_scl /= ave_T
        a_vec *= 2 / ave_T
#        w = 1.0
        w = self.get_w( info[ "t" ] )

        x_t = []
        for c in range( self.cls_n ):
            deno = (1 - w)*a_scl0[ c ] + w*a_scl[ c ] # 分母
            if deno == 0: # 分母が0のときは前回の重心を採用する
                self.center[ c ] = center_keep[ c ]
            else:
                self.center[ c ] = ( (1 - w)*a_vec0[ c ] + w*a_vec[ c ] ) / ( 2 * deno )
            x_t += list( self.center[ c ] )

        info[ "a_scl0" ] = a_scl.copy()
        info[ "a_vec0" ] = a_vec.copy()

        return x_t


    def get_val( self, x ):
        '''
        parm array x: 値を求めたい座標
        return array val: 関数の値
        '''
        center_keep = self.center.copy()
#        print("val", x)
        self.center = self.sep_vec( x ).copy()
        loss = self.cal_loss()
        self.center = center_keep.copy()

        return loss

    def valid_err( self, loss_f, x0, x1, err ):
        '''
        parm func loss_f: 誤差関数 (in: array x0; out: float val)
        parm array x0: 前回の解
        parm array x1: 今回の解
        parm float err: 目標誤差
        return bool: 前回と今回の違いが目標誤差に収まっているか否か
        '''
        loss0 = loss_f( x0 )
        loss1 = loss_f( x1 )
        if ( abs( loss0 - loss1 ) / loss0 ) < err:
            if np.allclose(x0, x1): print("completely converge")
            return True
        return False

    def get_grad( self, x ):
        '''
        parm array x: 勾配を求めたい座標
        return array grad: 関数の勾配
        '''
        self.center = self.sep_vec( x )
        grad = self.loss_grad()
        return self.mrg_mat( grad )


    def get_w( self, w ):
        '''
        parm int t: 重みwを求めたい時刻
        return float eta: 時刻tにおけるw
        '''
        return 1 / ( w + 1 )

    def get_eta_gd( self, t ):
        '''
        parm int t: learning rateを求めたい時刻
        return float eta: 時刻tにおけるlearning rate
        '''
        return 1 / ( 30 * t + 180 ) # good
#        return 1 / ( 40 * t + 150 )
#        return 1 / ( 30 * t + 120 )

    def get_eta_sgd( self, t ):
        '''
        parm int t: learning rateを求めたい時刻
        return float eta: 時刻tにおけるlearning rate
        '''
        return 1 / ( 30* t + 90 ) # good
#        return 1 / ( 35 * t + 90 )
#        return 1 / ( 30 * t + 60 )

    def update_radius_sgd( self, delta ):
        '''
        parm float delta: 射影先の領域の半径を求めたいdelta
        return float radius: 次の射影先の領域の半径
        '''
        return delta * 20
#        return delta * 10 # good
#        return delta * 5
#        return 3900 good

    def update_radius_smm( self, delta ):
        '''
        parm float delta: 射影先の領域の半径を求めたいdelta
        return float radius: 次の射影先の領域の半径
        '''
        return delta * 10
#        return delta * 5 # good

    def update_ep_sgd( self, delta ):
        '''
        parm float delta: エポック内誤差を求めたいdelta
        return float epsilon: 次のエポック内誤差
        '''
        return delta * 2

    def update_ep_smm( self, delta ):
        '''
        parm float delta: エポック内誤差を求めたいdelta
        return float epsilon: 次のエポック内誤差
        '''
        return delta / ( 1000 * 1e6 )
#        return delta / ( 1000 * 1e4 )
#        return delta / (1000 * 1e3)
