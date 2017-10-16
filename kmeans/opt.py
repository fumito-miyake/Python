# -*- coding: utf-8 -*-
import numpy as np
import time as tm
import dataIO as dt

INF = float("inf")

class OPT(object):

    def __init__( self, x_dim, seed=tm.time() ):
        '''
        parm int x_dim: データの次元数
        parm int seed: 乱数のシード値
        '''
        self.x_dim = x_dim
        np.random.seed( int(seed) )
        self.MU = MLUtil()

    def GradOpt( self, mode="GD", x_0=None, T=20, ep=1e-5, pr=0.5, K_ext=None, epoch=10, ave_T=10, delta=1.0, f_info=[], prt=False, write_info=None ):
        '''
        parm str mode: 最適化手法の選択 [ GD, SGD, MM, SMM ]
        parm array x_0: 初期推定値 [if None: ランダムに選ぶ]
        parm T: 各エポックの最大繰り返し回数 [if None: 20]
        parm float ep: 目標誤差 [if None: 1e-5]
        pram float pr: 最大誤り確率 [if None: 0.5]
        parm list K_ext: 定義域を指定(0:array center, 1:float rad) [if []: center=zero, rad=inf]
        parm int epoch: 最大エポック数 [if None: f_infoから計算] [if None: 10]
        parm float delta: 初期smoothingパラメータ [if None: 1.0]
        parm dict f_info: 関数の情報(L: Lipschitz, R:grad_lip, sig:strongly-convex, ...) [if None: []]
        parm bool prt: 各エポックでの座標の表示の有無
        return array x_hat: epochエポック後の推定値
        return float val: epochエポック後の損失
        '''
        file = writer = None
        if type(write_info)==dict:
            if "path" in write_info and "name" in write_info:
                file, writer = dt.csv_w( write_info["path"], write_info["name"] )

        if type(K_ext) != dict:
            K_ext = { "center": np.zeros(self.x_dim), "radius": 1.0 }

        if type(x_0) != list and type(x_0) != np.ndarray:
            x_0 = self.MU.make_a_data_ball( dim=self.x_dim, center=K_ext["center"], rad=K_ext["radius"] )
            print("new first point")

        x_hat = x_0.copy()
        x_hat0 = x_hat.copy()

        for m in range( epoch ):
            if prt:
                print(">>epoch {}<<".format(m + 1))

            if mode == "GD":
                x_hat = self.GD( x_0=x_hat, m=m, pro_info=K_ext, max_T=T, ep=ep, prt=True, writer=writer )

            elif mode == "SGD":
                K_ext["center"] = x_hat.copy()
                K_ext["radius"] = self.update_radius_sgd( delta )
                ep = self.update_radius_sgd( delta )
                x_hat = self.SGD( x_0=x_hat, m=m, pro_info=K_ext, max_T=T, delta=delta, ave_T=10, ep=ep*1e3, prt=True, writer=writer )

            elif mode == "MM":
                x_hat = self.MM( x_0=x_hat, m=m, pro_info=K_ext, max_T=T, ep=ep, prt=True, writer=writer )

            elif mode == "SMM":
                K_ext["center"] = x_hat.copy()
                K_ext["radius"] = self.update_radius_sgd( delta )
                ep = self.update_radius_smm( delta )
                x_hat = self.SMM( x_0=x_hat, m=m, pro_info=K_ext, max_T=T, delta=delta, ave_T=ave_T, ep=ep, prt=True, writer=writer )

            val = self.get_val( x_hat ) # 現在の損失を計算
            if prt:
                if val != INF:
                    print( "loss: {}\n".format(val) )
                else:
                    print( "loss: INF\n" )

            if m >= 1:
                if self.valid_err( loss_f=self.get_val, x0=x_hat0, x1=x_hat, err=1e-6 ): # 誤差が小さくなったら終了
                    print("converge")
                    break

            x_hat0 = x_hat.copy()
            delta /= 2

        if prt and val != INF:
            print("result: {}\n".format(val))

        if type(file) != type(None):
            file.close()

        return x_hat, val

    def GD( self, x_0, m, pro_info=None, max_T=10, ep=1e-5, prt=False, writer=None ):
        '''
        parm array x_0: 初期推定ベクトル
        parm list pro_info: 射影先の範囲を指定(0:array center, 1:float rad) [if None: 射影しない]
        parm int max_T: 最大ラウンド数
        return array x_hat: max_Tラウンド後の推定値
        need MLUtil.project: 射影する関数(in:array x_original, list pro_info; out:array x_new)
        OR func get_grad: 目的関数の勾配を計算する関数(in:array x; out:array grad)
        OR func get_eta: 学習率を計算する関数(in:int t; out:float eta)
        '''
        x_t = x_0.copy()
        x_t0 = x_0.copy()

        for t in range( 1, max_T + 1 ):
            x_t -= self.get_eta_gd( m ) * self.get_grad( x_t )
            if type(pro_info) != type(None): #  射影
                x_t = self.MU.project( x_t, pro_info )

            if prt: self.round_prt( t, x_t )
            if type(writer) != type(None): self.round_wrt( m, t, x_t, writer=writer )

            if t >= 1:
                if self.valid_err( loss_f=self.get_val, x0=x_t0, x1=x_t, err=ep ): # 誤差が小さくなったら終了
                    break
            x_t0 = x_t.copy()

        return x_t

    def SGD( self, x_0, m, pro_info=None, max_T=10, delta=1.0, ave_T=10, ep=1e-5, prt=False, writer=None ):
        '''
        parm array x_0: 初期推定ベクトル
        parm list pro_info: 射影先の範囲を指定(0:array center, 1:float rad) [if None: 射影しない]
        parm int max_T: 最大ラウンド数 [if None: 10]
        parm float delta: smoothingパラメータ [if None: 1.0]
        parm float seed: smoothing時に用いる乱数のシード値 [if None: time]
        parm int ave_T: smoothing時の乱数の平均を取る回数 [if None: 10]
        return array x_hat: max_Tラウンド後の推定値
        need MLUtil.project: 射影する関数(in:array x_original, array center, float rad; out:array x_new)
        OR func get_grad: 目的関数の勾配を計算する関数(in:array x; out:array grad)
        OR func get_eta: 学習率を計算する関数(in:int t; out:float eta)
        '''
        x_t = x_0.copy()
        x_t_l = []
        suf_T = max_T // 2 # 後半のsuf_T回分のx_tの平均をとる
        for t in range( 1, max_T + 1 ):
            x_t -= self.get_eta_sgd( t ) * self.smooth_grad( x_t, delta=delta, ave_T=ave_T )
            if type(pro_info) != type(None): # 射影
                x_t = self.MU.project( x_t, pro_info )

            if t > max_T - suf_T:
                x_t_l += [ x_t ]

            if prt: self.round_prt( t, x_t )
            if type(writer) != type(None): self.round_wrt( m, t, x_t, writer=writer )

        return sum( x_t_l ) / suf_T

    def MM( self, x_0, m, pro_info=None, max_T=10, ep=1e-5, prt=False, writer=None ):
        '''
        parm array x_0: 初期推定ベクトル
        parm list pro_info: 射影先の範囲を指定(0:array center, 1:float rad) [if None: 射影しない]
        parm int max_T: 最大ラウンド数 [if None: 10]
        return array x_hat: max_Tラウンド後の推定値
        need MLUtil.project: 射影する関数(in:array x_original, array center, float rad; out:array x_new)
        OR func srg_min: surrogate関数の最小解を計算する関数(in:array x_original, list info; out:array argmin_x)
        '''
        x_t = x_0.copy()
        x_t0 = x_0.copy()

        for t in range( max_T ):
            x_t = self.srg_min( x_org=x_t )
            if type(pro_info) != type(None): # 射影
                x_t = self.MU.project( x_t, pro_info )

            if prt: self.round_prt( t, x_t )
            if type(writer) != type(None): self.round_wrt( m, t, x_t, writer=writer )

            if t >= 1:
                if self.valid_err( loss_f=self.get_val, x0=x_t0, x1=x_t, err=ep ): # 誤差が小さくなったら終了
                    break
            x_t0 = x_t.copy()

        return x_t

    def SMM( self, x_0, m, pro_info=None, max_T=10, delta=1.0, ave_T=10, ep=1e-5, prt=False, writer=None ):
        '''
        parm array x_0: 初期推定ベクトル
        parm list pro_info: 射影先の範囲を指定(0:array center, 1:float rad) [if None: 射影しない]
        parm int max_T: 最大ラウンド数
        return array x_hat: max_Tラウンド後の推定値
        need MLUtil.project: 射影する関数(in:array x_original, array center, float rad; out:array x_new)
        OR func srg_min: surrogate関数（の不偏推定）の最小解を計算する関数(in:array x_original, list info, int seed; out:array argmin_x)
        '''
        x_t = x_0.copy()
        x_t0 = x_0.copy()
        rho = 0.1
        info = { "t": 0, "rho": rho }

        for t in range( max_T ):
            info[ "t" ] += 1
            x_t = self.smooth_srg_min( x_org=x_t, info=info, delta=delta, ave_T=ave_T )
            if type(pro_info) != type(None):
                x_t = self.MU.project( x_t, pro_info )

            if prt: self.round_prt( t, x_t )
            if type(writer) != type(None): self.round_wrt( m, t, x_t, writer=writer )

            if t >= 1:
                if self.valid_err( loss_f=self.get_val, x0=x_t0, x1=x_t, err=ep ): # 誤差が小さくなったら終了
                    break
            x_t0 = x_t.copy()

        return x_t

    def smooth_val( self, x, delta=1.0, ave_T=10 ):
        '''
        parm array x: smoothing後の値を返したい座標
        parm float delta: smoothingの荒さを決めるパラメータ
        parm int seed: smoothingの際に用いる乱数のシード値
        parm int ave_T: smoothingの際，ave_T回不偏推定を求め，その平均をsmoothing後の値とする
        return float smoothed_val: smoothing後の値
        '''
        if delta <= 0:
            return self.get_val( x )

        zero_vec = np.zeros_like( x, np.float )

        val = 0.0
        for t in range(ave_T):
            u = self.MU.make_a_data_ball( dim=self.x_dim, center=zero_vec, rad=delta )
            val += self.get_val( x + u )
        return val / ave_T

    def smooth_grad( self, x, delta=1.0, ave_T=10 ):
        '''
        parm array x: smoothing後の勾配を返したい座標
        parm float delta: smoothingの荒さを決めるパラメータ
        parm int seed: smoothingの際に用いる乱数のシード値
        parm int ave_T: smoothingの際，ave_T回不偏推定を求め，その平均をsmoothing後の勾配とする
        return float smoothed_val: smoothing後のfの勾配
        '''
        if delta <= 0:
            return self.get_grad( x )

        zero_vec = np.zeros_like( x, np.float )

        grad = np.zeros_like( x, np.float )
        for t in range(ave_T):
            u = self.MU.make_a_data_ball( dim=self.x_dim, center=zero_vec, rad=delta )
            grad += self.get_grad( x + u )
        return grad / ave_T

    def smooth_generic( self, get_smoothed, x, delta=1.0, ave_T=10 ):
        '''
        parm func get_smoothed: 目的関数の何かを計算する関数(in:array x; out:anything smoothed)[ex: get_val, get_grad]
        parm array x: smoothing後の勾配を返したい座標
        parm float delta: smoothingの荒さを決めるパラメータ
        parm int ave_T: smoothingの際，ave_T回不偏推定を求め，その平均をsmoothing後の勾配とする
        return float smoothed_val: smoothing後のfの勾配
        '''
        if delta <= 0:
            return self.get_smoothed( x )

        zero_vec = np.zeros_like( x, np.float )

        smoothed = np.zeros_like( get_smoothed( x ), np.float )
        for t in range(ave_T):
            u = self.MU.make_a_data_ball( dim=self.x_dim, center=zero_vec, rad=delta )
            smoothed += self.get_smoothed( x + u )
        return smoothed / ave_T

    def round_prt( self, t, x_t ):
        '''
        parm int t: 現在のラウンド
        parm array x_t: 現在の解
        '''
        val = self.get_val( x_t )
        if val != INF:
            print( "round {}: {}".format(t, val ) )
        else:
            print( "round {}: INF".format(t) )

    def round_wrt( self, m, t, x_t, writer=None ):
        '''
        parm int m: 現在のエポック
        parm int t: 現在のラウンド
        parm array x_t: 現在の解
        '''
        val = self.get_val( x_t )
        if val != INF:
            writer.writerow( [m, t, val] )
        else:
            writer.writerow( [m, t, "INF"] )


    ### Need to Override ###
    def srg_min( self, x_org, info=None ):
        '''
        parm array x_org: 現在の予測点
        parm list info: surrogate関数を作成するために必要なパラメータ等(eg. L-lipschitz)
        parm int seed: 乱数を用いてsurrogate関数を作成するときのシード値 [if 0: 乱数を用いない]
        return array argmin_x: 作成したsurrogate関数の最小解
        '''
        print("Override the function. \"srg_min\".")
        return np.zeros_like( x_org )

    def smooth_srg_min( self, x_org, info=None, delta=1.0, ave_T=10 ):
        '''
        parm array x_org: 現在の予測点
        parm list info: surrogate関数を作成するために必要なパラメータ等(eg. L-lipschitz)
        parm float delta: smoothingの荒さを決めるパラメータ
        parm int ave_T: smoothingの際，ave_T回不偏推定を求め，その平均をsmoothing後の勾配とする
        return array argmin_x: 作成したsurrogate関数の最小解
        '''
        print("Override the function. \"stc_srg_min\".")
        return np.zeros_like( x_org )

    def get_val( self, x ):
        '''
        parm array x: 値を求めたい座標
        return array val: 関数の値
        '''
        print("Override the function \"get_val\".")
        return 0.0

    def valid_err( self, loss_f, x0, x1, err ):
        '''
        parm func loss_f: 誤差関数 (in: array x0; out: float val)
        parm array x0: 前回の解
        parm array x1: 今回の解
        parm float err: 目標誤差
        return bool: 前回と今回の違いが目標誤差に収まっているか否か
        '''
        print("Override the function \"valid_err\".")
        return True

    def get_w( self, t ):
        '''
        parm int t: 重みwを求めたい時刻
        return float eta: 時刻tにおけるw
        '''
        print("Override the function \"get_w\".")
        return 1 / t

    def get_grad( self, x ):
        '''
        parm array x: 勾配を求めたい座標
        return array grad: 関数の勾配
        '''
        print("Override the function \"get_grad\".")
        return np.zeros_like( x )

    def get_eta_gd( self, t ):
        '''
        parm int t: learning rateを求めたい時刻
        return float eta: 時刻tにおけるlearning rate
        '''
        print("Override the function \"get_eta_gd\".")
        return 1 / t

    def get_eta_sgd( self, t ):
        '''
        parm int t: learning rateを求めたい時刻
        return float eta: 時刻tにおけるlearning rate
        '''
        print("Override the function \"get_eta_sgd\".")
        return 1 / t

    def update_radius_sgd( self, delta ):
        '''
        parm float delta: 射影先の領域の半径を求めたいdelta
        return float radius: 次の射影先の領域の半径
        '''
        print("Override the function \"update_radius_sgd\".")
        return 1.5 * delta

    def update_radius_smm( self, delta ):
        '''
        parm float delta: 射影先の領域の半径を求めたいdelta
        return float radius: 次の射影先の領域の半径
        '''
        print("Override the function \"update_radius_smm\".")
        return delta

    def update_ep_sgd( self, delta ):
        '''
        parm float delta: エポック内誤差を求めたいdelta
        return float epsilon: 次の射ポック内誤差
        '''
        print("Override the function \"update_radius_sgd\".")
        return (delta **2) / 32

    def update_ep_smm( self, delta ):
        '''
        parm float delta: ポック内誤差を求めたいdelta
        return float epsilon: 次のポック内誤差
        '''
        print("Override the function \"update_radius_smm\".")
        return delta * 2


class MLUtil:

    def __init__( self ):
        pass

    def project( self, x_org, pro_info ):
        '''
        parm array x_org: 射影前の座標
        parm list pro_info: 射影先の範囲を指定(0:array center, 1:float rad)
        return array x_new: 射影後の座標
        '''
        center = np.array( pro_info["center"].copy() )
        rad = pro_info["radius"]

        norm = MLUtil.norm2( [center, x_org] )
        if norm <= rad:
            return x_org
        return  ( x_org - center ) * rad / norm + center

    @staticmethod
    def norm2( vec_list ):
        '''
        parm list vec_list: 要素数1，または2のlistで，各要素はarray型のベクトル
        return float norm: ベクトルのノルムを返す [if 要素数1: ベクトルの2ノルム],[if 要素数2：二つのベクトルの差の2ノルム],[else: 0]
        '''
        dim = len( vec_list )
        if dim == 1:
            return np.linalg.norm( np.array(vec_list[0]) )
        elif dim == 2:
            return np.linalg.norm( np.array(vec_list[0]) - np.array(vec_list[1]) )
        else:
            return 0

    @staticmethod
    def valid_err( loss_f, x0, x1, err ):
        '''
        parm func loss_f: 誤差関数 (in: array x0; out: float val)
        parm array x0: 前回の解
        parm array x1: 今回の解
        parm float err: 目標誤差
        return bool: 前回と今回の違いが目標誤差に収まっているか否か
        '''
        if ( abs( loss_f( x0 ) - loss_f( x1 ) ) / loss_f ( x0 ) ) < err:
            return True
        return False

    def numerical_gradient( self, f, x ):
        '''
        parm func f: 勾配を求めたい関数(in:array x; out:float val)
        parm array x: 勾配を求めたい座標
        return array grad: fの勾配
        '''
        # 勾配を入れるベクトルをゼロで初期化する
        grad = np.zeros_like(x)

        for i in range( len(x) ):
            # i 番目の変数で偏微分する
            grad[i] = MLUtil.numerical_diff(f, x, i)

        # 計算した勾配を返す
        return grad

    @staticmethod
    def numerical_diff( f, x, i ):
        '''中央差分を元に数値微分する関数 (偏微分)
        :param func f: 偏微分する関数
        :param array x: 偏微分する引数
        :param int i: 偏微分する変数のインデックス
        '''
        # 丸め誤差で無視されない程度に小さな値を用意する
        h = 1
        # 偏微分する変数のインデックスにだけ上記の値を入れる
        h_vec = np.zeros_like(x)
        h_vec[i] = h
        # 数値微分を使って偏微分する
        return (f(x + h_vec) - f(x - h_vec)) / (2 * h)

    @staticmethod
    def make_a_data_ball( dim, center, rad=1.0 ):
        '''
        parm int dim: 生成するデータの次元数
        parm array center: 生成するデータの各次元の中心座標 [if None: zero]
        parm float rad: 生成するデータの球の半径 [if None: 1.0]
        return :生成した一つのデータ
        '''
        x = np.random.randn( dim )
        r = np.linalg.norm( x )
        if r != 0.:
            x /= r
        x *= rad
        reg_r = np.power( np.random.random(), 1. / dim )
        if type(center) == type(None):
            return x * reg_r
        return x * reg_r + center

    def make_data_ball( self, dim, num, center=None, rad=1.0 ):
        '''
        parm int dim: 生成するデータの次元数
        parm int num: 生成するデータの個数
        parm array center: 生成するデータの各次元の中心座標 [if None: None]
        parm float rad: 生成するデータの球の半径 [if None: 1.0]
        return :生成したnum個のデータ
        '''
        data_l = [ MLUtil.make_a_data_ball( dim, center, rad ) for i in range(num) ]
        return np.array( data_l )

