import random
from sampling.load_configuration import load_conf
from optimizer import Dimension
import pickle
from optimizer import Optimizer
from optimizer import RacosOptimization
import numpy as np
from evaluater import Evaluater
from multiprocessing import Process,Pool
import multiprocessing
from base import NetworkUnit
import time
import pickle
import copy

class Sampler_tree:
    def __init__(self, nn):
        # 设置结构大小
        self.node_number = len(nn.graph_part)
        # 读取配置表得到操作的概率映射
        self.setting, self.pros, self.parameters_subscript, = load_conf()
        self.dic_index = self._init_dict()
        #
        # print(self.dic_index)
        # print(len(self.pros))
        self.p = []
        # 设置优化Dimension
        self.dim = Dimension()
        self.dim.set_dimension_size(len(self.pros))
        # 设置优化的参数

        self.dim.set_regions(self.pros, [0 for _ in range(len(self.pros))])


    # 更新p
    def renewp(self, newp):
        self.p = newp

    def sample(self, node_number):
        res = []
        # 基于节点的搜索结构参数
        for num in range(node_number):
            # 取概率
            p_node = self.p
            first = self.range_sample(p_node, self.dic_index['conv'])
            tmp = ()
            # 首位置确定 conv 还是 pooling
            if first == 0:
                # 搜索conv下的操作
                # 取概率所在的区间，进行采样
                tmp = tmp + ('conv',)
                struct_conv = ['conv filter_size', 'conv kernel_size', 'conv activation']
                for key in struct_conv:
                    tmp = tmp + (self.setting['conv'][key.split(' ')[-1]]['val'][
                                     self.range_sample(p_node,self.dic_index[key])],)
            else:
                # 搜索pooling下的操作
                # 取概率所在的区间，进行采样
                tmp = tmp + ('pooling',)
                struct_pooling = ['pooling pooling_type', 'pooling kernel_size']
                for key in struct_pooling:
                    tmp = tmp + (self.setting['pooling'][key.split(' ')[-1]]['val'][
                        self.range_sample(p_node, self.dic_index[key])],)
            res.append(tmp)
        return res

    # 基于opt.sample()所得结果，基于均匀，进行采样
    def range_sample(self, p_node, range_index):
        k = p_node[range_index[0]:range_index[1]]
        k.append(1-np.array(k).sum())
        # print(k)
        # print(k)
        r = random.random()
        # print(r)
        for j, i in enumerate(k):
            if r <= i:
                return j
            r = r - i

    # 基于读取配置表信息，得到操作的概率区间映射
    def _init_dict(self):
        dic = {}
        dic['conv'] = (0,1,)
        cnt = 1
        for key in self.setting:
            for k in self.setting[key]:
                tmp = len(self.setting[key][k]['val']) - 1
                dic[key + ' ' + k ] = (cnt,cnt + tmp,)
                cnt += tmp

        return dic

    def get_dim(self):
        return self.dim

    def get_parametets_subscript(self):
        return self.parameters_subscript

    # log
    def get_cell_log(self, POOL, PATH, date):
        for i, j in enumerate(POOL):
            s = 'nn_param_' + str(i) + '_' + str(date)
            fp = open(PATH + s, "wb")
            # print(s)
            pickle.dump(j.cell_list, fp)


class Experiment_tree:
    def __init__(self, nn, sample_size=5, budget=20, positive_num=2, r_p=0.99, uncertain_bit=3, add_num=20000):
        self.nn = nn
        self.spl = Sampler_tree(nn)
        dim = self.spl.get_dim()
        self.opt = Optimizer(self.spl.get_dim(), self.spl.get_parametets_subscript())
        # sample_size = 5  # the instance number of sampling in an iteration
        # budget = 20  # budget in online style
        # positive_num = 2  # the set size of PosPop
        # r_p = 0.99  # the probability of sample in model
        # uncertain_bit = 3  # the dimension size that is sampled randomly
        # set hyper-parameter for optimization, budget is useless for single step optimization
        self.opt.set_parameters(ss=sample_size, bud=budget, pn=positive_num, rp=r_p, ub=uncertain_bit)
        # clear optimization model
        self.opt.clear()
        self.budget = budget
        pros = self.opt.sample()
        self.spl.renewp(pros)
        self.eva = Evaluater()
        self.eva.add_data(add_num)
        self.opt_p_log = []
        print(self.eva.max_steps)

        for i in range(self.budget):
            self.opt_p_log.append(pros)
            spl_list = self.spl.sample(len(nn.graph_part))
            self.nn.cell_list.append(spl_list)
            # time_tmp = time.time()
            # score = self.eva.evaluate(self.nn, i, time_tmp)
            score = self.eva.evaluate(self.nn)
            # Updating optimization based on the obtained scores
            # Upadting pros in spl
            self.opt.update_model(pros, -score)
            pros = self.opt.sample()
            self.spl.renewp(pros)
            #
        self.res_fea = self.opt.get_optimal().get_features()
        self.res_fit = self.opt.get_optimal().get_fitness()
        print('best:')
        print('features', self.res_fea)  # pros
        print('fitness', self.res_fit)  # scores


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    S = NetworkUnit()
    S.graph_part = [[1, 10], [2, 14], [3], [4], [5], [6], [7], [8], [9], [], [11], [12], [13], [6], [7]]
    #
    time1 = time.time()
    ob = Experiment_tree(S, sample_size=5, budget=100, positive_num=2, uncertain_bit=3, add_num=10000)
    time2 = time.time()
    print('time cost: ', time2 - time1)

    res = [S.graph_part, ob.res_fea, ob.res_fit]
    op = open('enumerate_best_pros.pickle', 'wb')
    pickle.dump(res, op)
    op.close()
    # print(S.cell_list)
    # spl.renewp(tmp)
    # res = spl.sample()
    # for i in res:
    #     print(i)
    # [('conv', 64, 5, 'leakyrelu'), ('conv', 48, 3, 'relu'),
    # ('conv', 48, 5, 'leakyrelu'), ('conv', 64, 1, 'leakyrelu'),
    # ('pooling', 'global', 3), ('pooling', 'avg', 2),
    # ('conv', 64, 1, 'relu'), ('pooling', 'avg', 3),
    # ('conv', 64, 1, 'leakyrelu'), ('conv', 64, 5, 'leakyrelu')]
