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


class Sampler_struct:

    def __init__(self, nn):
        '''
        :param nn: NetworkUnit

        '''
        # 设置结构大小
        self.node_number = len(nn.graph_part)
        # 读取配置表得到操作的对应映射
        self.setting, self.pros, self.parameters_subscript_node, = load_conf()
        self.dic_index = self._init_dict()

        # print(len(self.pros))
        self.p = []

        # 设置优化Dimension
        # 设置优化的参数
        self.__region, self.__type = self.opt_parameters()
        self.dim = Dimension()
        self.dim.set_dimension_size(len(self.__region))
        self.dim.set_regions(self.__region, self.__type)
        self.parameters_subscript = []  #

    # 更新p
    def renewp(self, newp):
        self.p = newp

    # 基于操作的对应映射得到优化参数
    def opt_parameters(self):
        __type_tmp = []
        __region_tmp = []
        for key in self.dic_index:
            tmp = int(self.dic_index[key][1] - self.dic_index[key][0])
            __region_tmp.append([0, tmp])
            __type_tmp.append(2)

        __region = []
        __type = []
        for i in range(self.node_number):
            __region = __region + __region_tmp
            __type.extend(__type_tmp)
        return __region, __type

    def sample(self):
        res = []
        # 基于节点的搜索结构参数
        for num in range(self.node_number):
            # 取一个节点大小的概率
            p_node = self.p[num*len(self.dic_index):(num+1)*len(self.dic_index)]
            print(p_node)
            first = p_node[0]
            tmp = ()
            # 首位置确定 conv 还是 pooling
            if first == 0:
                # 搜索conv下的操作
                # 基于操作的对应映射取配置所在的地址，进行取值
                tmp = tmp + ('conv',)
                struct_conv = ['conv filter_size', 'conv kernel_size', 'conv activation']
                for key in struct_conv:
                    tmp = tmp + (self.setting['conv'][key.split(' ')[-1]]['val'][p_node[self.dic_index[key][-1]]],)
            else:
                # 搜索pooling下的操作
                # 基于操作的对应映射取配置所在的地址，进行取值
                tmp = tmp + ('pooling',)
                struct_pooling = ['pooling pooling_type', 'pooling kernel_size']
                for key in struct_pooling:
                    tmp = tmp + (self.setting['pooling'][key.split(' ')[-1]]['val'][p_node[self.dic_index[key][-1]]],)
            res.append(tmp)
        return res

    # 基于opt.sample()所得结果，基于位置得到操作
    def _init_dict(self):
        dic = {}
        dic['conv'] = (0,1,0)
        cnt = 1
        num = 1
        for key in self.setting:
            for k in self.setting[key]:
                tmp = len(self.setting[key][k]['val']) - 1
                dic[key + ' ' + k ] = (cnt, cnt + tmp, num)
                num += 1
                cnt += tmp

        return dic
    # log
    def get_cell_log(self, POOL, PATH, date):
        for i, j in enumerate(POOL):
            s = 'nn_param_' + str(i) + '_' + str(date)
            fp = open(PATH + s, "wb")
            # print(s)
            pickle.dump(j.cell_list, fp)


class Experiment_struct:
    def __init__(self, nn, sample_size=5, budget=20, positive_num=2, r_p=0.99, uncertain_bit=3, add_num=20000):
        self.nn = nn
        self.spl = Sampler_struct(nn)
        self.opt = Optimizer(self.spl.dim, self.spl.parameters_subscript)
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

        print(self.eva.max_steps)

    def start_experiment(self):
        for i in range(self.budget):
            spl_list = self.spl.sample()
            self.nn.cell_list.append(spl_list)
            # time_tmp = time.time()
            score = self.eva.evaluate(self.nn)
            print('##################' * 10)
            print(score)
            # Updating optimization based on the obtained scores
            # Upadting pros in spl
            self.opt.update_model(self.spl.pros, -score)
            pros = self.opt.sample()
            self.spl.renewp(pros)

        # a = self.opt.get_optimal().get_features()  # pros
        # b = self.opt.get_optimal().get_fitness()  # scores
        # print(a)
        # print(b)


if __name__ == '__main__':
    S = NetworkUnit()
    S.graph_part = [[1, 10], [2, 14], [3], [4], [5], [6], [7], [8], [9], [], [11], [12], [13], [6], [7]]
    #
    time1 = time.time()
    ob = Experiment_struct(S, sample_size=5, budget=100, positive_num=2, uncertain_bit=3, add_num=10000)
    # ob.start_experiment()
    time2 = time.time()
    print('time cost: ', time2 - time1)
    # op = open('enumerater_best_struct_samper.pickle','wb')
    # pickle.dump(S.cell_list, op)
    # op.close()
    # # print(S.cell_list)


