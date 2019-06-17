import random
from sampling.load_configuration import load_conf
from optimizer import Dimension
import pickle
from optimizer import Optimizer
from optimizer import RacosOptimization
import numpy as np

class Sampler_tree:
    def __init__(self, node_number):
        self.node_number = node_number
        self.setting, self.pros, self.parameters_subscript_node, = load_conf()
        self.dic_index = self._init_dict()
        #
        print(self.dic_index)

        # print(len(self.pros))

        self.p = []

        # parameters
        self.__region, self.__type = self.opt_parameters()
        self.dim = Dimension()
        self.dim.set_dimension_size(len(self.__region))
        self.dim.set_regions(self.__region, self.__type)

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

    # 更新p
    def renewp(self, newp):
        self.p = newp


    def sample(self):
        res = []
        for num in range(self.node_number):
            p_node = self.p[num*len(self.dic_index):(num+1)*len(self.dic_index)]
            first = p_node[0]
            tmp = ()
            if first == 0:
                tmp = tmp + ('conv',)
                struct_conv = ['conv filter_size', 'conv kernel_size', 'conv activation']
                for key in struct_conv:
                    tmp = tmp + (self.setting['conv'][key.split(' ')[-1]]['val'][p_node[self.dic_index[key][-1]]],)

            else:
                tmp = tmp + ('pooling',)
                struct_pooling = ['pooling pooling_type', 'pooling kernel_size']
                for key in struct_pooling:
                    tmp = tmp + (self.setting['pooling'][key.split(' ')[-1]]['val'][p_node[self.dic_index[key][-1]]],)
            res.append(tmp)
        return res


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

    def get_cell_log(self, POOL, PATH, date):
        for i, j in enumerate(POOL):
            s = 'nn_param_' + str(i) + '_' + str(date)
            fp = open(PATH + s, "wb")
            # print(s)
            pickle.dump(j.cell_list, fp)

if __name__ == '__main__':
    a = [[1, 10, 14], [2], [3], [4], [5], [6], [7], [8], [9], [], [11], [12], [13], [7], [15], [16], [17], [18], [19],
         [20], [21], [9]]
    b = [[1, 10], [2, 3], [3], [4], [5], [6], [7], [8], [9], [], [11], [12], [13], [7]]
    c = [[1], [2, 10], [3], [4], [5], [6, 13], [7], [8], [9], [], [11], [12], [6], [14], [15], [9]]
    d = [[1], [2, 10], [3, 7], [4], [5], [6], [7], [8], [9], [], [11], [12], [13], [6]]
    s = [a, b, c, d]

    spl = Sampler_tree(10)
    opt = RacosOptimization(spl.dim)
    #
    # #
    # opt = Optimizer(spl.dim, spl.parameters_subscript)
    sample_size = 3  # the instance number of sampling in an iteration
    budget = 20000  # budget in online style
    positive_num = 2  # the set size of PosPop
    rand_probability = 0.99  # the probability of sample in model
    uncertain_bit = 3  # the dimension size that is sampled randomly
    # set hyper-parameter for optimization, budget is useless for single step optimization
    opt.set_parameters(ss=sample_size, bud=budget, pn=positive_num, rp=rand_probability, ub=uncertain_bit)
    # clear optimization model
    opt.clear()

    #

    pros = opt.sample()
    print(pros)
    spl.renewp(pros)
    res = spl.sample()
    for i in res:
        print(i)

    #

    pros = opt.sample()
    print(pros)
    spl.renewp(pros)
    res = spl.sample()
    for i in res:
        print(i)

    # [('conv', 64, 5, 'leakyrelu'), ('conv', 48, 3, 'relu'),
    # ('conv', 48, 5, 'leakyrelu'), ('conv', 64, 1, 'leakyrelu'),
    # ('pooling', 'global', 3), ('pooling', 'avg', 2),
    # ('conv', 64, 1, 'relu'), ('pooling', 'avg', 3),
    # ('conv', 64, 1, 'leakyrelu'), ('conv', 64, 5, 'leakyrelu')]
