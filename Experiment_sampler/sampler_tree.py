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
        # print(self.dic_index)
        # print(len(self.pros))
        self.p = []

        self.dim = Dimension()
        self.dim.set_dimension_size(node_number * len(self.pros))
        # parameters
        tmp = []
        self.parameters_subscript = []
        for i in range(self.node_number):
            tmp = tmp + self.pros
            self.parameters_subscript.extend(self.parameters_subscript_node)

        self.dim.set_regions(tmp, [0 for _ in range(len(tmp))])


    # 更新p
    def renewp(self, newp):
        self.p = newp


    def sample(self):
        res = []
        for num in range(self.node_number):
            p_node = self.p[num*len(self.pros):(num+1)*len(self.pros)]
            first = self.range_sample(p_node,self.dic_index['conv'])
            tmp = ()
            if first == 0:
                tmp = tmp + ('conv',)
                struct_conv = ['conv filter_size', 'conv kernel_size', 'conv activation']
                for key in struct_conv:
                    tmp = tmp + (self.setting['conv'][key.split(' ')[-1]]['val'][
                                     self.range_sample(p_node,self.dic_index[key])],)
            else:
                tmp = tmp + ('pooling',)
                struct_pooling = ['pooling pooling_type', 'pooling kernel_size']
                for key in struct_pooling:
                    tmp = tmp + (self.setting['pooling'][key.split(' ')[-1]]['val'][
                        self.range_sample(p_node, self.dic_index[key])],)
            res.append(tmp)
        return res


    def range_sample(self,p_node,range_index):
        k = p_node[range_index[0]:range_index[1]]
        k.append(1-np.array(k).sum())
        # print(k)
        print(k)
        r = random.random()
        print(r)
        for j, i in enumerate(k):
            if r <= i:
                return j
            r = r - i


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

    # opt = RacosOptimization(spl.dim)
    opt = Optimizer(spl.dim, spl.parameters_subscript)

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
    spl.renewp(pros)
    res = spl.sample()
    print(res)

    #

    newp = [0.8, 0.2, 0.3, 0.6, 0.2, 0.4, 0.3, 0.6, 0.2,
            0.3, 0.4, 0.3, 0.3, 0.7, 0.2, 0.4, 0.3, 0.5,
            0.2, 0.3, 0.4,0.3, 0.3, ]  # 优化更新
    tmp = []
    for i in range(10):
        tmp = tmp + newp

    spl.renewp(tmp)
    res = spl.sample()
    for i in res:
        print(i)
    # [('conv', 64, 5, 'leakyrelu'), ('conv', 48, 3, 'relu'),
    # ('conv', 48, 5, 'leakyrelu'), ('conv', 64, 1, 'leakyrelu'),
    # ('pooling', 'global', 3), ('pooling', 'avg', 2),
    # ('conv', 64, 1, 'relu'), ('pooling', 'avg', 3),
    # ('conv', 64, 1, 'leakyrelu'), ('conv', 64, 5, 'leakyrelu')]
