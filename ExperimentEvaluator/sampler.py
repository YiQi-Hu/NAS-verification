# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 14:03:47 2019

@author: Chlori
"""
import random
from .sampling.load_configuration import load_conf
from .optimizer import Dimension
import pickle


class Sampler():
    def __init__(self):
        '''
        p：优化得到概率向量,
        parameters_count:p中概率所属参数类型的个数，分别为filter_count,convkernel_count,activation_count,poolingtype_count,poolingkernel_count
        parameters_subscript:二维list,p中conv_or_pooling,filter,convkernel,activation,poolingtype,poolingkernel的下标
        '''
        self.setting, self.pros, self.const, = load_conf()
        self.p = []
        #self.parameters_count = self._get_parameters_count(self.setting)
        self.parameters_subscript = self.const
#        print('parameters_count',self.parameters_count)
#        print('parameters_subscript',self.parameters_subscript)
        self.order = self._get_ord_sub()
        self.dim = Dimension()
        self.dim.set_dimension_size(len(self.pros))

        self.dim.set_regions(self.pros, [0 for _ in range(len(self.pros))])
        pass
        
        
    #更新p
    def renewp(self,newp):
        self.p = newp

    #get parameters' name ordered by setting
    def _get_unord(self):
        unord_1 = []
        unord_2 = []
        for key in self.setting.keys():
            unord_1.append(key)
        for i in self.setting.keys():
            obj = self.setting[i]
            temp_keys = []
            for key in obj.keys():
                temp_keys.append((key))
            unord_2.append(temp_keys)
        return unord_1, unord_2

    #find each parameter's index in unord list
    def _sort(self, ord, unord):
        ord_sub = []
        for i in range(len(ord)):
            for j in range(len(unord)):
                if ord[i] == unord[j]:
                    ord_sub.append(j)
                    break
            continue
        return ord_sub

    #ord is the order we need
    #the reture ord_sub is 'filter_size, kernel_size, activation, pooling_type, kernel_size' 's index in setting's seqence
    def _get_ord_sub(self):
        ord_0 = ['conv', 'pooling']
        ord_1 = [['filter_size', 'kernel_size', 'activation'], ['pooling_type', 'kernel_size']]
        unord_0, unord_1 = self._get_unord()
        first_sub = self._sort(ord_0, unord_0)
        ord_sub = []
        for i in range(len(ord_0)):
            start_sub = 1
            for j in range(first_sub[i]):
                start_sub += len(unord_1[j])
            temp_sub = self._sort(ord_1[i], unord_1[first_sub[i]])
            if ord_sub == []:
                ord_sub = [(start_sub + _) for _ in temp_sub]
            else:
                ord_sub += [(start_sub + _) for _ in temp_sub]
        return ord_sub

    '''
    def _get_parameters_count(self, setting):
        filter_count, convkernel_count, activation_count = len(setting['conv']['filter_size']['val']), len(setting['conv']['kernel_size']['val']), len(
            setting['conv']['activation']['val'])
        poolingtype_count, poolingkernel_count = len(setting['pooling']['pooling_type']['val']), len(setting['pooling']['kernel_size']['val'])
        return filter_count-1, convkernel_count-1, activation_count-1, poolingtype_count-1, poolingkernel_count-1
    
    #创建一个二维数组，为p中概率所属的参数类型分类,根据parameters_count生成
    #sample output:[[0][1,2,3][4,5][6,7][8,9][10]]
    def _create_parameters_subscript(self):
        parameters_subscript = []
        parameters_subscript.append([0])
        j = 1
        for i in range(len(self.parameters_count)):
            temp_parameters_subscript = []
            for k in range(self.parameters_count[i]):
                temp_parameters_subscript.append(j)
                j += 1
            parameters_subscript.append(temp_parameters_subscript)
        return parameters_subscript
    '''

    #第二步采样
    '''
    	sample series number of one paramter
    	output probability is cumulative probability
    	example
    	input:[0.7,0.2]
    	output probablity:[0.7,0.9]
    	ran_number: 0.50
    	output: 0
    '''
    def _sample_step_2(self,probability):
        for i in range(len(probability) - 1):
            probability[i + 1] += probability[i]
        #print('cumulative probability',probability)
        ran_num = random.random()
        for j in range(len(probability)):
            if ran_num < probability[j]:
                series_num = j
                break
        if ran_num >= probability[len(probability)-1]:
            series_num = len(probability)
        return series_num

    #采样
    '''
		step 1:sample conv or pooling
		step 2:sample filter,convkernel,activation or poolingtype,poolingkernel
		output:list of paramters series numbers 
    '''
    def _sample_step_1(self):
        series_num = []
        ord = self.order
        # print('ord', ord)
        ran = random.random()
        if ran < self.p[0]:
            series_num.append(0)
            # print(self.parameters_subscript)
            # print(self.parameters_subscript[ord[1]][0],self.parameters_subscript[ord[1]][-1]+1)
            series_num.append(self._sample_step_2(self.p[self.parameters_subscript[ord[0]][0]:self.parameters_subscript[ord[0]][-1]+1]))
            series_num.append(self._sample_step_2(self.p[self.parameters_subscript[ord[1]][0]:self.parameters_subscript[ord[1]][-1]+1]))
            series_num.append(self._sample_step_2(self.p[self.parameters_subscript[ord[2]][0]:self.parameters_subscript[ord[2]][-1]+1]))
        else:
            # print(self.parameters_subscript)
            # print(self.parameters_subscript[ord[3]][0], self.parameters_subscript[ord[3]][-1] + 1)
            # print(self.parameters_subscript[ord[4]][0], self.parameters_subscript[ord[4]][-1] + 1)
            series_num.append(1)
            series_num.append(self._sample_step_2(self.p[self.parameters_subscript[ord[3]][0]:self.parameters_subscript[ord[3]][-1]+1]))
            series_num.append(self._sample_step_2(self.p[self.parameters_subscript[ord[4]][0]:self.parameters_subscript[ord[4]][-1]+1]))
        return series_num

    # 判断概率是否溢出
    '''
    def issample(self):
        for i in range(len(self.parameters_count)):
            pros_sum = sum(self.p[self.parameters_subscript[i][0]:self.parameters_subscript[i][-1]+1])
            #print('pros_sum',pros_sum)
            if pros_sum > 1:
                return 0
        return 1
    '''
    def _get_parameters(self,series_num):
        # setting, pros,const = load_conf()
        if series_num[0] == 1:
            # print(self.setting['pooling']['pooling_type']['val'][series_num[1]])
            # print(self.setting['pooling']['kernel_size']['val'][series_num[2]])
            return 'pooling',self.setting['pooling']['pooling_type']['val'][series_num[1]], self.setting['pooling']['kernel_size']['val'][series_num[2]]
        if series_num[0] == 0:
            return 'conv',self.setting['conv']['filter_size']['val'][series_num[1]], self.setting['conv']['kernel_size']['val'][series_num[2]], self.setting['conv']['activation']['val'][series_num[3]]

    #采样，最终结果
    '''
    	series_num is the list of paramters series numbers 
    	parameters is parameters' value depends on series numbers
    '''
    def sample(self, n):   	
        parameters = []
        #print(self.issample())

        for i in range(n):
            series_num = self._sample_step_1()
            parameters.append(self._get_parameters(series_num))
        return parameters


    def get_cell_log(self,POOL,PATH,date):
        for i, j in enumerate(POOL):
            s = 'nn_param_' + str(i) + '_' + str(date)
            fp = open(PATH + s, "wb")
            # print(s)
            pickle.dump(j.cell_list, fp)

        
if __name__ == '__main__':

    p = [0.7,0.103,0.1,0.1,0.13,0.04,0.1,0.03,0.03,0.003,0.004,0.003,0.003,0.003,0.003,0.0004,0.000003,0.00003,0.00002,0.00003,0.00004,0.00003,0.00003,0.00004,0.0003,0.00003,0.00003,0.00001,0.00002,0.0003,0.0004,0.00004,0.0003,0.0001,0.02,0.004,0.03,0.01,0.2,0.03,0.04,0.1,0.1,0.1]#优化给出
    newp = [0.8,0.2,0.3,0.6,0.2,0.4,0.3,0.6,0.2,0.3,0.4,0.3,0.3,0.7,0.2,0.4,0.3,0.5,0.2,0.3,0.4,0.3,0.3,0.1,0.2,0.4,0.3,0.1,0.2,0.3,0.4,0.3,0.3,0.1,0.2,0.4,0.3,0.1,0.2,0.3,0.4,0.1,0.1,0.1]#优化更新
    print(len(p))
    cell_list = []
#    #append cell_list
#    def renewcl(r):
#        if r[0]=='conv':
#            u,a, b, c = r
#            cell1 = ConvolutionalCell(filter_size=a, kernel_size=b, activation=c)
#        elif r[0]=='pooling':
#            u,a, b = r
#            cell1 = PoolingCell(pool_type=a,pool_size=b)
#        cell_list.append(cell1)


    sampling = Sampler()
    #print(sampling.ls,sampling.q)
    sampling.renewp(p)
    r = sampling.sample(2)#采样返回结果
    print('sample:',r)
    
    # print('filter_size:',cell_list[0].filter_size)
    
    sampling.renewp(newp)#更新p
    print('new p:',sampling.p)
    newr = sampling.sample(10)# 重新采样
    print('newsample:',newr)
    
    # print('new filter_size:',cell_list[1].filter_size)
    
    #print(cell_list[0],cell_list[1])
    #POOL = pickle.load('log_test')
    #sampling.get_cell_log(POOL,"C:/TEST/",'2018')



