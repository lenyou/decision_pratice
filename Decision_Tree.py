import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold



# 计算数据集x的经验熵H(x)
def calc_ent(x):
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp
    return ent

# 计算条件熵H(y/x)
def calc_condition_ent(x, y):
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        sub_y = y[x == x_value]
        temp_ent = calc_ent(sub_y)
        ent += (float(sub_y.shape[0]) / y.shape[0]) * temp_ent
    return ent

def calc_ent_grap(x,y):
    base_ent = calc_ent(y)
    condition_ent = calc_condition_ent(x, y)
    ent_grap = base_ent - condition_ent
    return ent_grap



def process_continuous_calc_ent_grap(x,y):
    copy_x = x.copy()
    sorted(copy_x)
    previous_pivot = None
    center_points_list = []
    if len(x)==0:
        center_points_list=copy_x
    else:
        for i in copy_x:
            if previous_pivot is not None:
                center_points_list.append((previous_pivot+i)/2.0)
            previous_pivot=i
    max_ent = None
    center_record = 0
    for center in center_points_list:
        binary_x = x>=center
        tmp_grap = calc_ent_grap(binary_x,y)
        tmp_H = calc_ent(binary_x)
        tmp = tmp_grap/tmp_H
        if max_ent is None:
            max_ent=tmp
            center_record=center
        else:
            if tmp>max_ent:
                max_ent=tmp
                center_record=center
    return max_ent,center_record




class Tree:
    def __init__(self,dim_index,split_record):
        self.dim_index=dim_index
        self.split_record=split_record
        self.left=None
        self.right=None

    def add_sub_tree(self, tree, mode='left'):
        if mode not in ['left','right','leaf']:
            raise NotImplementedError('leaf')
        if mode == 'left':
            self.left = tree
        elif mode=='right':
            self.right = tree
        elif mode=='leaf':
            self.left=None
            self.right=None
    def blank_tree(self):
        return None


def recursive_predict(tree,inputdata):
    dim_index = tree.dim_index
    split_record = tree.split_record
    current_value = inputdata[dim_index]

    if current_value>=split_record:
        if tree.right is None:
            return 0
        result = recursive_predict(tree.right,inputdata)
    else:
        if tree.left is None:
            return 1
        result = recursive_predict(tree.left,inputdata)
    return result

def k_fold(tran_data,train_target,k=5):
    length = train_data.shape[0]
    interval = length/k
    interval_list = [(k*interval,min(length,(k+1)*interval)) for i in range(k-1)]

    for ran_ge in interval_list:
        yield (train_data[ran_ge[0]:ran_ge[1],:],train_target[ran_ge[0]:ran_ge[1]])




def recursive_train(train_data,train_target,mode,train_tree=None,thresh=0.5):

    if len(np.unique(train_target))==1:
        return None

    elif train_data.shape[1]==0:
        return None

    else:
        max_ent = None
        index_record = None
        split_record = None
        for i in range(train_data.shape[1]):
            x = train_data[:,i]
            y = train_target
            gda,split_point = process_continuous_calc_ent_grap(x,y)
            if max_ent is None:
                max_ent=gda
                split_record=split_point
                index_record=i
            else:
                if gda>max_ent:
                    max_ent=gda
                    split_record=split_point
                    index_record=i
        # print (max_ent)
        if max_ent<thresh:
            return None

        #construction treh

        left_train_data=train_data[train_data[:,index_record]<split_record,:]
        right_train_data = train_data[train_data[:,index_record]>=split_record,:]
        left_target = train_target[train_data[:,index_record]<split_record]
        right_target = train_target[train_data[:,index_record]>=split_record]
        current_node = Tree(index_record,split_record)
        current_node.add_sub_tree(recursive_train(left_train_data,left_target,'left',current_node),mode='left')
        current_node.add_sub_tree(recursive_train(right_train_data,right_target,'right',current_node),mode='right')
        if mode!='root':
            train_tree.add_sub_tree(current_node,mode=mode)
        return current_node




if __name__ == '__main__':

    data = load_breast_cancer()
    target = data.target
    train_data = data.data
    train_data = np.concatenate((train_data, target[:,np.newaxis]), axis=1)
    kfolds = KFold(n_splits=5, shuffle=True)
    acc_list = []
    for f_data,f_target in kfolds.split(train_data,target):
        train_array = None
        test_array = None
        root_node = Tree(-1,-1)
        for i in f_data:
            if train_array is None:
                train_array = train_data[i,:][np.newaxis,:]
            else:
                train_array = np.concatenate((train_array,train_data[i,:][np.newaxis,:]),axis=0)
        for i in f_target:
            if test_array is None:
                test_array =  train_data[i,:][np.newaxis,:]
            else:
                test_array = np.concatenate((train_array,train_data[i,:][np.newaxis,:]),axis=0)
        print (train_array.shape)
        print (test_array.shape)
        decision_tree = recursive_train(train_array[:,:30],train_array[:,-1],train_tree=root_node,mode='root')
        correct_num = 0
        for item in test_array:
            result = recursive_predict(decision_tree,item[:30])
            if result == item[-1]:
                correct_num+=1
        acc_list.append(float(correct_num)/test_array.shape[0])
        print (float(correct_num)/test_array.shape[0])

        