'''
@Project ：jueceshu 
@File    ：treeutils.py
@IDE     ：PyCharm 
@Author  ：zsx
@Date    ：27/06/2023 21:24
@Desc    :决策树规则提取，从第一个叶子节点开始，到最后一个非叶子节点为止，提取对应每个叶子节点的值
'''
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


def get_data():
    df = pd.read_csv('data/train.csv')
    return df


def fun_feature_sel(X_train, y_train):
    """
    :param X_train:
    :param y_train:
    :return: 返回选择特征，根据特征重要性返回
    """
    model = DecisionTreeClassifier(max_depth=5)
    model.fit(X_train, y_train)
    df_col_import = pd.DataFrame(model.feature_importances_, index=model.feature_names_in_, columns=['import_'])
    col_list = list(df_col_import[df_col_import['import_'] >= 0.001].index)
    print(len(col_list))
    return col_list


def fun_create_data(model):
    """
    生成数字编号和各个节点一一对应内容以及叶子节点对应的下一层节点明细
    :param model: 模型
    :return: d2>层级指向数据框、d1>条件序号对应数据数据框
    """
    # 模型dot文件
    str1 = tree.export_graphviz(model)
    str1_split_list = str1.split(';')
    list_next = [i for i in str1_split_list if '->' in i]  # 节点指向情况
    list_where_num = [i for i in str1_split_list if '->' not in i]  # 每个条件对应的序号情况
    d1 = pd.DataFrame(list_where_num, columns=['A']).iloc[2:, :]
    # 获取每一个叶子节点对应的判断条件详情内容
    d1['num'] = d1['A'].apply(lambda x: x.split()[0].replace('\n', ''))
    # 获取每一个叶子节点对应的下一层级的叶子节点/非叶子节点
    d2 = pd.DataFrame(list_next, columns=['A'])
    d2['now_'] = d2['A'].apply(lambda x: x.split(' ')[0].replace('\n', ''))
    d2['next_'] = d2['A'].apply(lambda x: x.split(' ')[2].replace(' ', ''))
    return d1, d2


def fun_create_col_name(data):
    """
    用来动态生成层级字段名，多少个对应的非叶子节点，即多少组字段名
    :param data: 层级指向数据框
    :return: 根据节点层级创建多个字段组合
    """
    # 每个节点有唯一的编号，寻找整个决策树中最大的节点序号，用来确定字段名的长度
    num = max([int(i) for i in set(data['now_']) | set(data['next_'])])
    data = data[['now_', 'next_']]
    col_list_all = []
    for i in range(0, num):
        col_ = []
        for j in data.columns:
            col_.append(j + str(i))
        col_list_all.append(col_)
    return col_list_all


def fun_every_rule(data, col_list_all):
    """
    获取每一条规则组成的叶子节点和非叶子节点序号
    :param data: 叶子节点和下一层节点指向数据框
    :param col_list_all: 数据框字段名list
    :return:
    """
    data = data[['now_', 'next_']]
    # 以决策树为起点，判读是否存在叶子节点
    d1_sel = data[data['now_'] == '0']
    d1_sel.columns = col_list_all[0]
    # 添加第一个数据，df_list[-1]为每次添加的最后一个数据
    df_list = [d1_sel]
    for i in range(len(col_list_all) - 1):
        data.columns = col_list_all[i + 1]
        # 利用当前节点对应的数据框和下一层节点对应的数据框进行左链接
        df_cb = pd.merge(df_list[-1], data, left_on='next_' + str(i), right_on='now_' + str(i + 1), how='left')
        df_list.append(df_cb)
    # 获取单个规则最终的节点走向，纬度为:1 X len(col_list_all)
    df_cb_fin = df_list[-1]
    # 获取每一层层级指向，只保留对应的下一层
    col_list = [i for i in df_cb_fin.columns if 'next' in i]
    col_list.insert(0, 'now_0')
    value_list = df_cb_fin[col_list].values
    # 输出每一条规则不为空的组成条件
    list_value_fin = []
    for i in value_list:
        list_ = []
        for j in i:
            # 每一条规则出现第一个节点为空时，后面所有的节点都为空
            if pd.isna(j) == False:
                list_.append(j)
        list_value_fin.append(list_)
    return list_value_fin


def fun_case_next_num(data):
    """
    判断每一个叶子节点后面是否存在其他节点情况。规则层级（子节点）

    :param data: 节点指向数据框
    :return:
    """
    dict_val = {}
    # 遍历所有的节点序号，通过层级指向判断该节点后面是否存在其他节点
    for i in set([j for i in data.values for j in i]):
        d_sel = data[data['now_'] == i].drop_duplicates()
        d3 = d_sel['next_'].values
        if len(d3) != 0:
            dict_val[i] = d3
    return dict_val


# 规则信息和编号字典
def fun_case_num_info(data):
    """
    生成节点序号和节点详情的字典
    :param data: 节点详情和节点编号数据框
    :return:
    """
    return data.set_index('num')['A'].to_dict()


def fun_every_rule_num_info(list_value, dict_info, dict_next):
    """
    根据节点，生成对应的各个节点的规则信息
    :param list_value:规则集合
    :param dict_info:规则信息和编号字典
    :param dict_next:规则层级（子节点）
    :return:
    """
    list_fin_ = []
    for one in list_value:
        list_ = []
        for i in range(len(one)):
            val_every = dict_info[one[i]]
            # 通过节点符号，判断当前节点是否为左节点，左True，右节点，False,当为右节点时，规则内容符号变换
            if one[i] in dict_next.keys():
                print('one[i]:{},----one[i+1]:{},---dict_next[one[i]]:{}'.format(one[i], one[i + 1], dict_next[one[i]]))
                if one[i + 1] == dict_next[one[i]][0]:
                    list_.append(val_every)
                else:
                    list_.append(val_every.replace("<=", '>'))
            else:
                list_.append(dict_info[one[i]])
        list_fin_.append(list_)
    return list_fin_


# 特征名字和对应代码
def fun_feature_name_dict(model):
    """
    生成模型选择特征和字段名一一对应
    :param model:
    :return:
    """
    d3 = pd.DataFrame(model.feature_names_in_, columns=['A']).reset_index()
    d3['val_key'] = d3['index'].apply(lambda x: 'X[' + str(x) + ']')
    return d3.set_index('val_key')['A'].to_dict()


# 规则条件提取
def fun_every_rule_info(list_value):
    """
    根据各个节点的详情，进行条件内容提取
    :param list_value:
    :return:
    """
    # 所有规则中，每个规则——各个条件的list
    list_ = []
    # 所有规则，每个规则——各个条件和数据结果拼接后的list
    list_str = []
    for every_rule in list_value:
        every_ = []
        for i in range(len(every_rule)):
            a = every_rule[i]
            if i < len(every_rule) - 1:
                # 提取各个条件的字段名和阈值以及阈值符号
                str2 = re.findall('(X.*)gini', a)[0].replace('\\n', '')
                every_.append(str2)
            else:
                # 提取每个规则最后的样本数据量
                str2 = re.findall('value = (.*)', a)[0]
                every_.append(str2)
        # 各个条件及样本数据拼接组合，
        list_str.append(' & '.join(every_[0:-1]) + ' : ' + every_[-1])
        list_.append(every_)
    return list_, list_str


# 特征和字段名的映射
def fun_rule_feature_ys(list_value, dict_feature_name):
    '''
    规则中，特征名字和字段名映射
    :param list_value: 规则字符串
    :param dict_feature_name: 特征名字典
    :return:
    '''
    list_ = []
    for rule in list_value:
        rule_list = [rule]
        # 遍历每个字段名，每遍历一次，替换后的结果添加在rule_lsit中，每次拿最新添加的结果再次进行替换，直至遍历完成
        for key in dict_feature_name.keys():
            if key in rule_list[-1]:
                str4 = rule_list[-1].replace(key, dict_feature_name[key])
                rule_list.append(str4)
        list_.append(rule_list[-1])
    return list_


# 规则明细结果写出
def fun_rule_write(list_str):
    """
    规则及规则对应样本数据输出
    :param list_str: 所有规则中，拼接完成后list
    :return:
    """
    df_fin = pd.DataFrame(list_str, columns=['rule_mx'])
    df_fin['good'] = df_fin['rule_mx'].apply(lambda x: re.findall('\d+', x)[-2])
    df_fin['bad'] = df_fin['rule_mx'].apply(lambda x: re.findall('\d+', x)[-1])
    df_fin['good'] = df_fin['good'].astype(int)
    df_fin['bad'] = df_fin['bad'].astype(int)
    df_fin['all'] = df_fin['good'] + df_fin['bad']
    df_fin['white/black'] = df_fin['good'] / df_fin['bad']
    df_fin['bad_rate'] = df_fin['bad'] / df_fin['all']
    return df_fin


from collections import defaultdict


def transform_list_to_dict(input_list):
    """
    输入一个列表，统计每个元素出现的个数，并完成一次和多次的拆分
    :param input_list:
    :return:
    """
    count_dict = defaultdict(int)
    for item in input_list:
        count_dict[item] += 1
    value_01 = [key for key, value in count_dict.items() if value == 1]
    value_02 = [key for key, value in count_dict.items() if value > 1]
    return value_01, value_02


def make_rule_yuzhi(input_list):
    """
    列表组合，形如：# a = [['age', '<=', 30], ['age', '>', 15], ['heigt', '<', 180], ['heigt', '<', 170]]
    :param input_list:
    :return: 保留唯一条件，同大取大，同小取小
    """
    input_list01 = [i[0] for i in input_list]
    unique_cols, duplicate_cols = transform_list_to_dict([i for i in input_list01])
    result_list = []
    # 遍历唯一特征
    for col in unique_cols:
        result_list.append(next(item for item in input_list if item[0] == col))
    # 遍历出现多个特征
    for col in duplicate_cols:
        sub_list = [i[1] for i in input_list if i[0] == col]
        # 同一特征对应特征出现的次数对应的逻辑符号
        unique_subs, duplicate_subs = transform_list_to_dict(sub_list)
        # 遍历一个？ 感觉无用，能到这块说明出现两次了
        # for sub in unique_subs:
        #     result_list.append(next(item for item in input_list if item[0] == col and item[1] == sub))
        for sub in duplicate_subs:
            if sub in ['<=', '<']:
                value_tmp = min([i[2] for i in input_list if i[0] == col and i[1] == sub])
            elif sub in ['>=', '>']:
                value_tmp = max([i[2] for i in input_list if i[0] == col and i[1] == sub])
            result_list.append(
                next(item for item in input_list if item[0] == col and item[1] == sub and item[2] == value_tmp))
    return result_list

# a = [['age', '<=', 30], ['age', '>', 15], ['heigt', '<', 180], ['heigt', '<', 170]]

# make_rule_yuzhi(a)
# [['age', '<=', 30], ['age', '>', 15], ['heigt', '<', 170]]
