# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


import treeutils
import matplotlib.pyplot as plt

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = treeutils.get_data()
    X = df.loc[:, 'XINGBIE':'DKLL']
    Y = df['label']
    X_train, X_test, y_train, y_test = treeutils.train_test_split(X, Y)
    # 特征选择
    col_sel = treeutils.fun_feature_sel(X_train, y_train)
    X_train_sel = X_train[col_sel]
    # 选择后的特征进行模型训练
    model = treeutils.DecisionTreeClassifier(max_depth=5)
    model.fit(X_train_sel, y_train)
    # 决策树可视化
    plt.figure(figsize=[8, 8])
    treeutils.tree.plot_tree(model)
    plt.savefig('tree.png')
    # 生成数字编号和各个节点一一对应内容以及叶子节点对应的下一层节点明细
    d1, d2 = treeutils.fun_create_data(model)
    # 用来动态生成层级字段名，
    col_list_all = treeutils.fun_create_col_name(d2)
    # 获取每一条规则组成的叶子节点和非叶子节点序号
    list_val_fin = treeutils.fun_every_rule(d2, col_list_all)
    # 生成节点序号和节点详情的字典
    dict_num_info = treeutils.fun_case_num_info(d1)
    # 判断每一个叶子节点后面是否存在其他节点情况
    dict_num_case = treeutils.fun_case_next_num(d2)
    print("list_val_fin", list_val_fin)
    # 根据节点，生成对应的各个节点的规则信息
    list_val_fin_str = treeutils.fun_every_rule_num_info(list_value=list_val_fin, dict_info=dict_num_info,
                                                         dict_next=dict_num_case)
    # 根据各个节点的详情，进行条件内容提取
    rule_info, rule_info_str = treeutils.fun_every_rule_info(list_value=list_val_fin_str)
    dict_feature_name = treeutils.fun_feature_name_dict(model=model)

    # 生成模型选择特征和字段名一一对应
    rule_info_str_fin = treeutils.fun_rule_feature_ys(list_value=rule_info_str, dict_feature_name=dict_feature_name)
    print('----rule_info_str_fin---', rule_info_str_fin)
    df_fin = treeutils.fun_rule_write(list_str=rule_info_str_fin)
    print('=====df_fin=', df_fin[['good', 'bad', 'white/black', 'bad_rate']])
