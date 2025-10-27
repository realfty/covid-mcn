import numpy as np
import pandas as pd
import networkx as nx
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV
import community.community_louvain as louvain
import random
import itertools
from scipy import stats
import datetime
import csv
import os

def all_min_dominating_set(nxG):
    min_dominating_set_result = []
    strength_of_all_min_dominating_set = []
    # nxG中的节点非重复且不考虑顺序的排列组合,暴力求解最小支配集
    node_list = nx.nodes(nxG)
    node_num = nxG.number_of_nodes()
    # 初始化最小支配集大小为所有节点
    min_dominating_size = node_num
    print("start searching....")
    for i in range(1,node_num+1):
        print("searching for size " + str(i) +" ...")
        print(str(datetime.datetime.now()))
        set_list = list(itertools.combinations(node_list, i))
        for temp_set in set_list:
            if nx.is_dominating_set(nxG,temp_set):
                min_dominating_size = i
                min_dominating_set_result.append(temp_set)
                # 计算最小支配集权重
                temp_total_strength = 0
                for dominating_node in temp_set:
                    temp_strength = nxG.degree(dominating_node,weight='weight')
                    temp_total_strength = temp_total_strength + temp_strength
                strength_of_all_min_dominating_set.append(temp_total_strength)
        # 如果已经找到最小支配集，那么就不再继续寻找最小支配集
        if i >= min_dominating_size:
            break
        print(str(datetime.datetime.now()))
    return min_dominating_set_result,strength_of_all_min_dominating_set

def dominating_frequency(all_dom_set, nxG):
    num_dom_set = len(all_dom_set)
    node_num = nxG.number_of_nodes()
    # 初始化支配节点计数
    as_dom_node_count = {node: 0 for node in nxG.nodes()}
    
    # 统计每个节点作为支配节点的频率
    for min_dom_set in all_dom_set:
        for dom_node in min_dom_set:
            as_dom_node_count[dom_node] += 1

    # 计算频率
    for node in as_dom_node_count:
        as_dom_node_count[node] /= num_dom_set
    print(as_dom_node_count)
    
    return as_dom_node_count

def matrix_preprocess(matrix):
    number_of_nodes = matrix.shape[1]
    matrix_result = matrix
    # 去对角线
    for i in range(0,number_of_nodes):
        matrix_result[i,i] = 0
    # 取绝对值
    matrix_result = abs(matrix_result)
    return matrix_result

def module_controllability(nxG,all_dom_set,louvain_communities):

    # louvain_communities = louvain.best_partition(nxG)
    number_of_communities = max(louvain_communities.values())+1
    print("module number: "+str(number_of_communities))
    # init
    module = {}
    for index in range(0,number_of_communities):
        module[index] = []
    # finding module
    for node in louvain_communities:
        community_index = louvain_communities[node]
        module[community_index].append(node)

    for i in range(0,number_of_communities):
        print("module "+str(i)+" has "+ str(module[i])+" nodes")

    # 初始化结果
    average_module_controllability_result = {}
    for module_source in module:
        for module_target in module:
            average_module_controllability_result[str(module_source) + "_" + str(module_target)] = 0


    # all_dom_set,strength_of_all_dom_set = all_min_dominating_set(nxG)
    # print(all_dom_set)
    for min_dom_set in all_dom_set:
        dominated_area = {}
        for dom_node in min_dom_set:
            temp_neighbor = set()
            temp_neighbor.clear()
            for neighbor in nxG.neighbors(dom_node):
                temp_neighbor.add(neighbor)
            #支配域还有节点自身
            temp_neighbor.add(dom_node)
            dominated_area[dom_node] = temp_neighbor

        # 计算社团支配域
        modules_control_area = {}
        for module_index in module:
            node_in_module = module[module_index]
            single_module_control_area = set()
            for node in node_in_module:
                if node in min_dom_set:
                    # 添加module支配域
                    temp_dom_set = set()
                    for temp_node in dominated_area[node]:
                        single_module_control_area.add(temp_node)
            modules_control_area[module_index] = single_module_control_area
        # print(modules_control_area)

        # 计算社团间支配能力
        temp_module_controllability_result = {}
        temp_module_controllability_result.clear()
        for module_source in module:
            for module_target in module:
                # 社团控制域
                control_area = modules_control_area[module_source]
                # 被控社团节点集
                target_module_area = module[module_target]
                # 两者交集大小
                target_module_area_set = set(target_module_area)
                inter = control_area.intersection(target_module_area_set)
                temp_module_controllability_result[str(module_source)+"_"+str(module_target)] = len(inter) / len(target_module_area)
                average_module_controllability_result[str(module_source) + "_" + str(module_target)] = average_module_controllability_result[str(module_source) + "_" + str(module_target)] + (len(inter) / len(target_module_area))
        print("dom_set: "+str(min_dom_set) + "   module_controllability: "+ str(temp_module_controllability_result))
    for total_module_controllability in average_module_controllability_result:
        average_module_controllability_result[total_module_controllability] = average_module_controllability_result[total_module_controllability] / len(all_dom_set)
    print("average_module_controllability: "+ str(average_module_controllability_result))
    return average_module_controllability_result

def network_analysis(nxG):
    network_analysis_result = {}
    # 聚集系数
    clustering = nx.clustering(nxG)
    # 接近中心性
    closeness = nx.closeness_centrality(nxG)
    # 介数中心性
    betweenness = nx.betweenness_centrality(nxG)
    # 度中心性（这里要补充进强度）
    degree = nx.degree_centrality(nxG)
    # 平均强度
    average_strength = {}
    print(nxG.nodes)
    for node in list(nxG.nodes):
        if nxG.degree(node) != 0:
            average_strength[node] = nxG.degree(node,weight='weight') / nxG.degree(node)
        else:
            average_strength[node] = 0
    # k core
    nG_nonself = nxG
    nG_nonself.remove_edges_from(nx.selfloop_edges(nxG))
    kcore = nx.core_number(nG_nonself)

    network_analysis_result["clustering"] = clustering
    network_analysis_result["closeness"] = closeness
    network_analysis_result["betweenness"] = betweenness
    network_analysis_result["degree"] = degree
    network_analysis_result["average_strength"] = average_strength
    network_analysis_result["kcore"] = kcore

    return network_analysis_result


def greedy_minimum_dominating_set(nxG, times):
    min_dominating_set = []

    for time in range(times):
        nxG_copy = nxG.copy()
        dominating_set = []

        while nxG_copy.nodes():
            node = random.choice(list(nxG_copy.nodes()))
            dominating_set.append(node)
            remove_list = []
            remove_list.clear()
            remove_list.append(node)
            for neighbor in nxG_copy.neighbors(node):
                remove_list.append(neighbor)

            for node in remove_list:
                nxG_copy.remove_node(node)

        dominating_set = set(dominating_set)
        if len(min_dominating_set) == 0:
            min_dominating_set.append(dominating_set)
        elif len(min_dominating_set[0]) == len(dominating_set) and dominating_set not in min_dominating_set:
            min_dominating_set.append(dominating_set)
        elif len(min_dominating_set[0]) > len(dominating_set):
            min_dominating_set.clear()
            min_dominating_set.append(dominating_set)

        print("times: " + str(time + 1) +" MDSet size: "+ str(len(min_dominating_set[0]))+ " MDSet number: "+ str(len(min_dominating_set)) +"  MDSet: " + str(min_dominating_set))

    return min_dominating_set
 # 整合贪心算法寻找的所有最小支配集
def consolidate_and_save_results(all_sets, result_path, data_file, data_type="greedy_result"):
    unique_sets = set(map(tuple, all_sets))
    min_set_size = min(len(s) for s in unique_sets)
    smallest_sets = [set(s) for s in unique_sets if len(s) == min_set_size]

    root_dir = result_path
    output_file = os.path.join(root_dir, data_type, f"{data_file}_consolidate_result.txt")

    # 确保路径存在
    os.makedirs(os.path.join(root_dir, data_type), exist_ok=True)

    with open(output_file, 'w') as f:
        for index, set_items in enumerate(smallest_sets, start=1):
            f.write(f"Set {index}: {set(set_items)}\n")

    print(f"整合后的最小支配集结果已保存到 {output_file}")
    return output_file   

def generate_percolation_net(matrix):
    # 筛选阈值,最大弱连通分支被破坏时，终止筛选
    temp_list = matrix.flatten(order="C")
    # 只看绝对强度
    temp_list = abs(temp_list)
    sort_list = np.sort(temp_list)
    sort_index = -1
    largest_cc = matrix.shape[0]
    nxG = nx.DiGraph()
    return_nxG = nxG.copy()
    # 权重取绝对值
    for i in range(0, matrix.shape[0]):
        for j in range(0, matrix.shape[1]):
            if matrix[i,j] < 0:
                matrix[i, j] = abs(matrix[i,j])
    return_matrix = matrix.copy()
    temp_matrix = matrix.copy()
    while largest_cc == matrix.shape[0]:
        return_nxG = nxG.copy()
        return_matrix = temp_matrix.copy()
        nxG.clear()
        for i in range(0,matrix.shape[0]):
            nxG.add_node(i)
        sort_index = sort_index + 1
        temp_data = matrix
        threshold = sort_list[sort_index]
        # print("threshold: "+str(threshold))
        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[1]):
                if abs(temp_data[i, j]) > threshold:
                    nxG.add_edge(i, j, weight=abs(temp_data[i, j]))
                else:
                    temp_matrix[i,j] = 0
        largest_cc = max(nx.weakly_connected_components(nxG),key=len).__len__()
        # print("largest_cc: " + str(largest_cc))
        # print()
    return return_nxG.to_undirected(),return_matrix

def save_network_analysis_to_csv(specific_result_path, data_file, nxG, network_analysis_result, louvain_communities, as_dom_node_count, average_module_controllability_result, all_dom_set, algorithm):
    # 确保 specific_result_path 文件夹存在
    os.makedirs(specific_result_path, exist_ok=True)

    # 创建节点属性的 DataFrame
    node_data = {
        "item": list(nxG.nodes),
        "degree_centrality": [network_analysis_result["degree"][node] for node in nxG.nodes],
        "average_strength": [network_analysis_result["average_strength"][node] for node in nxG.nodes],
        "clustering": [network_analysis_result["clustering"][node] for node in nxG.nodes],
        "closeness": [network_analysis_result["closeness"][node] for node in nxG.nodes],
        "betweenness": [network_analysis_result["betweenness"][node] for node in nxG.nodes],
        "kcore": [network_analysis_result["kcore"][node] for node in nxG.nodes],
        "module": [louvain_communities[node] for node in nxG.nodes],
        "CF": [as_dom_node_count[node] for node in nxG.nodes]
    }
    node_df = pd.DataFrame(node_data)

    # 保存节点属性的 DataFrame 到 CSV
    node_df.to_csv(os.path.join(specific_result_path, f"{data_file}_nodes.csv"), index=False)

    # 创建边属性的 DataFrame
    edge_data = {
        "source": [edge[0] for edge in nxG.edges],
        "target": [edge[1] for edge in nxG.edges],
        "weight": [nxG.get_edge_data(edge[0], edge[1])['weight'] for edge in nxG.edges]
    }
    edge_df = pd.DataFrame(edge_data)

    # 保存边属性的 DataFrame 到 CSV
    edge_df.to_csv(os.path.join(specific_result_path, f"{data_file}_edges.csv"), index=False)

    # 创建模块可控性结果的 DataFrame
    module_controllability_data = {
        "module_2_module(direct)": list(average_module_controllability_result.keys()),
        "AMCS": list(average_module_controllability_result.values())
    }
    module_controllability_df = pd.DataFrame(module_controllability_data)

    # 保存模块可控性结果的 DataFrame 到 CSV
    module_controllability_df.to_csv(os.path.join(specific_result_path, f"{data_file}_module_controllability.csv"), index=False)

    # 保存支配频率到 CSV
    try:
        cf_output_file = os.path.join(specific_result_path, f"{data_file}_CF.csv")
        cf_df = pd.DataFrame(list(as_dom_node_count.items()), columns=['Node', 'Frequency'])
        cf_df.to_csv(cf_output_file, index=False, encoding='utf-8')
        print(f"支配频率已保存到文件: {cf_output_file}")
    except IOError as e:
        print(f"无法写入文件 {cf_output_file}，错误：{e}")

    # 保存最小支配集及其权重
    try:
        with open(os.path.join(specific_result_path, f"{data_file}_dominating_set.csv"), 'w', encoding='utf-8') as file_object:
            file_object.write(f"Minimum dominating set ({algorithm}) size: {len(all_dom_set[0])}\n")
            for mds in all_dom_set:
                file_object.write(f"{mds}\n")
        print(f"最小支配集已保存到 {specific_result_path}")
    except IOError as e:
        print(f"无法写入文件 {specific_result_path}，错误：{e}")

    print(f"数据已保存到 {specific_result_path}，并生成了 CSV 文件。")