import os
import numpy as np
import networkx as nx
import community.community_louvain as louvain
import datetime
import pandas as pd
from tools import (
    all_min_dominating_set,
    dominating_frequency,
    matrix_preprocess,
    module_controllability,
    greedy_minimum_dominating_set,
    consolidate_and_save_results,
    generate_percolation_net,
    save_network_analysis_to_csv,
    network_analysis
)
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV

def process_covid_phase(data_path, result_path, phase_file, log_file):
    try:
        # 构建结果保存路径
        specific_result_path = os.path.join(result_path, phase_file)
        os.makedirs(specific_result_path, exist_ok=True)

        # 读取CSV数据
        pd_data = pd.read_csv(os.path.join(data_path, f"{phase_file}.csv"))
        log_file.write(f"{datetime.datetime.now()} | 处理阶段: {phase_file}\n")
        print(pd_data.head())
        
        # 检查缺失值
        null_checks = pd_data.isnull().any()
        log_file.write(f"缺失值检查: {null_checks}\n")
        print(null_checks)

        # 使用GraphicalLasso构建网络
        estimator = GraphicalLasso(alpha=0.4)
        # 使用GraphicalLassoCV自动确定alpha
        # estimator = GraphicalLassoCV()
        estimator.fit(pd_data)
        log_file.write(f"精度矩阵形状: {estimator.precision_.shape}\n")
        
        # 计算偏相关矩阵
        diag_sqrt = np.sqrt(np.diag(estimator.precision_))
        partial_corr_matrix = -estimator.precision_ / np.outer(diag_sqrt, diag_sqrt)
        np.fill_diagonal(partial_corr_matrix, 1)
        log_file.write("偏相关矩阵计算完成\n")
        
        # 预处理矩阵并构建网络
        matrix = matrix_preprocess(partial_corr_matrix)
        nxG = nx.Graph(matrix)
        log_file.write(f"边数: {nxG.number_of_edges()}\n")
        
        # 网络分析
        network_analysis_result = network_analysis(nxG)
        
        # 社团检测
        louvain_communities = louvain.best_partition(nxG)
        number_of_communities = max(louvain_communities.values()) + 1
        log_file.write(f"社团数: {number_of_communities}\n")
        
        # 计算支配集
        log_file.write(f"{datetime.datetime.now()} | 开始计算支配集\n")
        # all_dom_set = greedy_minimum_dominating_set(nxG, 5000)
        all_dom_set,strength_of_all_dom_set = all_min_dominating_set(nxG)
        algorithm = "precise"
        log_file.write(f"{datetime.datetime.now()} | 支配集计算完成\n")
        
        # 保存支配集结果
        consolidate_and_save_results(all_dom_set, specific_result_path, phase_file)
        
        # 计算支配频率
        as_dom_node_count = dominating_frequency(all_dom_set, nxG)
        log_file.write(f"支配频率计算完成\n")
        
        # 计算模块可控性
        average_module_controllability_result = module_controllability(nxG, all_dom_set, louvain_communities)
        log_file.write(f"模块可控性计算完成\n")
        
        # 保存网络文件
        nx.write_gexf(nxG, os.path.join(specific_result_path, f"{phase_file}.gexf"))
        
        # 保存分析结果
        save_network_analysis_to_csv(
            specific_result_path,
            phase_file,
            nxG,
            network_analysis_result,
            louvain_communities,
            as_dom_node_count,
            average_module_controllability_result,
            all_dom_set,
            algorithm
        )
        
        log_file.write(f"{datetime.datetime.now()} | {phase_file} 处理完成\n\n")
        print(f"{phase_file} 处理完成")
        
    except Exception as e:
        error_msg = f"处理 {phase_file} 时发生错误: {str(e)}"
        print(error_msg)
        log_file.write(f"{error_msg}\n")

def main():
    # 设置路径 - 已更新为新的路径
    data_path = r"/Volumes/Apple/globalMind/US/US_18-24_data/filter_columns"
    result_path = r"/Volumes/Apple/globalMind/US/主结果"
    log_dir = os.path.join(result_path, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志文件
    log_file_path = os.path.join(log_dir, "covid_network_analysis_log.txt")
    
    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f"{datetime.datetime.now()} | 开始COVID数据网络分析\n")
        print("开始COVID数据网络分析...")
        
        # 获取所有阶段文件
        phase_files = [f.split('.')[0] for f in os.listdir(data_path) 
                      if f.endswith('.csv') and 'nan' not in f]
        
        log_file.write(f"发现 {len(phase_files)} 个阶段数据需要处理\n")
        print(f"发现 {len(phase_files)} 个阶段数据需要处理")
        
        # 处理每个阶段
        for idx, phase_file in enumerate(phase_files, 1):
            print(f"处理阶段 {idx}/{len(phase_files)}: {phase_file}")
            process_covid_phase(data_path, result_path, phase_file, log_file)
        
        log_file.write(f"{datetime.datetime.now()} | 所有阶段处理完成\n")
        print("所有阶段处理完成")

if __name__ == "__main__":
    main()
