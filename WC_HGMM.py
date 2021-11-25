import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import copy
import math
import warnings
import itertools

import sklearn.metrics
from sklearn import metrics
import warnings
import matplotlib.patches as mpathes
from sklearn import preprocessing
import time
# import vision_WC

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)  # 精确表示小数
np.seterr(invalid='ignore')


def read_dataset(data):  # 用于s1,s2,s3,s4文件数据的读取
    label_n = list()
    pos = dict()
    node_dict = dict()
    data_size = data.shape[0]
    for i in range(data_size):
        label_n.append(i)
        node_dict[i] = len(node_dict)
        pos[i] = data[i].tolist()  # 用于可视化的字典

    return label_n, pos, node_dict, data_size


def large_pknn_calculation(data, label_n, k):  # 计算节点之间的欧氏距离
    """根据每个样本的一个较大值的近邻，这里设置为k， 主要用于后期的算法测试（循环）,避免反复计算近邻造成的高时间成本"""
    pknn, pknd = [], []
    data_size = data.shape[0]
    for i in range(data_size):
        temp = np.linalg.norm(data - data[i], axis=1, keepdims=True).reshape(1, -1)
        simi_list = temp[0]
        simi_sorted = np.argsort(simi_list)  # 按欧氏距离升序排序
        kns0 = [label_n[simi_sorted[j]] for j in range(k + 1)]
        kvs0 = [simi_list[simi_sorted[j]] for j in range(k + 1)]

        for m in range(k + 1):
            if kns0[m] == i:
                kns0.pop(m)
                kvs0.pop(m)
                break
        pknn.append(kns0)
        pknd.append(kvs0)

    large_pknn = np.array(pknn)
    large_pknd = np.array(pknd)

    return large_pknn, large_pknd


def rnn_calculation(data_size, t, ptnn):
    """计算每个样本的 反向t近邻 状态 1 or 0 """
    rnn_t = [list() for i in range(data_size)]  # 记录每个目标的自然近邻 (不是数量)
    rnn_edge = []  # 记录反向近邻的无向边集
    in_degree = [0] * data_size

    nn_t = [list() for i in range(data_size)]  # 记录每个目标的自然近邻 (不是数量)
    nn_cnt_t = [0] * data_size  # 记录每个目标的自然近邻的数量
    nn_edge_t = []  # 自然邻居图的边集合

    for i in range(data_size):
        for j in range(t):
            jj = ptnn[i][j]
            in_degree[jj] += 1

            if i in ptnn[jj]:
                """更新反向近邻"""
                rnn_t[i].append(True)
                rnn_edge.append((i, jj))

                """更新自然近邻"""
                if not (jj in nn_t[i]):
                    nn_t[i].append(jj)
                    nn_cnt_t[i] += 1
                    nn_edge_t.append((i, jj))  # ---添加自然邻居图的自然边
                """更新自然近邻"""
                if not (i in nn_t[jj]):
                    nn_t[jj].append(i)
                    nn_cnt_t[jj] += 1
            else:
                rnn_t[i].append(False)

    # print(f'最大入度：{max(in_degree)}')
    in_degree = preprocessing.minmax_scale(in_degree)
    return rnn_t, in_degree, nn_t, nn_cnt_t, nn_edge_t


def height_nn(data, ptnn, ptnd, t, rnn_t, label_n):
    """方法1"""
    # data_size = data.shape[0]
    # alpha_nn = [-1 for i in range(data_size)]
    # weight = np.exp(ptnd) / np.sum(np.exp(ptnd), axis=1, keepdims=True)  # 大小为： 点数 × KNN
    # for i in range(data_size):
    #     temp = []
    #     for x in range(t):
    #         j = ptnn[i][x]
    #         if rnn_t[j][x]:
    #             temp.append(ptnd[i][x])
    #         else:
    #             temp.append(ptnd[i][x] * 2)
    #     # alpha_nn[i] = temp / t
    #     alpha_nn[i] = np.dot(weight[i], np.array(temp))
    """********************方法2结束*********************"""

    """方法2"""
    data_size = data.shape[0]
    """ 这个函数用于计算每个数据点的海拔或高度，使用了基于相互近邻的点级系数和基于共享近邻的领域，融入了基于tnn的区域级系数"""
    ptnd = np.array(ptnd)  # 将二维列表进行数组化
    # """权重权重＊＊＊＊＊＊＊＊＊＊＊＊＊＊"""
    """不能显示，由于权值较大，导致计算机上界溢出为 nan"""
    # weight = np.exp(pnv_nd) / np.sum(np.exp(pnv_nd), axis=1, keepdims=True)  # 大小为： 点数 × KNN

    """此权重的计算方式能够显示"""
    divergence = np.sum(ptnd, axis=1, keepdims=True)
    weight = ptnd / divergence  # 大小为： 点数 × KNN

    alpha_nn = list()
    for i in range(data_size):  # 读取所有节点中的一个节点
        # """计算 xi与xj互为近邻，且共享近邻数量大于等于: 1/2 """
        height = []
        snn_temp = list()

        for j in range(t):
            xj = ptnn[i][j]

            shares = list(set(ptnn[i]).intersection(ptnn[xj]))

            snn_temp.append(len(shares))

            if xj in rnn_t[i]:
                l1 = len(shares)
                if l1 > 0:
                    sum0 = 0
                    for xk in shares:
                        sum0 += np.linalg.norm(data[i] - data[xk]) + np.linalg.norm(data[xj] - data[xk])  # dist(Ai-Xi)+dist(Aj-Xi)
                    height.append(sum0 / l1)  # 节点Xi与KNN中的某一个节点之间的共享近邻距离和/共享近邻点数，即共享近邻平均距离
                else:
                    height.append(2 * np.linalg.norm(data[i] - data[xj]))  # 用 dist(Ai-Aj)  # 来构成一个更长的距离，说明该点距离Xi更远
            else:
                height.append(2 * np.linalg.norm(data[i] - data[xj]))  # 用 dist(Ai-Aj)  # 来构成一个更长的距离，说明该点距离Xi更远

        alpha_nn.append(np.dot(weight[i], np.array(height)))
    """********************方法2结束*********************"""

    mean_alpha_nn = []
    median_alpha_nn = []
    std_alpha_nn = []

    for i in range(data_size):
        tt = [alpha_nn[i]]
        tt.extend([alpha_nn[x] for x in ptnn[i]])
        median_alpha_nn.append(np.median(np.array(tt)))
        mean_alpha_nn.append(np.mean(np.array(tt)))
        std_alpha_nn.append(np.std(np.array(tt), ddof=0))

    pc_nn = []
    for i in range(data_size):
        temp = []
        for j in rnn_t[i]:
            if abs(alpha_nn[i] - median_alpha_nn[j]) <= 2 * std_alpha_nn[j] and abs(alpha_nn[j] - median_alpha_nn[i] <= 2 * std_alpha_nn[i]):
                temp.append(j)
        pc_nn.append(temp)

    node_and_altitude_ = list(zip(label_n, alpha_nn))
    node_and_altitude = sorted(node_and_altitude_, key=lambda x: x[1], reverse=False)
    points_i = [item[0] for item in node_and_altitude]  # 临时保存排序后的节点列表

    return alpha_nn, mean_alpha_nn, median_alpha_nn, std_alpha_nn, points_i, pc_nn


def specified_natural_neighbors(data_size, ptnn, t):
    nn_t = [list() for i in range(data_size)]  # 记录每个目标的自然近邻 (不是数量)

    nn_increase_ratio = [0 for i in range(data_size)]
    nn_cnt_t = [0] * data_size  # 记录每个目标的自然近邻的数量
    nn_edge_t = []  # 自然邻居图的边集合

    for j in range(t):  # 控制邻居数量的增长
        for i in range(data_size):
            """---i的每个近邻xj是不是i的自然邻居"""
            xj = ptnn[i][j]
            xj_nn_t = ptnn[xj][:j+1]

            if i in xj_nn_t:  # 自然邻居条件成立
                if not (xj in nn_t[i]):
                    nn_t[i].append(xj)
                    nn_cnt_t[i] += 1
                    nn_edge_t.append((i, xj))  # ---添加自然邻居图的自然边

                if not (i in nn_t[xj]):
                    nn_t[xj].append(i)
                    nn_cnt_t[xj] += 1

        for i in range(data_size):
            if nn_cnt_t[i] == j:
                nn_increase_ratio[i] += 1

    return nn_t, nn_cnt_t, nn_edge_t


def hubness_score(data_size, ptnn, ptnd, t, nn_t, nn_cnt_t, in_degree):
    """计算每个数据点的局部密度（基于平均散度和t近邻的密度）-------------------"""
    # tmax = np.max(np.array(nn_cnt_t))
    local_density = []
    for i in range(data_size):
        densi = t / np.mean(np.array(ptnd[i]))  # 每个数据点的局部密度。分布越密集，该值越大
        nn_ratio = nn_cnt_t[i] / (t * np.mean(np.array(ptnd[i])))  # 分布越密集，该值越大

        """数据点i 的 基于自然邻居的 局部密度"""
        # tt = densi * nn_ratio
        tt = densi * nn_ratio * in_degree[i]

        local_density.append(tt)

    return local_density


def box_plot_single_outlier(nums):
    """
    计算单个数据点的基于箱盒法的离群性------------------------如果数据点 i 在自己的领域内为海拔离群，则 i 必然为离群点.
    被检测的变量 放置在 nums[0], 检验num[0]在nums数列中是否为离群值，这里只检测(海拔的)上离群点
    """
    a = np.percentile(nums, (25, 75))
    iqr = a[1] - a[0]
    upper_bound = a[1] + 1.5 * iqr
    # lower_bound = a[0] - 1.5 * iqr

    if nums[0] > upper_bound:
        single_bp = True
    else:
        single_bp = False

    return single_bp


def box_plot_based_nnoutlier(nums):
    """除nums[0]以外，计算数据点的多个近邻的基于箱盒法的离群性"""
    a = np.percentile(nums, (25, 75))
    iqr = a[1] - a[0]
    upper_bound = a[1] + 1.5 * iqr
    # lower_bound = a[0] - 1.5 * iqr

    multi_bp = []
    for x in nums[1:]:
        if x > upper_bound:
            multi_bp.append(True)  # 异常性
        else:
            multi_bp.append(False)  # 非异常性
    return multi_bp


def stability_via_nn_pc_and_bp(data, nv, ptnn, nn, alpha_nn, median_alpha_nn, mean_alpha_nn, std_alpha_nn, lamda):
    """粗化处理： 输出异常点（不同类型，但性质相同）； 输出糙化结果(结合各种弱化方法)"""
    data_size = data.shape[0]

    """计算两个异常情况------------------------------------"""
    anomaly_bp, anomaly_pc = [], []
    single_bp, multi_bp = [], []
    single_pc, multi_pc = [], []

    for i in range(data_size):
        nums = [alpha_nn[i]]
        nums.extend([alpha_nn[j] for j in ptnn[i]])

        """箱盒法的弱化"""
        if box_plot_single_outlier(nums):  # 判断 i 是否为自己领域内的异常
            anomaly_bp.append(i)
            single_bp.append(True)  # 异常性
        else:
            single_bp.append(False)  # 非异常性
        """箱盒法的近邻异常"""
        multi_bp.append(box_plot_based_nnoutlier(nums))

        if abs(alpha_nn[i] - median_alpha_nn[i]) > 2 * std_alpha_nn[i]:  # 取中值
            anomaly_pc.append(i)
            single_pc.append(True)  # 异常性
        else:
            single_pc.append(False)  # 非异常性
        """拉依达法则的近邻异常"""
        flag = []
        for j in ptnn[i]:
            """用于计算稳定性或糙化"""
            if abs(alpha_nn[j] - mean_alpha_nn[i]) > 2 * std_alpha_nn[i] or abs(alpha_nn[i] - mean_alpha_nn[j]) > 2 * std_alpha_nn[j]:
            # if not (abs(alpha_nn[j] - median_alpha_nn[i]) <= 2 * std_alpha_nn[i] and abs(alpha_nn[i] - median_alpha_nn[j]) <= 2 * std_alpha_nn[j]):
                flag.append(True)
            else:
                flag.append(False)

        multi_pc.append(flag)
    """计算糙化因子--------------------------"""
    cf = [-1 for i in range(data_size)]  # 粗糙化因子值
    cps = []  # 粗糙点集
    cf_list = []
    coarsened_status = [False for i in range(data_size)]
    strengthed_points = [list() for x in range(data_size)]

    stable_c1, unstable_c1 = [], []  # 用于后面的数据点的飘移

    for i in range(data_size):
        temp_cf = []
        for j in range(nv):
            if ptnn[i][j] in nn[i] and not (multi_bp[i][j]) and not (multi_pc[i][j]):
                temp_cf.append(1)
                strengthed_points[i].append(ptnn[i][j])
            else:
                temp_cf.append(0)

        cf_list.append(temp_cf)
        cf[i] = np.mean(np.array(temp_cf))

        if cf[i] <= lamda:
            cps.append(i)
            coarsened_status[i] = True
            unstable_c1.append(i)
        else:
            stable_c1.append(i)

    return cf, cps, coarsened_status, single_bp, multi_bp, single_pc, multi_pc, anomaly_bp, anomaly_pc, cf_list, strengthed_points, stable_c1, unstable_c1



def nan_of_nan(i, coarsened_status, nn_t):
    nodesfromnaturalneighbors = []
    for x in nn_t[i]:
        if not coarsened_status[x]:
            nodesfromnaturalneighbors.append(x)
            for x1 in nn_t[x]:
                if not coarsened_status[x1]:
                    nodesfromnaturalneighbors.append(x1)
    nodesfromnaturalneighbors = list(nodesfromnaturalneighbors)
    return nodesfromnaturalneighbors


def hubness_points(data,nn_t, coarsened_status, local_density, stable_c1, ptnn, t_nan, label_n):
    # porder = list(zip(stable_c1, [local_density[x] for x in stable_c1]))
    # porder = sorted(porder, key=lambda x: x[1], reverse=True)  # 按密度的降序排列稳定数据点
    # porder = [x[0] for x in porder]  # 临时保存排序后的节点列表

    hubs = []  # 局部区域内密度最大点，
    hubs_vector = []
    for i in stable_c1:
        """标准1： 局部密度最大"""
        if len(nn_t[i]) > 0:  # 由于lamda的取值存在负数，所以某些数据点可能出现个0个自然邻居，也就是包含0个自然邻居的数据点也不会被识别为不稳定点。
            # nodesfromnaturalneighbors = nan_of_nan(i, coarsened_status, nn_t)
            # nodesfromnaturalneighbors = ptnn[i]
            nodesfromnaturalneighbors = t_nan[i]
            """如果下面条件成立，表示i可以作为枢纽点，因为枢纽得分为局部最高的点"""
            if len(nodesfromnaturalneighbors) > 0:  # lamda过大，可能导致部分数据点虽然有自然邻居，但是由于不稳定点的影响，使得自然近邻的自然近邻为空，即nodesfromnaturalneighbors等于空集
                if local_density[i] >= max([local_density[x] for x in nodesfromnaturalneighbors]):  # 以局部区域内密度差的局部最大的点 为 局部中心 local center
                    if len(set(nodesfromnaturalneighbors).intersection(hubs)) == 0:  # 避免相同点拥有相同枢纽度（得分）都被选择作为枢纽点
                        hubs.append(i)
                        hubs_vector.append(data[i].tolist())

    if len(hubs) <= 1:
        for i in stable_c1:
            """标准1： 局部密度最大"""
            if len(nn_t[i]) > 0:  # 由于lamda的取值存在负数，所以某些数据点可能出现个0个自然邻居，也就是包含0个自然邻居的数据点也不会被识别为不稳定点。
                # nodesfromnaturalneighbors = nan_of_nan(i, coarsened_status, nn_t)
                # nodesfromnaturalneighbors = ptnn[i]
                nodesfromnaturalneighbors = t_nan[i]
                """如果下面条件成立，表示i可以作为枢纽点，因为枢纽得分为局部最高的点"""
                if len(nodesfromnaturalneighbors) > 0:  # lamda过大，可能导致部分数据点虽然有自然邻居，但是由于不稳定点的影响，使得自然近邻的自然近邻为空，即nodesfromnaturalneighbors等于空集
                    if local_density[i] >= max([local_density[x] for x in nodesfromnaturalneighbors if not coarsened_status[x]]):  # 以局部区域内密度差的局部最大的点 为 局部中心 local center
                        if len(set(nodesfromnaturalneighbors).intersection(hubs)) == 0:  # 避免相同点拥有相同枢纽度（得分）都被选择作为枢纽点
                            hubs.append(i)
                            hubs_vector.append(data[i].tolist())

    if len(hubs) <= 1:
        for i in label_n:
            """标准1： 局部密度最大"""
            if len(nn_t[i]) > 0:  # 由于lamda的取值存在负数，所以某些数据点可能出现个0个自然邻居，也就是包含0个自然邻居的数据点也不会被识别为不稳定点。
                # nodesfromnaturalneighbors = nan_of_nan(i, coarsened_status, nn_t)
                # nodesfromnaturalneighbors = ptnn[i]
                nodesfromnaturalneighbors = t_nan[i]
                """如果下面条件成立，表示i可以作为枢纽点，因为枢纽得分为局部最高的点"""
                if len(nodesfromnaturalneighbors) > 0:  # lamda过大，可能导致部分数据点虽然有自然邻居，但是由于不稳定点的影响，使得自然近邻的自然近邻为空，即nodesfromnaturalneighbors等于空集
                    if local_density[i] >= max([local_density[x] for x in nodesfromnaturalneighbors]):  # 以局部区域内密度差的局部最大的点 为 局部中心 local center
                        if len(set(nodesfromnaturalneighbors).intersection(hubs)) == 0:  # 避免相同点拥有相同枢纽度（得分）都被选择作为枢纽点
                            hubs.append(i)
                            hubs_vector.append(data[i].tolist())

    return hubs, hubs_vector


def Gaussian_Mixture_Model(data, num):
    from sklearn.mixture import GaussianMixture as GMM
    gmm_new = GMM(n_components=num, covariance_type='full', random_state=0)
    labels_gmm = gmm_new.fit(data).predict(data)

    center_set = list(set(labels_gmm))
    clusters_gmm = [list() for i in range(len(center_set))]

    for i in range(data.shape[0]):
        clusters_gmm[center_set.index(labels_gmm[i])].append(i)
    # print(f'\nclusters_gmm: {clusters_gmm}\n')

    return labels_gmm, clusters_gmm


def cluster_gmm_centers(data, clusters_gmm):
    final_centers_gmm = []
    for pts in clusters_gmm:
        final_centers_gmm.append(np.mean([data[x] for x in pts], axis=0))
    return final_centers_gmm


def connectivity_component(data, nn_t, clusters_kmeans, final_centers_kmeans):
    cluster_nn = []
    final_centers_kmeans_nn = []  # 后面用于直接接在 cluster_nn 的后面

    cluster_size = len(clusters_kmeans)

    for i in range(cluster_size):
        cluster = copy.copy(clusters_kmeans[i])  # 取k-means的一个簇（类）
        # cluster = copy.copy(clusters_kmeans[3])  # 取k-means的一个簇（类）
        # print(f'\n cluster 长度： {cluster}, {len(cluster)}')
        sub_cluster = [cluster[0]]  # 取簇的第一个点, 一个类中至少有一人数据点 或顶点
        # print(f'sub_cluster: {sub_cluster}')

        processing_list = [cluster[0]]  # 当前处理列表，并用于更新
        # print(f'当前处理数据点集: {processing_list}')
        cluster.pop(0)

        while len(cluster) > 0:
            nodes = set()
            for j in processing_list:
                nodes = nodes.union(set(nn_t[j]).intersection(cluster))
            ts = set(nodes).intersection(cluster)
            ts = list(ts.difference(sub_cluster))

            if len(ts) == 0:  # 当前处理数据集点与类（簇）内的其他点没有自然交集，则停止，生成一个子类
                """添加一个新的子类"""
                cluster_nn.append(list(set(sub_cluster)))
                """添加一个新的中心"""
                final_centers_kmeans_nn.append(np.mean(np.array([data[x] for x in sub_cluster]), axis=0))
                # print(f'new cluster: {list(set(sub_cluster))}')
                """---更新临时变量---"""
                sub_cluster = [cluster[0]]
                processing_list = [cluster[0]]
                cluster = list(set(cluster).difference(processing_list))
            else:
                """---更新临时变量---"""
                sub_cluster.extend(ts)
                processing_list = ts
                cluster = list(set(cluster).difference(ts))

        t_subclus = list(set(sub_cluster))
        cluster_nn.append(t_subclus)

        if len(t_subclus) == len(clusters_kmeans[i]):
            """直接使用原始中心点向量添加"""
            final_centers_kmeans_nn.append(final_centers_kmeans[i])
        else:
            """添加一个新的中心点向量"""
            final_centers_kmeans_nn.append(np.mean(np.array([data[x] for x in sub_cluster]), axis=0))

    return cluster_nn, final_centers_kmeans_nn


def e_natural_neighbors(data, nn_t, rnn_t, ptnn, ptnd, coarsened_status, e, t):
    data_size = data.shape[0]

    k = max(e, t)
    k_nan = [list() for i in range(data_size)]

    pe_nan = [list() for i in range(data_size)]  # 用于合并阶段的扩展度
    t_nan = [list() for i in range(data_size)]  # 计算(合并阶段)海拔的均值和方差

    for i in range(data_size):
        if len(nn_t[i]) == 0:
            t_nan[i] = [i]
            t_nan[i].extend(ptnn[i])
            # print(f'{i}, t_nan: {[y for y in t_nan[i]]}, 不稳定点')
            continue

        temp_e_nan = [(i, 0)]
        if len(nn_t[i]) > 0:

            process_list = [(i, 0)]  # 迭代处理列表

            while len(process_list) > 0 and len(temp_e_nan) <= 2*k + 1:  # 包含数据点i自己
                new_process_list = []

                for l in range(len(process_list)):
                    x = process_list[l][0]
                    x_dist = process_list[l][1]

                    for j in range(t):
                        if rnn_t[x][j] and not coarsened_status[ptnn[x][j]]:
                            x1 = ptnn[x][j]
                            x1_dist = ptnd[x][j]

                            if not (x1 in [p[0] for p in temp_e_nan]):
                                new_process_list.append((x1, x1_dist+x_dist))  # 记录扩展节点列表
                                temp_e_nan.append((x1, x1_dist+x_dist))
                process_list = new_process_list

            points = sorted(temp_e_nan, key=lambda x: x[1], reverse=False)
            points = [item[0] for item in points]  # 临时保存排序后的节点列表
            k_nan[i] = points[:k + 1]

            pe_nan[i] = k_nan[i][1:e + 1]
            t_nan[i] = k_nan[i][:t + 1]

            """下面进行迭代自然近邻的选择处理"""
            if len(t_nan[i]) < t + 1:
                for x in ptnn[i]:
                    if not (x in t_nan[i]) and len(t_nan[i]) <= t + 1:
                        t_nan[i].append(x)
    return pe_nan, t_nan


def nan_based_mean_std_alpha(data, t_nan, alpha_nn):
    """根据自然近邻扩散，计算每个数据点的t个自然近邻,"""
    data_size = data.shape[0]
    nn_mean_alpha = []
    nn_median_alpha = []
    nn_std_alpha = []

    for i in range(data_size):
        la = [alpha_nn[q] for q in t_nan[i]]
        nn_mean_alpha.append(np.mean(la))
        nn_median_alpha.append(np.median(la))
        nn_std_alpha.append(np.std(la, ddof=0))
    return nn_mean_alpha, nn_std_alpha, nn_median_alpha


def basin_core_interval_pauta_criterion(i, center, cluster_nn, basin_stable_points, basin_instable_points, alpha_nn, nn_mean_alpha, nn_std_alpha, cf, data, final_centers_kmeans_nn, core_node, basin_core_threshold, basin_center_point, lamda):
    """计算子类i的相关信息"""
    for j in cluster_nn[i]:  # 取一个类 的一个节点
        if cf[j] <= lamda:
            basin_instable_points[i].append(j)  # 记录每个集水盆的不稳定点
        else:
            basin_stable_points[i].append(j)

    if len(basin_stable_points[i]) > 0:
        """计算子类i的中心点basin_core_threshold[i]"""
        y = final_centers_kmeans_nn[i]
        distance = []
        for x in basin_stable_points[i]:
            dd = np.linalg.norm(y - data[x])
            distance.append((x, dd))
        dd_order = sorted(distance, key=lambda x: x[1], reverse=False)
        bp = dd_order[0][0]

        bmp = np.median([alpha_nn[x] for x in basin_stable_points[i]])
        basin_core_threshold[i] = bmp
        basin_center_point[i] = bp  # 离kmeans中心点最近的点作为子类的中心点

        """计算子类的核心点"""
        for x in basin_stable_points[i]:
            if alpha_nn[x] - nn_mean_alpha[bp] <= 2 * nn_std_alpha[bp]:
                core_node[x] = True
            if alpha_nn[x] <= bmp:
                core_node[x] = True

    else:
        """异常子类只有中心点，没有阈值"""
        basin_center_point[i] = basin_instable_points[i][np.argmin([alpha_nn[y1] for y1 in basin_instable_points[i]])]

    for x1 in cluster_nn[i]:
        center[x1] = basin_center_point[i]

    return core_node, basin_center_point, basin_core_threshold, basin_stable_points, basin_instable_points


def basin_labelizing(data, alpha_nn, cluster_nn, final_centers_kmeans_nn, cf, lamda, t, nn_mean_alpha, nn_std_alpha, nn_median_alpha):
    """-----------  根据 k-means 获得的类来获得集水盆， 盆点标签  -----------"""
    center = [-1 for i in range(data.shape[0])]
    csize = len(cluster_nn)

    basin_dict = dict()  # 集水盆词典
    basin_instable_points = [list() for i in range(csize)]
    basin_stable_points = [list() for i in range(csize)]

    abnormal_clusters = []
    abnormal_centers = []

    basin_core_threshold = [-1 for i in range(csize)]

    core_node = [False for i in range(data.shape[0])]
    basin_center_point = [-1 for i in range(csize)]

    for i in range(len(cluster_nn)):
        core_node, basin_center_point, basin_core_threshold, basin_stable_points, basin_instable_points = basin_core_interval_pauta_criterion(i, center, cluster_nn, basin_stable_points, basin_instable_points, alpha_nn, nn_mean_alpha, nn_std_alpha, cf, data, final_centers_kmeans_nn, core_node, basin_core_threshold, basin_center_point, lamda)

        basin_dict[basin_center_point[i]] = len(basin_dict)

        """这是一种特殊情况"""
        if max([cf[x] for x in cluster_nn[i]]) <= lamda:
            abnormal_clusters.append(cluster_nn[i])
            abnormal_centers.append(basin_center_point[i])
            """集水盆的盆点为不稳定点，则其他盆内点即使不是不稳定点，也视为不稳定点"""
            basin_instable_points[i].extend(basin_stable_points[i])
            basin_stable_points[i] = []

    """改变量名, 便于后面的计算"""
    basin_points = basin_stable_points

    return center, basin_dict, basin_points, abnormal_centers, abnormal_clusters, basin_stable_points, basin_instable_points, core_node, basin_core_threshold, basin_center_point


def detecting_watershed_basin(data, pe_nan, e, ptnn, center, abnormal_centers, basin_dict, abnormal_clusters, alpha_nn, coarsened_status, nn_mean_alpha, nn_std_alpha, sigma, basin_stable_points, basin_instable_points, core_node, basin_core_threshold, basin_center_point):
    data_size = data.shape[0]
    if e > 0:
        """---按数据点的海拔排序所有数据点---"""
        node_altitude_ = list(zip([i for i in range(data_size)], alpha_nn))
        node_altitude_ = sorted(node_altitude_, key=lambda x: x[1], reverse=False)
        points = [item[0] for item in node_altitude_]  # 临时保存排序后的节点列表

        # print(f'未合并之前的异常簇中心： {abnormal_centers}')
        # print(f'未合并之前的异常簇： {abnormal_clusters}')

        """step 1 利用 近邻penn 获得聚类"""
        for i in points:
            if not coarsened_status[i]:
                for j in pe_nan[i]:
                    if center[j] != center[i] and not coarsened_status[j]:
                        if alpha_nn[i] - nn_mean_alpha[j] <= sigma * nn_std_alpha[j] and alpha_nn[j] - nn_mean_alpha[i] <= sigma * nn_std_alpha[i] and len(set(pe_nan[i]).intersection(pe_nan[j]))/e >= 0.4:

                            if core_node[i] or core_node[j]:
                                l1 = [alpha_nn[center[i]], alpha_nn[center[j]]]
                                l2 = [center[i], center[j]]

                                master = l2[np.argmin(l1)]
                                child = l2[np.argmax(l1)]

                                if alpha_nn[i] < alpha_nn[j] and not core_node[i] and alpha_nn[i] < alpha_nn[center[j]]:
                                    # print(f'child: {child}   master: {master}  -----------1111××××××××××××××××××××××××××××××××')
                                    """去除这个异常子类的核心点"""
                                    basin_core_threshold[basin_dict[center[j]]] = 0
                                    for p in basin_stable_points[basin_dict[center[j]]]:  # 集水盆的稳定点集
                                        core_node[p] = False
                                    for p in basin_instable_points[basin_dict[center[j]]]:  # 集水盆的不稳定点集
                                        core_node[p] = False
                                elif alpha_nn[j] < alpha_nn[i] and not core_node[j] and alpha_nn[j] < alpha_nn[center[i]]:
                                    # print(f'child: {child}   master: {master}  -----------2222××××××××××××××××××××××××××××××××')
                                    basin_core_threshold[basin_dict[center[i]]] = 0
                                    for p in basin_stable_points[basin_dict[center[i]]]:  # 集水盆的稳定点集
                                        core_node[p] = False
                                    for p in basin_instable_points[basin_dict[center[i]]]:  # 集水盆的不稳定点集
                                        core_node[p] = False

                                basin_stable_points[basin_dict[master]].extend(basin_stable_points[basin_dict[child]])
                                basin_instable_points[basin_dict[master]].extend(basin_instable_points[basin_dict[child]])

                                for p in basin_stable_points[basin_dict[child]]:  # 集水盆的稳定点集
                                    center[p] = master
                                for p in basin_instable_points[basin_dict[child]]:  # 集水盆的不稳定点集
                                    center[p] = master

                                basin_stable_points[basin_dict[child]] = []
                                basin_instable_points[basin_dict[child]] = []

        """step 2 异常子类（集水盆）的合并 """
        # print(f'异常子类（集水盆）的中心: {abnormal_centers}')
        # print(f'异常子类（集水盆）: {abnormal_clusters}\n')

        for pts in abnormal_clusters:
            ppts = list(zip([i for i in pts], [alpha_nn[j] for j in pts]))
            ppts = sorted(ppts, key=lambda x: x[1], reverse=False)
            ppts = [item[0] for item in ppts]  # 临时保存排序后的节点列表

            temp_center = center[ppts[0]]  # 记录原始盆点标签
            for x in ppts:
                for j in pe_nan[x]:
                    if not(center[j] in abnormal_centers) and not coarsened_status[j]:
                        for y in ppts:
                            center[y] = center[j]
                        break

            if center[ppts[0]] == temp_center:  # 从自然邻居或相互近邻的角度，未获得合并，
                for x in ppts:
                    for j in ptnn[x]:
                        if not (center[j] in abnormal_centers):
                            for y in ppts:
                                center[y] = center[j]
                            break

            abnormal_centers.remove(temp_center)  # 去除已经合并的异常集水盆的盆点

    return center


"""************************************************************************************************************************************"""


def clustering(data, t, e, lamda):
    assert t > 1
    assert e >= 0
    sigma = 2
    """------------------------------------------------clustering---------------------------------------------------------------------"""
    k = max(t, e)
    label_n, pos, node_dict, data_size = read_dataset(data)
    large_pknn, large_pknd = large_pknn_calculation(data, label_n, k)

    ptnn = large_pknn[:, :t]
    ptnd = large_pknd[:, :t]

    # ××××××××××××××××××××××××××"""在t近邻范围内获得每个数据点的自然邻居---×××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××"""
    rnn_t, in_degree, nn_t, nn_cnt_t, nn_edge_t = rnn_calculation(data_size, t, ptnn)

    alpha_nn, mean_alpha_nn, median_alpha_nn, std_alpha_nn, points_i, pc_nn = height_nn(data, ptnn, ptnd, t, rnn_t, label_n)

    local_density = hubness_score(data_size, ptnn, ptnd, t, nn_t, nn_cnt_t, in_degree)

    # ×××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    """稳定性方法1"""
    cf, cps, coarsened_status, single_bp, multi_bp, single_pc, multi_pc, anomaly_bp, anomaly_pc, cf_list, strengthed_points, stable_c1, unstable_c1 = stability_via_nn_pc_and_bp(data, t, ptnn, nn_t, alpha_nn, median_alpha_nn, mean_alpha_nn, std_alpha_nn, lamda)

    pe_nan, t_nan = e_natural_neighbors(data, nn_t, rnn_t, ptnn, ptnd, coarsened_status, e, t)

    """局部代表点"""
    hubs, hubs_vector = hubness_points(data,nn_t, coarsened_status, local_density, stable_c1, ptnn, t_nan, label_n)

    labels_gmm, clusters_gmm = Gaussian_Mixture_Model(data=data, num=len(hubs))
    final_centers_gmm = cluster_gmm_centers(data, clusters_gmm)

    """如果e==0，不计算每个潜在类的连通分量"""
    if e > 0:
        cluster_nn, final_centers_kmeans_nn = connectivity_component(data, nn_t, clusters_gmm, final_centers_gmm)
    else:
        cluster_nn = clusters_gmm
        final_centers_kmeans_nn = final_centers_gmm

    nn_mean_alpha, nn_std_alpha, nn_median_alpha = nan_based_mean_std_alpha(data, t_nan, alpha_nn)

    center, basin_dict, basin_points, abnormal_centers, abnormal_clusters, basin_stable_points, basin_instable_points, core_node, basin_core_threshold, basin_center_point = basin_labelizing(data, alpha_nn, cluster_nn, final_centers_kmeans_nn, cf, lamda, t, nn_mean_alpha, nn_std_alpha, nn_median_alpha)

    """   集水盆的发现和聚类   """
    center = detecting_watershed_basin(data, pe_nan, e, ptnn, center, abnormal_centers, basin_dict, abnormal_clusters, alpha_nn, coarsened_status, nn_mean_alpha, nn_std_alpha, sigma, basin_stable_points, basin_instable_points, core_node, basin_core_threshold, basin_center_point)

    return center

