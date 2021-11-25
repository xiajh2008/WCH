from sklearn import metrics
import numpy as np

np.set_printoptions(suppress=True)  # 精确表示小数
np.seterr(invalid='ignore')


if __name__ == '__main__':
    path = 'Data sets/'
    # file_name = 'Aggregation.txt'
    # file_name = 'CMC.txt'
    # file_name = 'Compound.txt'
    # file_name = 'D31.txt'
    # file_name = 'flame.txt'
    # file_name = 'jain.txt'
    file_name = 'pathbased.txt'
    # file_name = 'R15.txt'
    # file_name = 'spiral.txt'
    # file_name = 'rings.txt'

    data_ = np.loadtxt(path + file_name, delimiter=',')

    data = data_[:, :-1]
    truth = data_[:, -1]
    del data_

    t = 7
    e = 12
    lamda = 0.5
    import WC_HGMM
    center = WC_HGMM.clustering(data, t, e, lamda)

    print(metrics.adjusted_rand_score(truth, center))
    print(metrics.fowlkes_mallows_score(truth, center))
