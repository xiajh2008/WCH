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

    t = 6
    e = 5
    lamda = 1/6
    import WC_Hmeans
    center = WC_Hmeans.clustering(data, t, e, lamda)

    print(metrics.adjusted_rand_score(truth, center))