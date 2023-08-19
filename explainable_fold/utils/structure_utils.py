import numpy as np

def check_realignment(tags, seqs, alignment_dir, args):
    """
    TODO: check the quality of alignments and decide if re-compute alignments
    """
    return False


def compute_transformation(pos_1, pos_2):
    """
    https://github.com/scipy/scipy/blob/main/scipy/spatial/transform/_rotation.pyx#L2367
    https://en.wikipedia.org/wiki/Kabsch_algorithm
    """
    P = np.array(pos_1)
    Q = np.array(pos_2)
    
    P_mean = P.mean(0)
    Q_mean = Q.mean(0)

    H = ((Q - Q_mean)).T @ (P - P_mean)
    U, _, V_t = np.linalg.svd(H)
    # print(U, _, V_t)
    # ensure that R is in the right-hand coordinate system, very important!!!
    d = np.sign(np.linalg.det(U @ V_t))
    U[:, -1] = d * U[:, -1]
    R = U @ V_t
    t = Q_mean - R @ P_mean
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def transform(pos, T):
    R = T[:3, :3]
    t = T[:3, 3]
    return pos @ R.T + t


# def compute_tm_score(pos_1, pos_2):
#     """
#     https://github.com/Dapid/tmscoring/blob/master/tmscoring/tmscore.py
#     """
#     pos_1 = np.array(pos_1)
#     pos_2 = np.array(pos_2)
#     d0 = 1.24 * (len(pos_1) - 15) ** (1.0 / 3.0) - 1.8
#     dist = abs(pos_1 - pos_2).sum(axis=1)
#     sum = 0
#     for i in range(len(pos_1)):
#         sum += 1 / (1 + (dist[i] / d0) ** 2)
#     return sum / len(pos_1)


def compute_tm_score(pos_1, pos_2):
    """
    https://github.com/Dapid/tmscoring/blob/master/tmscoring/tmscore.py
    """
    d0 = 1.24 * (len(pos_1) - 15) ** (1.0 / 3.0) - 1.8
    dist = abs(pos_1 - pos_2).sum(axis=1)
    sum = 0
    for i in range(len(pos_1)):
        sum += 1 / (1 + (dist[i] / d0) ** 2)
    return sum / len(pos_1)
