import numpy as np


def is_rankings_valid(rankings: np.ndarray):
    if len(rankings.shape) != 2:
        return False
    if not ((rankings == (1 / rankings.T))).all():
        return False
    return True


def norming(ranking):
    size = ranking.shape[0]
    column_sums = np.ones((1, size)) @ ranking
    result = column_sums.copy()
    for i in range(size - 1):
        result = np.hstack((result, column_sums))
    return (ranking / result.reshape((size, size)) @ np.ones((size, 1))) / size


def ahp(alternatives_ranking: np.ndarray, criteria_ranking: np.ndarray):
    assert is_rankings_valid(criteria_ranking)
    assert alternatives_ranking.shape[0] == criteria_ranking.shape[0]
    size = alternatives_ranking.shape[0]
    union_weights = norming(alternatives_ranking[0])
    for i in range(1, size):
        union_weights = np.hstack((union_weights, norming(alternatives_ranking[i])))

    return np.array(union_weights) @ norming(criteria_ranking)


criteria_rankings = np.array([
    [1, 1, 1 / 2],
    [1, 1, 7],
    [2, 1 / 7, 1]
])

alternatives_rankings = np.array([
    [[1, 4, 1 / 2],
     [1 / 4, 1, 1 / 5],
     [2, 5, 1]],

    [[1, 1, 2],
     [1, 1, 3],
     [1 / 2, 1 / 3, 1]],

    [[1, 1 / 3, 4],
     [3, 1, 5],
     [1 / 4, 1 / 5, 1]]
])

if __name__ == '__main__':
    print(ahp(alternatives_rankings, criteria_rankings))
