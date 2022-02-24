import logging

import numpy as np


def is_rankings_valid(rankings: np.ndarray):
    if len(rankings.shape) != 2:
        return False
    if not (rankings == (1 / rankings.T)).all():
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

    union_weights = norming(alternatives_ranking[0])
    for i in range(1, alternatives_ranking.shape[0]):
        union_weights = np.hstack((union_weights, norming(alternatives_ranking[i])))
    logging.debug(union_weights)
    return np.array(union_weights) @ norming(criteria_ranking)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    criteria_rankings = np.array([[1,   1,      1 / 2],
                                  [1,   1,      7],
                                  [2,   1 / 7,  1]])

    alternatives_rankings = np.array([[[1,      4,      1 / 2],
                                       [1 / 4,  1,      1 / 5],
                                       [2,      5,      1]],

                                      [[1,      1,      2],
                                       [1,      1,      3],
                                       [1 / 2,  1 / 3,  1]],

                                      [[1,      1 / 3,  4],
                                       [3,      1,      5],
                                       [1 / 4,  1 / 5,  1]]])
    print(ahp(alternatives_rankings, criteria_rankings))

    phone_criteria = np.array([
        # display       camera      price   performance
            [1,             6,      9,          7],     # display
            [1 / 6,         1,      8,      1 / 3],     # camera
            [1 / 9,     1 / 8,      1,      1 / 3],     # price
            [1 / 7,         3,      3,          1]])    # performance

    phone_rankings = np.array([
        [  # iPhone 13     Poco F3     Galaxy A72
            # display
            [1,             3,          6],  # iPhone 13
            [1 / 3,         1,          4],  # Poco F3
            [1 / 6,         1 / 4,      1]   # Galaxy A72
        ],
        [   # camera
            [1,             9,          7],  # iPhone 13
            [1 / 9,         1,      1 / 3],  # Poco F3
            [1,             3,          1]  # Galaxy A72
        ],
        [  # price
            [1,             9,        1/6],  # iPhone 13
            [1 / 9,         1,          2],  # Poco F3
            [6,             1 / 2,      1]  # Galaxy A72
        ],
        [  # body
            [1,             3,          5],  # iPhone 13
            [1 / 3,         1,          3],  # Poco F3
            [1 / 5,         1 / 3,      1]  # Galaxy A72
        ]
    ])

    print(ahp(phone_rankings,phone_criteria))


