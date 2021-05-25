import numpy as np


def two_group_ent(first, tot):
    return -(first / tot * np.log2(first / tot) +
             (tot - first) / tot * np.log2((tot - first) / tot))


if __name__ == "__main__":
    # L<17
    tot_ent = two_group_ent(10, 24)
    g17_ent = 15 / 24 * two_group_ent(11, 15) + 9 / 24 * two_group_ent(6, 9)
    answer = tot_ent - g17_ent
    print('L<17:', answer)

    # L < 20
    g17_ent = 7 / 24 * two_group_ent(6, 7) + 17 / 24 * two_group_ent(9, 17)
    answer = tot_ent - g17_ent
    print('L<20:', answer)

    # color - brown
    g17_ent = 18 / 24 * two_group_ent(6, 18) + 6 / 24 * two_group_ent(4, 6)
    answer = tot_ent - g17_ent
    print('color = brown:', answer)

