Models:
1.直接学习：
    NN：GRU or CNN
    Input: resultant_force[3] + resultant_moment[3] + delta_action_now[3]?
    Output: delta_action_nextstep[3]

2.DP:
    NN: UNet or others
    Input: feature_vector[256] + delta_action_now[3]?
    Output: delta_action_nextstep[3]

3.
