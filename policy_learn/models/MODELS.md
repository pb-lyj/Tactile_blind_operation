Models:
1. 低维策略：
    NN: MLP
    Input: resultant_force[6] + resultant_moment[6]
    Output: delta_action_nextstep[3]
    Remarks: 作为无时序下界；建议 Huber/L2；隐藏层 256×3；输入做 z-score
2. 低维策略（时序）：
    NN: GRU
    Input: resultant_force[6] + resultant_moment[6] {t}
    Output: delta_action_nextstep[3]
    Remarks: GRU 隐层 256；因果；动作平滑正则（Δu 与 jerk）

3. feature-MLP:
    NN: MLP-BC（单帧）
    Input: tactile_feature[256]
    Output:
    Remarks: 隐藏层可用 512×3
4. feature-GRU:
    暂不做

5. ACT:
    NN: 标准 causal Transformer （轻量：GRU/TCN 前端 + Transformer）
    Input: tactile_feature[256] +（可选）{t}
    Output:
    Remarks: 历史 H=2.0s；分块长度 K=5–10 步；未来动作块 Δa[3]×K（推理时逐块滚动）
6. DP:
    NN: UNet
    Input: tactile_feature[256] +（可选）{t}
    Output: 
    Remarks: 历史 H=2.0s；预测视界 M=8–16 步；未来序列 Δa[3]×M（采样得到第一步或整段）

Comparasion：
1. 重建精度
    (cir + rect + tri).pt 重建 (cir + rect + tri) 动作
2. 泛化性
    2.1 重建泛化性：
        (cir + rect).pt 重建 (cir + rect)动作 & （tri）动作
        (rect + tri).pt 重建 (rect + tri)动作 & （cir）动作
    2.2 策略泛化性：
        (cir + rect).pt 操作 (cir + rect)动作 & （tri）动作
        (cir + rect + tri).pt 操作 USB插入
