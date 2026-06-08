# QinleTorus DimRotation Sanity Reproduction

这个目录只复现 Qinle Torus SOTA 中的一个核心调度机制：

把每个节点的数据均匀划分为与网络维度数相同的多个 chunk，然后让不同 chunk 从不同维度开始通信，并按轮转顺序经过所有维度。

在 3D Torus 中，这个机制应当得到：

```text
chunk 1: X->Y->Z
chunk 2: Y->Z->X
chunk 3: Z->X->Y
```

这样第一阶段三个 chunk 会分别占用 X、Y、Z 三个维度链路，用于表达更充分的维度并行和带宽利用。

运行：

```bash
python3 SOTA/QinleTorus/scripts/dimrotation_3d_sanity.py
```

当前脚本是调度机制 sanity reproduction，不声称覆盖完整 3D PopNet 性能仿真或整篇 SOTA 的全部实验。
