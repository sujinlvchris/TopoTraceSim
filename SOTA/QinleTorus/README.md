# QinleTorus

复现并评估 Qinle 等人提出的 **DimRotation multi-dimensional scheduling**：
针对 Torus 网络中的 All-to-All（A2A）通信，将每节点的数据按网络维度数 D
均匀划分为 D 个 chunk，并为不同 chunk 分配不同的维度通信顺序
（2D Torus 下 chunk 0 走 `X→Y`、chunk 1 走 `Y→X`），让不同 chunk 在同一时间
占用不同维度的链路，达成 conflict-free、full-coverage 的多维并行重叠。

对照模式是经典 **direct A2A**：每对 (src,dst) 一个完整 packet、统一注入、
PopNet TXY 路由自由竞争。

## 数据流

```
PyTorchSim (Docker)                       Python scheduler                 PopNet
┌────────────────────────┐  CSV          ┌─────────────────────┐  bench   ┌──────────────┐
│ per-chunk matmul       │ ─────────────▶│ direct.py /          │ ───────▶ │ popnet -R 1  │
│ → compute_done_cycle   │               │ dimrotation.py       │           │ TXY routing  │
└────────────────────────┘               └─────────────────────┘          └──────────────┘
                                                                                 │
                                                                                 ▼
                                                        ┌──────────────────────────────────┐
                                                        │ analysis/compute_metrics + plot  │
                                                        └──────────────────────────────────┘
```

## 目录

```text
SOTA/QinleTorus/
├── configs/           # torus_4x4.yaml / torus_8x8.yaml
├── pytorchsim/        # 在 torchsim-ci 容器内跑 per-chunk matmul，产 CSV
├── scheduler/         # topology / timing / direct / dimrotation
├── popnet_io/         # bench 写入 + popnet.log 解析
├── analysis/          # 指标 + 画图
├── scripts/           # run_one.sh / run_sweep.sh / Switch layer benchmark
├── traces/            # 中间 CSV / bench，默认不提交
├── output/            # popnet stdout & 日志，默认不提交
└── results/           # 每跑一份 JSON + sweep 汇总 PNG，默认不提交
```

## 在服务器上一键跑

服务器：`wyf@10.98.36.113:9370`，目标目录 `/mnt/sdb1/wyf/SOTA/QinleTorus/`。
依赖 `/mnt/sdb1/wyf/TopoTraceSim/third_party/popnet_anytopo/build/popnet` 已构建。

```bash
bash scripts/sync_to_server.sh        # 本地 → 服务器
ssh -p 9370 -i ~/.ssh/id_ecdsa wyf@10.98.36.113 \
    "cd /mnt/sdb1/wyf/SOTA/QinleTorus && bash scripts/run_sweep.sh"
```

或单跑一组：

```bash
bash scripts/run_one.sh --torus 4x4 --msg 16KB --scheduler dimrotation
bash scripts/run_one.sh --torus 4x4 --msg 16KB --scheduler direct
```

## Switch MoE Layer Benchmark

`scripts/run_switch_layer_benchmark.sh` 用 `google/switch-base-8` 的层形状
构建 layer-level trace：

- router projection：`tokens @ router_weight`，通过 PyTorchSim BackendSim 测 cycle；
- dispatch：把 source chiplet 的 token chunk 发到 expert 所在 chiplet；
- expert FFN：使用真实 Switch expert 形状 `768 -> 3072 -> 768` 的 BackendSim cycle；
- return/combine traffic：把 expert 输出返回原 source chiplet。

默认配置是 `2x2` Torus、4 个 compute chiplet、8 个 experts、每 chiplet 2 个
experts、每个 source 16 tokens。该脚本默认复用已经生成的 expert FFN CSV；如果
需要重跑 expert FFN，可加 `--rerun-expert`。

```bash
bash scripts/run_switch_layer_benchmark.sh \
    --torus 2x2 \
    --tokens-per-source 16 \
    --scheduler dimrotation \
    --hardware-config /mnt/sdb1/wyf/TopoTraceSim/configs/noi_hbm_reconfigurable.yaml
```

## 复现要点

- **PopNet 不改 C++**：DimRotation 在 trace 层把每个 chunk 拆成 D 个单维跳，
  每跳 src/dst 只差一个坐标，现有 TXY (`-R 1`) 天然只走那条维度的链路。
- **chunk 数 = D = 2**：与 2D Torus 维度一致；扩到 3D 时改 `--dims 3 --chunks 3`，
  但 PopNet 需要补 ND-DOR 路由。
- **inject_cycle 时间链**：chunk 第 `k` 跳的 inject 时间 = 第 `k-1` 跳的 inject
  时间 + `estimate_hop_cycles` 解析估计值，保证后跳基本在前跳出链路后进入。

更详细的设计见上一级 `TopoTraceSim/README.zh-CN.md` 与本目录 `analysis/`。
