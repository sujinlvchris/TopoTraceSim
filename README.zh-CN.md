# TopoTraceSim

**TopoTraceSim** 将 [PyTorchSim](https://github.com/PSAL-POSTECH/PyTorchSim)（NPU 计算仿真）与 [popnet_anytopo](https://github.com/FCAS-SCUT/popnet_anytopo)（NoC 网络仿真）通过最小 **All-to-All（A2A）** trace 流程串联起来。

当前阶段（Phase 1）：

- 通过 `configs/noi_hbm_reconfigurable.yaml` 支持 NoI/HBM 硬件配置
- 支持配置 compute-side NoI ports、HBM-side NoI ports、NoI-NI、D2D 和 HBM-side 参数
- 12 条有向通信（每节点向其余 3 个节点各发 16 KB）
- 每条流走完整 PyTorchSim：`torch.compile` → Gem5 → Spike → BackendSim/TOGSim
- 带 NoI/HBM metadata 的 CSV trace → PopNet `bench` 格式 → PopNet 周期仿真

尚未包含：完整 EIGEN/PIMnet/Torus-A2A+ 调度算法、heatmap、大规模 sweep。

---

## 目录结构

```text
TopoTraceSim/
├── README.md
├── README.zh-CN.md           # 中文说明（本文件）
├── configs/
│   └── noi_hbm_reconfigurable.yaml # NoI/HBM 硬件配置
├── scripts/
│   ├── run_a2a_full_pipeline.sh    # 一键流水线
│   ├── pytorchsim_csv_to_popnet.py # CSV → PopNet bench
│   ├── topotrace_hardware_config.py # NoI/HBM 配置解析
│   └── （PyTorchSim 驱动见 PyTorchSim/scripts/run_a2a_pytorchsim.py）
├── PyTorchSim/                # PyTorchSim 与 A2A 驱动脚本
├── third_party/
│   └── popnet_anytopo/       # PopNet（CMake 编译）
├── traces/                   # PyTorchSim 输出的 A2A CSV
└── popnet_exp/
    ├── traces/a2a_n*/        # PopNet bench
    └── logs/                 # PopNet 运行日志
```

---

## 环境要求

| 组件 | 环境 |
|------|------|
| PyTorchSim | Linux + Docker（`ghcr.io/psal-postech/torchsim-ci:v1.0.0`） |
| PopNet | Linux、g++、CMake、Boost（`libboost-graph-dev`） |
| Python | 3.8+（转换脚本） |

macOS：完整流水线请在 Linux 服务器上跑；本地仅适合改代码，PopNet 需在 Linux 上编译运行。

---

## 快速开始（Linux 服务器）

```bash
cd /mnt/sdb1/wyf/TopoTraceSim

# 首次编译 PopNet
cd third_party/popnet_anytopo
mkdir -p build && cd build && cmake .. && make -j
cd ../..

# 完整流水线（12 次 PyTorchSim 内核，约 10–15 分钟）
bash scripts/run_a2a_full_pipeline.sh
```

**成功标志**

- PyTorchSim：`A2A trace check passed`，12 events
- PopNet 日志：`Packet count: 12`，`Finished packets: 12`

快速冒烟（约 1 分钟）：

```bash
bash scripts/run_a2a_full_pipeline.sh --smoke
```

显式指定 NoI/HBM 硬件配置：

```bash
bash scripts/run_a2a_full_pipeline.sh \
  --hardware-config configs/noi_hbm_reconfigurable.yaml \
  --smoke
```

若 PyTorchSim 已跑完，只做转换 + PopNet：

```bash
bash scripts/run_a2a_full_pipeline.sh --convert-only
```

---

## 分步说明

### 步骤 1 — PyTorchSim

在 Docker 内运行（镜像自带 Gem5 / Spike / BackendSim）：

```bash
docker run --rm --ipc=host \
  -v "$(pwd)/PyTorchSim/scripts:/workspace/PyTorchSim/scripts:ro" \
  -v "$(pwd)/traces:/workspace/PyTorchSim/traces" \
  -v "$(pwd)/PyTorchSim/togsim_results:/workspace/PyTorchSim/togsim_results" \
  -w /workspace/PyTorchSim \
  ghcr.io/psal-postech/torchsim-ci:v1.0.0 \
  python scripts/run_a2a_pytorchsim.py
```

root pipeline 会写出带硬件标签的 CSV 文件名，例如：

```text
traces/a2a_n4_16KB_runtimeReconfigurableCrossbar_pytorchsim.csv
```

如果单独运行 PyTorchSim driver 且不传 `--out`，则仍使用 driver 内部默认输出名。

root pipeline 还会写出：

- `traces/noi_hbm_hardware_metadata.json`：展开后的硬件字段，会附加到 CSV 每一行
- `traces/noi_hbm_hardware_summary.txt`：便于阅读的 NoI/HBM 配置摘要

### 硬件配置

默认硬件配置为：

```text
configs/noi_hbm_reconfigurable.yaml
```

它包含三个顶层部分：

| 部分 | 作用 |
|------|------|
| `NoI` | 互连结构类型、compute/HBM 端口、重构粒度、控制延迟、流控 |
| `NoINI` | compute-side endpoint adapter 数量、注入/弹出 buffer、D2D lane 与 lane rate |
| `HBMSide` | HBM stack 数量、HBM-side NoI stop、staging buffer、logical channel、HBM bandwidth |

`NoI.interconnectType` 支持：

- `runtimeReconfigurableCrossbar`
- `fixedInterconnect`

### 步骤 2 — 转为 PopNet trace

任意目录均可（脚本使用绝对默认路径）：

```bash
python3 /path/to/TopoTraceSim/scripts/pytorchsim_csv_to_popnet.py
```

输出路径由 root pipeline 根据硬件配置选择，例如
`popnet_exp/traces/a2a_n4_runtimeReconfigurableCrossbar/bench`
（格式：`T sx sy dx dy n`）。

### 步骤 3 — PopNet

```bash
./third_party/popnet_anytopo/build/popnet \
  -A 2 -c 2 -V 3 -B 12 -O 12 -F 4 \
  -L 1000 -T 100000 -r 1 \
  -I "$(pwd)/popnet_exp/traces/a2a_n4_runtimeReconfigurableCrossbar/bench" -R 0
```

---

## 字段映射（PyTorchSim CSV → PopNet）

| PyTorchSim | PopNet | 规则 |
|------------|--------|------|
| `inject_cycle` | `T` | 注入时刻 |
| `src` | `sx sy` | 0→(0,0)，1→(0,1)，2→(1,0)，3→(1,1) |
| `dst` | `dx dy` | 同上 |
| `flits` | `n` | flit 数（16 KB、64 B/flit 时为 256） |

---

## 同步到服务器

```bash
rsync -avz -e "ssh -p 端口 -i 密钥" \
  --exclude 'third_party/popnet_anytopo/build' \
  --exclude '.git' \
  ./TopoTraceSim/ user@主机:/mnt/sdb1/wyf/TopoTraceSim/
```

在服务器 `third_party/popnet_anytopo/build/` 下编译一次 PopNet。

---

## 说明

- PyTorchSim 侧用 **matmul 负载** 驱动完整仿真栈；CSV 记录的是 **A2A 拓扑与通信量**，不是从 NoC 日志自动解析的包级 trace。
- PopNet 主输入为 **单个 `bench` 文件**（`-I`）；`bench.x.y` 为兼容 random_trace 布局的可选分片。

---

## 致谢

- PyTorchSim — POSTECH SAL  
- popnet_anytopo — FCAS-SCUT
