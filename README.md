# TopoTraceSim

**TopoTraceSim** connects [PyTorchSim](https://github.com/PSAL-POSTECH/PyTorchSim) (NPU compute simulation) with [popnet_anytopo](https://github.com/FCAS-SCUT/popnet_anytopo) (NoC simulation) through a minimal **All-to-All (A2A)** trace flow.

Current scope (Phase 1):

- NoI/HBM hardware configuration through `configs/noi_hbm_reconfigurable.yaml`
- Configurable compute-side NoI ports, HBM-side NoI ports, NoI-NI, D2D, and HBM-side parameters
- 12 directed flows (each node sends 16 KB to every other node)
- Full PyTorchSim stack per flow: `torch.compile` → Gem5 → Spike → BackendSim/TOGSim
- CSV trace with NoI/HBM metadata → PopNet `bench` format → PopNet cycle simulation

Not included yet: full EIGEN/PIMnet/Torus-A2A+ scheduling algorithms, heatmaps, large sweeps.

---

## Repository layout

```text
TopoTraceSim/
├── README.md                 # English (this file)
├── README.zh-CN.md           # Chinese
├── configs/
│   └── noi_hbm_reconfigurable.yaml # NoI/HBM hardware configuration
├── scripts/
│   ├── run_a2a_full_pipeline.sh    # one-shot pipeline
│   ├── pytorchsim_csv_to_popnet.py # CSV → PopNet bench
│   ├── topotrace_hardware_config.py # NoI/HBM config loader
│   └── (see PyTorchSim/scripts/run_a2a_pytorchsim.py)
├── PyTorchSim/                # PyTorchSim frontend + our A2A driver
├── third_party/
│   └── popnet_anytopo/       # PopNet (build with CMake)
├── traces/                   # PyTorchSim A2A CSV output
└── popnet_exp/
    ├── traces/a2a_n*/        # PopNet bench files
    └── logs/                 # PopNet run logs
```

---

## Requirements

| Component | Environment |
|-----------|-------------|
| PyTorchSim | Linux + Docker (`ghcr.io/psal-postech/torchsim-ci:v1.0.0`) |
| PopNet | Linux, g++, CMake, Boost (`libboost-graph-dev`) |
| Python | 3.8+ for conversion script |

macOS: use the remote Linux server for the full pipeline; PopNet must be rebuilt locally if you develop there.

---

## Quick start (Linux server)

```bash
# 1. Clone or copy TopoTraceSim to the server, e.g.
cd /mnt/sdb1/wyf/TopoTraceSim

# 2. Build PopNet (first time)
cd third_party/popnet_anytopo
mkdir -p build && cd build && cmake .. && make -j
cd ../..

# 3. Run full pipeline (~10–15 min for 12 PyTorchSim kernels)
bash scripts/run_a2a_full_pipeline.sh
```

**Success criteria**

- PyTorchSim: `A2A trace check passed`, 12 events
- PopNet log: `Packet count: 12`, `Finished packets: 12`

Smoke test (~1 min):

```bash
bash scripts/run_a2a_full_pipeline.sh --smoke
```

Run with an explicit NoI/HBM hardware config:

```bash
bash scripts/run_a2a_full_pipeline.sh \
  --hardware-config configs/noi_hbm_reconfigurable.yaml \
  --smoke
```

If PyTorchSim already finished, only convert + PopNet:

```bash
bash scripts/run_a2a_full_pipeline.sh --convert-only
```

---

## Step-by-step

### Step 1 — PyTorchSim

Runs inside Docker (image includes Gem5, Spike, BackendSim):

```bash
docker run --rm --ipc=host \
  -v "$(pwd)/PyTorchSim/scripts:/workspace/PyTorchSim/scripts:ro" \
  -v "$(pwd)/traces:/workspace/PyTorchSim/traces" \
  -v "$(pwd)/PyTorchSim/togsim_results:/workspace/PyTorchSim/togsim_results" \
  -w /workspace/PyTorchSim \
  ghcr.io/psal-postech/torchsim-ci:v1.0.0 \
  python scripts/run_a2a_pytorchsim.py
```

The root pipeline writes a hardware-tagged CSV name, for example:

```text
traces/a2a_n4_16KB_runtimeReconfigurableCrossbar_pytorchsim.csv
```

The standalone PyTorchSim driver still uses its internal default if `--out` is
not provided.

The root pipeline also writes:

- `traces/noi_hbm_hardware_metadata.json` — flattened hardware fields appended to each CSV row
- `traces/noi_hbm_hardware_summary.txt` — human-readable NoI/HBM summary

### Hardware Configuration

The default hardware configuration is:

```text
configs/noi_hbm_reconfigurable.yaml
```

It contains three top-level sections:

| Section | Purpose |
|---------|---------|
| `NoI` | Interconnect type, compute/HBM ports, reconfiguration granularity, control latency, flow control |
| `NoINI` | Compute-side endpoint adapter count, injection/ejection buffers, D2D lanes and lane rate |
| `HBMSide` | HBM stack count, HBM-side NoI stops, staging buffer, logical channels, HBM bandwidth |

`NoI.interconnectType` accepts:

- `runtimeReconfigurableCrossbar`
- `fixedInterconnect`

### Step 2 — Convert to PopNet

Works from **any directory** (absolute defaults):

```bash
python3 /path/to/TopoTraceSim/scripts/pytorchsim_csv_to_popnet.py
```

Output path is selected by the root pipeline from the hardware config, for
example `popnet_exp/traces/a2a_n4_runtimeReconfigurableCrossbar/bench`
(format: `T sx sy dx dy n`).

### Step 3 — PopNet

```bash
./third_party/popnet_anytopo/build/popnet \
  -A 2 -c 2 -V 3 -B 12 -O 12 -F 4 \
  -L 1000 -T 100000 -r 1 \
  -I "$(pwd)/popnet_exp/traces/a2a_n4_runtimeReconfigurableCrossbar/bench" -R 0
```

---

## Field mapping (PyTorchSim CSV → PopNet)

| PyTorchSim | PopNet | Rule |
|------------|--------|------|
| `inject_cycle` | `T` | injection time |
| `src` | `sx sy` | node 0→(0,0), 1→(0,1), 2→(1,0), 3→(1,1) |
| `dst` | `dx dy` | same map |
| `flits` | `n` | flit count (256 for 16 KB @ 64 B/flit) |

---

## Sync to server

```bash
rsync -avz -e "ssh -p PORT -i KEY" \
  --exclude 'third_party/popnet_anytopo/build' \
  --exclude '.git' \
  ./TopoTraceSim/ user@host:/mnt/sdb1/wyf/TopoTraceSim/
```

On server, build PopNet once under `third_party/popnet_anytopo/build/`.

---

## Acknowledgements

- PyTorchSim — POSTECH SAL
- popnet_anytopo — FCAS-SCUT
