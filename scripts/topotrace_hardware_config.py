#!/usr/bin/env python3
"""Load and summarize TopoTraceSim NoI/HBM hardware configuration."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


TOPOTRACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = TOPOTRACE_ROOT / "configs/noi_hbm_reconfigurable.yaml"

DEFAULT_CONFIG: dict[str, Any] = {
    "NoI": {
        "interconnectType": "runtimeReconfigurableCrossbar",
        "computePorts": 4,
        "hbmPorts": 4,
        "reconfigurationGranularity": "collectivePhase",
        "topologyGenerationLatencyNs": 1000,
        "crossbarConfigurationLatencyNs": 50,
        "syncBarrierLatencyNs": 20,
        "flowControl": "credit",
    },
    "NoINI": {
        "endpointsPerComputeChiplet": 1,
        "injectionBufferMiB": 1,
        "ejectionBufferMiB": 1,
        "D2DLanes": 128,
        "D2DDataRateGTpsPerLane": 64,
    },
    "HBMSide": {
        "stackCount": 4,
        "stopsPerStack": 1,
        "stagingBufferMiB": 4,
        "logicalChannelsPerStack": 32,
        "channelWidthBits": 64,
        "bandwidthTBpsPerStack": 2.8,
    },
}

VALID_INTERCONNECT_TYPES = {
    "runtimeReconfigurableCrossbar",
    "fixedInterconnect",
}
VALID_RECONFIGURATION_GRANULARITIES = {
    "collectivePhase",
}
VALID_FLOW_CONTROLS = {
    "credit",
    "scheduledToken",
}


def _merge_config(defaults: dict[str, Any], user: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(defaults)
    for key, value in user.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def _require_positive_number(config: dict[str, Any], path: str) -> None:
    cur: Any = config
    for part in path.split("."):
        cur = cur[part]
    if cur <= 0:
        raise ValueError(f"{path} must be positive, got {cur!r}")


def _require_non_negative_number(config: dict[str, Any], path: str) -> None:
    cur: Any = config
    for part in path.split("."):
        cur = cur[part]
    if cur < 0:
        raise ValueError(f"{path} must be non-negative, got {cur!r}")


def _require_choice(config: dict[str, Any], path: str, choices: set[str]) -> None:
    cur: Any = config
    for part in path.split("."):
        cur = cur[part]
    if cur not in choices:
        options = ", ".join(sorted(choices))
        raise ValueError(f"{path} must be one of [{options}], got {cur!r}")


def load_hardware_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load a hardware config and apply defaults for missing fields."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Hardware config not found: {path}")

    with path.open() as f:
        user_config = yaml.safe_load(f) or {}
    if not isinstance(user_config, dict):
        raise ValueError(f"Hardware config root must be a mapping: {path}")

    config = _merge_config(DEFAULT_CONFIG, user_config)
    validate_hardware_config(config)
    config["_configPath"] = str(path)
    config["_derived"] = derive_values(config)
    return config


def validate_hardware_config(config: dict[str, Any]) -> None:
    _require_choice(config, "NoI.interconnectType", VALID_INTERCONNECT_TYPES)
    _require_choice(
        config,
        "NoI.reconfigurationGranularity",
        VALID_RECONFIGURATION_GRANULARITIES,
    )
    _require_choice(config, "NoI.flowControl", VALID_FLOW_CONTROLS)

    numeric_paths = [
        "NoI.computePorts",
        "NoI.hbmPorts",
        "NoINI.endpointsPerComputeChiplet",
        "NoINI.injectionBufferMiB",
        "NoINI.ejectionBufferMiB",
        "NoINI.D2DLanes",
        "NoINI.D2DDataRateGTpsPerLane",
        "HBMSide.stackCount",
        "HBMSide.stopsPerStack",
        "HBMSide.stagingBufferMiB",
        "HBMSide.logicalChannelsPerStack",
        "HBMSide.channelWidthBits",
        "HBMSide.bandwidthTBpsPerStack",
    ]
    for path in numeric_paths:
        _require_positive_number(config, path)

    latency_paths = [
        "NoI.topologyGenerationLatencyNs",
        "NoI.crossbarConfigurationLatencyNs",
        "NoI.syncBarrierLatencyNs",
    ]
    for path in latency_paths:
        _require_non_negative_number(config, path)


def derive_values(config: dict[str, Any]) -> dict[str, Any]:
    noi = config["NoI"]
    noini = config["NoINI"]
    hbm = config["HBMSide"]

    compute_ports = int(noi["computePorts"])
    hbm_ports = int(noi["hbmPorts"])
    hbm_total_stops = int(hbm["stackCount"]) * int(hbm["stopsPerStack"])
    total_noi_ports = compute_ports + hbm_ports

    d2d_raw_tbps = (
        float(noini["D2DLanes"])
        * float(noini["D2DDataRateGTpsPerLane"])
        / 8.0
        / 1000.0
    )
    hbm_total_bandwidth_tbps = (
        float(hbm["stackCount"]) * float(hbm["bandwidthTBpsPerStack"])
    )

    runtime_reconfigurable = noi["interconnectType"] == "runtimeReconfigurableCrossbar"
    reconfiguration_latency_ns = (
        int(noi["topologyGenerationLatencyNs"])
        + int(noi["crossbarConfigurationLatencyNs"])
        + int(noi["syncBarrierLatencyNs"])
        if runtime_reconfigurable
        else 0
    )

    popnet_ary = int(math.sqrt(compute_ports))
    if popnet_ary * popnet_ary != compute_ports:
        raise ValueError(
            "computePorts must be a perfect square for the current 2D PopNet "
            f"mesh conversion, got {compute_ports}"
        )

    warnings: list[str] = []
    if hbm_total_stops != hbm_ports:
        warnings.append(
            "NoI.hbmPorts does not match HBMSide.stackCount * "
            f"HBMSide.stopsPerStack ({hbm_ports} != {hbm_total_stops})"
        )

    return {
        "computeNodes": compute_ports,
        "hbmTotalStops": hbm_total_stops,
        "totalNoIPorts": total_noi_ports,
        "runtimeReconfigurable": runtime_reconfigurable,
        "reconfigurationLatencyNs": reconfiguration_latency_ns,
        "rawD2DBandwidthTBpsPerComputeChiplet": round(d2d_raw_tbps, 6),
        "hbmTotalBandwidthTBps": round(hbm_total_bandwidth_tbps, 6),
        "popnetMeshDims": 2,
        "popnetMeshAry": popnet_ary,
        "warnings": warnings,
    }


def flatten_metadata(config: dict[str, Any]) -> dict[str, Any]:
    noi = config["NoI"]
    noini = config["NoINI"]
    hbm = config["HBMSide"]
    derived = config["_derived"]
    return {
        "hardware_config_path": config.get("_configPath", ""),
        "noi_interconnect_type": noi["interconnectType"],
        "noi_compute_ports": noi["computePorts"],
        "noi_hbm_ports": noi["hbmPorts"],
        "noi_total_ports": derived["totalNoIPorts"],
        "noi_reconfiguration_granularity": noi["reconfigurationGranularity"],
        "noi_topology_generation_latency_ns": noi["topologyGenerationLatencyNs"],
        "noi_crossbar_configuration_latency_ns": noi["crossbarConfigurationLatencyNs"],
        "noi_sync_barrier_latency_ns": noi["syncBarrierLatencyNs"],
        "noi_reconfiguration_latency_ns": derived["reconfigurationLatencyNs"],
        "noi_flow_control": noi["flowControl"],
        "noini_endpoints_per_compute_chiplet": noini["endpointsPerComputeChiplet"],
        "noini_injection_buffer_mib": noini["injectionBufferMiB"],
        "noini_ejection_buffer_mib": noini["ejectionBufferMiB"],
        "noini_d2d_lanes": noini["D2DLanes"],
        "noini_d2d_data_rate_gtps_per_lane": noini["D2DDataRateGTpsPerLane"],
        "noini_raw_d2d_bandwidth_tbps_per_compute_chiplet": derived[
            "rawD2DBandwidthTBpsPerComputeChiplet"
        ],
        "hbm_stack_count": hbm["stackCount"],
        "hbm_stops_per_stack": hbm["stopsPerStack"],
        "hbm_total_stops": derived["hbmTotalStops"],
        "hbm_staging_buffer_mib": hbm["stagingBufferMiB"],
        "hbm_logical_channels_per_stack": hbm["logicalChannelsPerStack"],
        "hbm_channel_width_bits": hbm["channelWidthBits"],
        "hbm_bandwidth_tbps_per_stack": hbm["bandwidthTBpsPerStack"],
        "hbm_total_bandwidth_tbps": derived["hbmTotalBandwidthTBps"],
    }


def summary_lines(config: dict[str, Any]) -> list[str]:
    metadata = flatten_metadata(config)
    lines = [
        "TopoTraceSim hardware configuration:",
        f"  config path                  : {metadata['hardware_config_path']}",
        f"  NoI interconnect type        : {metadata['noi_interconnect_type']}",
        f"  NoI compute/HBM/total ports  : {metadata['noi_compute_ports']} / "
        f"{metadata['noi_hbm_ports']} / {metadata['noi_total_ports']}",
        f"  NoI reconfiguration latency  : {metadata['noi_reconfiguration_latency_ns']} ns",
        f"  NoI flow control             : {metadata['noi_flow_control']}",
        f"  NoI-NI buffers               : injection {metadata['noini_injection_buffer_mib']} MiB, "
        f"ejection {metadata['noini_ejection_buffer_mib']} MiB",
        f"  D2D lanes/rate/raw bandwidth : {metadata['noini_d2d_lanes']} lanes, "
        f"{metadata['noini_d2d_data_rate_gtps_per_lane']} GT/s/lane, "
        f"{metadata['noini_raw_d2d_bandwidth_tbps_per_compute_chiplet']} TB/s per compute chiplet",
        f"  HBM stacks/stops             : {metadata['hbm_stack_count']} stacks, "
        f"{metadata['hbm_total_stops']} HBM-side NoI stops",
        f"  HBM staging buffer           : {metadata['hbm_staging_buffer_mib']} MiB per stop",
        f"  HBM logical channels         : {metadata['hbm_logical_channels_per_stack']} x "
        f"{metadata['hbm_channel_width_bits']}-bit per stack",
        f"  HBM total bandwidth          : {metadata['hbm_total_bandwidth_tbps']} TB/s",
    ]
    for warning in config["_derived"]["warnings"]:
        lines.append(f"  WARNING: {warning}")
    return lines


def env_lines(config: dict[str, Any]) -> list[str]:
    derived = config["_derived"]
    values = {
        "TOPOTRACE_COMPUTE_NODES": derived["computeNodes"],
        "TOPOTRACE_HBM_PORTS": config["NoI"]["hbmPorts"],
        "TOPOTRACE_TOTAL_NOI_PORTS": derived["totalNoIPorts"],
        "TOPOTRACE_POPNET_MESH_DIMS": derived["popnetMeshDims"],
        "TOPOTRACE_POPNET_MESH_ARY": derived["popnetMeshAry"],
        "TOPOTRACE_NOI_INTERCONNECT_TYPE": config["NoI"]["interconnectType"],
        "TOPOTRACE_NOI_RECONFIGURATION_LATENCY_NS": derived[
            "reconfigurationLatencyNs"
        ],
        "TOPOTRACE_HBM_TOTAL_STOPS": derived["hbmTotalStops"],
    }
    return [f"{key}={shlex.quote(str(value))}" for key, value in values.items()]


def write_metadata(config: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(flatten_metadata(config), f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load TopoTraceSim NoI/HBM hardware configuration."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to NoI/HBM hardware YAML configuration.",
    )
    parser.add_argument(
        "--format",
        choices=["summary", "json", "metadata-json", "env"],
        default="summary",
    )
    parser.add_argument(
        "--metadata-out",
        default="",
        help="Write flattened CSV metadata JSON to this path.",
    )
    args = parser.parse_args()

    try:
        config = load_hardware_config(args.config)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.metadata_out:
        write_metadata(config, Path(args.metadata_out))

    if args.format == "summary":
        print("\n".join(summary_lines(config)))
    elif args.format == "json":
        print(json.dumps(config, indent=2, sort_keys=True))
    elif args.format == "metadata-json":
        print(json.dumps(flatten_metadata(config), indent=2, sort_keys=True))
    elif args.format == "env":
        print("\n".join(env_lines(config)))


if __name__ == "__main__":
    main()
