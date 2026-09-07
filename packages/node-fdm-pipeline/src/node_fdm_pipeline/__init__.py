"""node-fdm-pipeline — CLI pipeline for the node-fdm Neural ODE framework."""

from __future__ import annotations

from node_fdm_pipeline.commands.fleet_campaign import run_fleet_campaign

__all__ = [
    "cli",
    "commands",
    "config",
    "resolver",
    "run_fleet_campaign",
]
