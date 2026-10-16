# Pipeline Orchestration Directory Placeholder

This directory is designated for orchestrating the multi-stage pipeline flow, connecting the individual modules.

Potential contents:
1. Orchestration scripts (e.g. airflow, cron, or simple Python script runners) that feed output from tree detection to segmentation, and then key crops to species and health classification.
2. CLI scripts that call the respective module entrypoints.
3. System-level orchestration configs determining execution paths and endpoints.
