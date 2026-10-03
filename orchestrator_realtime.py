#!/usr/bin/env python3
"""Compatibility wrapper: the real-time implementation now lives in orchestrator.py."""
from orchestrator import forecast, forecast_with_metadata, main

if __name__ == "__main__":
    raise SystemExit(main())
