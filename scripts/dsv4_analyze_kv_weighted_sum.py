#!/usr/bin/env python3
"""Compatibility wrapper for the HC weighted-sum analyzer.

Turn #98 validation historically referenced this filename once; the analyzer
is HC_PRE_NORM-specific and lives in dsv4_analyze_hc_weighted_sum.py.
"""

from __future__ import annotations

from dsv4_analyze_hc_weighted_sum import main


if __name__ == "__main__":
    raise SystemExit(main())
