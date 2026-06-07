#!/usr/bin/env python3
"""
test_all_modules.py — Smoke-test all math-rendu Python modules
==============================================================

For each module in ../modules/, import it and run all demo_* functions.
Reports pass/fail for each demo.

Usage:
    python scripts/test_all_modules.py

Li Zhou — 2027 — MIT License
"""

import os
import sys
import importlib.util
import traceback

# Add modules/ to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULES_DIR = os.path.join(ROOT, "modules")
sys.path.insert(0, MODULES_DIR)

# Force non-interactive matplotlib
os.environ["MPLBACKEND"] = "Agg"

MODULES = [
    "calculus_intro",
    "multi_calc",
    "vector_analysis",
    "ode",
    "pde",
    "linalg_intro",
    "linalg_core",
    "linalg_apps",
    "complex_analysis",
    "probability",
    "statistics",
    "numerical",
    "optimization",
    "discrete",
]


def test_module(name):
    """Import a module and run every demo_* it exposes."""
    print(f"\n=== {name}.py ===")
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(MODULES_DIR, f"{name}.py")
    )
    if spec is None or spec.loader is None:
        print(f"  [FAIL] could not load spec")
        return 0, 1
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"  [FAIL] import error: {e}")
        return 0, 1

    passed = failed = 0
    for attr in dir(mod):
        if attr.startswith("demo_"):
            try:
                getattr(mod, attr)()
                print(f"  [OK]   {attr}")
                passed += 1
            except Exception as e:
                print(f"  [FAIL] {attr}: {type(e).__name__}: {e}")
                failed += 1
    return passed, failed


def main():
    total_p = total_f = 0
    for m in MODULES:
        p, f = test_module(m)
        total_p += p
        total_f += f
    print("\n" + "=" * 50)
    print(f"Result: {total_p} passed, {total_f} failed")
    return 0 if total_f == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
