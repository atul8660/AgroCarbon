"""
End-to-end test for the rewired Module5Adapter.
Run from: D:\Capston Zip Files\tree_carbon_assessment_system\
    python test_m5_adapter.py
"""
import sys, os
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../../../..'))

from backend.app.pipeline.blackbox_adapters import Module5Adapter

m5 = Module5Adapter()

# ── STEP 4a: Supported species ─────────────────────────────────────────────
print("=" * 55)
print("STEP 4a: species='Mango'  leaf_length_cm=18.0")
print("=" * 55)
result = m5.predict(species="Mango", leaf_length_cm=18.0)
for k, v in result.items():
    print(f"  {k:<18}: {v}")

print()
print("=" * 55)
print("STEP 4a: species='Neem'  leaf_length_cm=18.0")
print("=" * 55)
result2 = m5.predict(species="Neem", leaf_length_cm=18.0)
for k, v in result2.items():
    print(f"  {k:<18}: {v}")

# ── STEP 4b: Unsupported species ───────────────────────────────────────────
print()
print("=" * 55)
print("STEP 4b: species='Guava' (unsupported)")
print("=" * 55)
try:
    m5.predict(species="Guava", leaf_length_cm=18.0)
    print("FAIL: ValueError was NOT raised — guard-rail broken!")
except ValueError as e:
    print(f"  PASS: ValueError raised correctly.")
    print(f"  Message: {e}")

print()
print("=" * 55)
print("STEP 4b: species='Jamun' (unsupported)")
print("=" * 55)
try:
    m5.predict(species="Jamun", leaf_length_cm=18.0)
    print("FAIL: ValueError was NOT raised — guard-rail broken!")
except ValueError as e:
    print(f"  PASS: ValueError raised correctly.")
    print(f"  Message: {e}")
