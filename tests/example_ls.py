"""
Example showing how to list available options
"""
import rpps as rp

print("Modulation")
for k, v in rp.mod.ls().items():
    print(f"  {k}: {', '.join(v)}")

print("Coding")
for k, v in rp.coding.ls().items():
    print(f"  {k}: {', '.join(v)}")

print("Scram")
for k, v in rp.scram.ls().items():
    print(f"  {k}: {', '.join(v)}")

print("Window Functions")
print(f"  {', '.join(rp.filters.Window.s())}")
