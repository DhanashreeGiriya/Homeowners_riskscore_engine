"""
setup.py  v3
Run ONCE before launching the app.
  python setup.py
"""
import os, sys, time
os.makedirs("data",   exist_ok=True)
os.makedirs("models", exist_ok=True)

print("="*65)
print("  US HOMEOWNERS RISK SCORING MODEL  —  SETUP  v3")
print("="*65)

t0 = time.time()
print("\n[1/2] Generating 100,000 synthetic policies …")
from data_generator import generate_dataset
df = generate_dataset(100_000)
df.to_parquet("data/homeowners_data.parquet", index=False)
print(f"      Saved  ({time.time()-t0:.1f}s)")

t1 = time.time()
print("\n[2/2] Training models (≈ 4-7 min on a modern laptop) …")
from model_trainer import train_all
arts = train_all(df)

print("\n" + "="*65)
print("  SETUP COMPLETE!")
print("="*65)
print("\nMetrics summary:")
for k,v in arts["metrics"].items():
    print(f"  {k:20s}: {v}")
print(f"\nTotal time: {time.time()-t0:.1f}s")
print("\nLaunch the app:")
print("  streamlit run app.py")
print("="*65)
