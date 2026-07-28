"""Simple trace script"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("Starting trace...")
print("=" * 80)

try:
    from algo_trading.data_loader.loader import load_yfinance
    print("✅ Import load_yfinance OK")
except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("✅ Script completed!")





































