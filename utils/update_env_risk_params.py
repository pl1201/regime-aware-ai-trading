"""
Script để cập nhật .env với các tham số SL, TP, Risk cho regime_specific strategy
"""
import os
from pathlib import Path
from dotenv import load_dotenv, set_key

def update_env_risk_params():
    """Cập nhật .env với các tham số risk management"""
    
    env_path = Path(".env")
    
    # Load .env hiện tại nếu có
    if env_path.exists():
        load_dotenv(env_path, override=True)
        print("=" * 60)
        print("CAP NHAT .ENV VOI THAM SO SL/TP/RISK")
        print("=" * 60)
        print()
        print("File .env hien tai:")
        print(f"  RISK_PER_TRADE: {os.getenv('RISK_PER_TRADE', 'chua co')}")
        print(f"  SL_PCT: {os.getenv('SL_PCT', 'chua co')}")
        print(f"  TP_PCT: {os.getenv('TP_PCT', 'chua co')}")
        print()
    else:
        print("=" * 60)
        print("TAO FILE .ENV MOI")
        print("=" * 60)
        print()
        # Tạo file .env mới từ .env.example nếu có
        example_path = Path(".env.example")
        if example_path.exists():
            with open(example_path, 'r', encoding='utf-8') as f:
                content = f.read()
            with open(env_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print("Da tao .env tu .env.example")
        else:
            # Tạo file .env rỗng
            env_path.touch()
            print("Da tao file .env moi")
        print()
    
    # Các tham số mặc định (conservative cho regime_specific strategy)
    print("Cac tham so mac dinh (conservative):")
    print("  RISK_PER_TRADE = 0.02 (2% so du moi lenh)")
    print("  SL_PCT = 0.015 (1.5% stop loss)")
    print("  TP_PCT = 0.03 (3% take profit, RR = 2:1)")
    print()
    
    use_default = input("Dung tham so mac dinh? (y/n) [y]: ").strip().lower() or "y"
    
    if use_default == "y":
        risk_per_trade = "0.02"
        sl_pct = "0.015"
        tp_pct = "0.03"
    else:
        print()
        risk_per_trade = input("RISK_PER_TRADE (0.01-0.05, mac dinh 0.02): ").strip() or "0.02"
        sl_pct = input("SL_PCT (0.01-0.03, mac dinh 0.015): ").strip() or "0.015"
        tp_pct = input("TP_PCT (0.02-0.05, mac dinh 0.03): ").strip() or "0.03"
    
    # Cập nhật .env
    set_key(env_path, "RISK_PER_TRADE", risk_per_trade)
    set_key(env_path, "SL_PCT", sl_pct)
    set_key(env_path, "TP_PCT", tp_pct)
    
    # Đảm bảo STRATEGY_PARAMS có use_sequence_features=True
    strategy_params_str = os.getenv("STRATEGY_PARAMS", "{}")
    import json
    try:
        strategy_params = json.loads(strategy_params_str)
    except:
        strategy_params = {}
    
    # Cập nhật strategy params cho regime_specific với sequence features
    strategy_params.update({
        "use_regime_specific": True,
        "use_sequence_features": True,
        "regime_specific_model_path": "models/regime_specific_models_optimized.pkl",
        "sequence_model_path": "models/seq_lstm_extractor.pt",
        "sequence_len": 64,
        "proba_threshold": 0.55,
        "allowed_regimes": ["trending", "ranging", "calm"]
    })
    
    set_key(env_path, "STRATEGY", "regime_specific")
    set_key(env_path, "STRATEGY_PARAMS", json.dumps(strategy_params))
    
    print()
    print("=" * 60)
    print("DA CAP NHAT .ENV")
    print("=" * 60)
    print()
    print(f"RISK_PER_TRADE = {risk_per_trade} ({float(risk_per_trade)*100:.1f}% so du)")
    print(f"SL_PCT = {sl_pct} ({float(sl_pct)*100:.1f}% stop loss)")
    print(f"TP_PCT = {tp_pct} ({float(tp_pct)*100:.1f}% take profit)")
    print(f"Risk-Reward Ratio = {float(tp_pct)/float(sl_pct):.2f}:1")
    print()
    print("STRATEGY = regime_specific")
    print("STRATEGY_PARAMS da duoc cap nhat voi:")
    print("  - use_regime_specific: true")
    print("  - use_sequence_features: true")
    print("  - regime_specific_model_path: models/regime_specific_models_optimized.pkl")
    print("  - sequence_model_path: models/seq_lstm_extractor.pt")
    print()
    print("OK Hoan tat!")

if __name__ == "__main__":
    update_env_risk_params()
