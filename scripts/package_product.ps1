$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$product = Join-Path $root 'product'

if (Test-Path $product) {
    Remove-Item $product -Recurse -Force
}

New-Item -ItemType Directory -Path $product | Out-Null

$itemsToCopy = @(
    'algo_trading',
    'config',
    'models',
    'start_trading_bot.py',
    'bot.py',
    'LIVE_TRADING_SETUP.md'
)

foreach ($item in $itemsToCopy) {
    $src = Join-Path $root $item
    $dst = Join-Path $product $item
    if (Test-Path $src) {
        Copy-Item -Path $src -Destination $dst -Recurse -Force
    }
}

# Create clean env template for public repo
@'
EXCHANGE=okx
MODE=paper
OKX_API_KEY=your_api_key_here
OKX_SECRET_KEY=your_secret_key_here
OKX_PASSPHRASE=your_passphrase_here
OKX_USE_SIMULATED_TRADING=1

SYMBOL=BTCUSDT
INTERVAL=5m
STRATEGY=moe_v2_enhanced
STRATEGY_PARAMS={}

MOE_MODEL_PATH=models/dynamic_moe_v2_enhanced_final.pkl
MOE_PROBA_THRESHOLD=0.35
MOE_USE_REGIME_SPECIFIC=false
MOE_USE_DYNAMIC_THRESHOLD=true
MOE_USE_QUANTILE_THRESHOLD=true
MOE_TARGET_SIGNAL_RATE=0.08
MOE_QUANTILE_WINDOW=400
MOE_QUANTILE_FLOOR=0.55

RISK_PER_TRADE=0.1
SL_PCT=0.02
TP_PCT=0.04
TRAILING_PCT=0

HISTORY_LIMIT=120
COOL_DOWN_SEC=300
CHECK_INTERVAL_SEC=30
MAX_POSITION_SIZE=0
MAX_DCA_ORDERS=1
'@ | Set-Content -Path (Join-Path $product '.env.example') -Encoding UTF8

@'
# Secrets
.env

# Runtime artifacts
*.pid
*.log

# Python
__pycache__/
*.pyc

# IDE
.vscode/
.idea/
'@ | Set-Content -Path (Join-Path $product '.gitignore') -Encoding UTF8

@'
# Product Bundle (MOE v2)

This folder is generated for deployment and GitHub publishing.

## Quick Start

1. Create virtual environment
2. Install dependencies from config/requirements.txt
3. Copy .env.example to .env and fill your keys
4. Start bot:

```bash
python start_trading_bot.py start
```

## Strategy Defaults

- STRATEGY=moe_v2_enhanced
- MOE_MODEL_PATH=models/dynamic_moe_v2_enhanced_final.pkl

## Notes

- Keep `.env` out of Git.
- Rotate API keys before publishing if they were exposed.
'@ | Set-Content -Path (Join-Path $product 'README.md') -Encoding UTF8

Write-Host 'Product bundle created at:' $product
