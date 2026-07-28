"""
Script mẫu thu thập tin tức từ API (Finnhub, CryptoPanic) để làm data cho sentiment.

Cần API key (miễn phí) từ:
- Finnhub: https://finnhub.io/
- CryptoPanic: https://cryptopanic.com/developers/api/

Usage:
    export FINNHUB_API_KEY=your_key
    python scripts/collect_news_sample.py --output data/sentiment/raw/news_crypto.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import requests


def fetch_finnhub_news(api_key: str, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Lấy tin crypto từ Finnhub (free tier: 60 calls/min).
    """
    url = "https://finnhub.io/api/v1/news"
    params = {"category": "crypto", "token": api_key}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    items = data[:limit] if isinstance(data, list) else []
    
    out = []
    for item in items:
        out.append({
            "text": (item.get("headline") or "") + " " + (item.get("summary") or ""),
            "title": item.get("headline", ""),
            "timestamp": datetime.fromtimestamp(item.get("datetime", 0)).isoformat(),
            "source": "finnhub",
            "symbol": "BTCUSDT",  # Finnhub crypto news không phân symbol, gán chung
            "url": item.get("url", ""),
        })
    return out


def fetch_cryptopanic_news(api_key: str, filter_: str = "hot") -> List[Dict[str, Any]]:
    """
    Lấy tin từ CryptoPanic (free: 30 req/day).
    """
    url = "https://cryptopanic.com/api/v1/posts/"
    params = {
        "auth_token": api_key,
        "filter": filter_,
        "public": "true",
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])
    
    out = []
    for item in results:
        title = item.get("title", "")
        url_news = item.get("url", "")
        created = item.get("created_at", "")[:19].replace("T", " ")
        currencies = item.get("currencies", [])
        symbol = "BTCUSDT"
        if currencies:
            code = currencies[0].get("code", "BTC")
            symbol = f"{code}USDT" if code != "USD" else "BTCUSDT"
        
        out.append({
            "text": title,
            "title": title,
            "timestamp": created,
            "source": "cryptopanic",
            "symbol": symbol,
            "url": url_news,
        })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/sentiment/raw/news_crypto.json")
    parser.add_argument("--source", choices=["finnhub", "cryptopanic"], default="finnhub")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    news = []
    
    if args.source == "finnhub":
        api_key = os.environ.get("FINNHUB_API_KEY")
        if not api_key:
            print("⚠️ Set FINNHUB_API_KEY. Lấy key miễn phí tại https://finnhub.io/")
            return 1
        news = fetch_finnhub_news(api_key, limit=args.limit)
    elif args.source == "cryptopanic":
        api_key = os.environ.get("CRYPTOPANIC_API_KEY")
        if not api_key:
            print("⚠️ Set CRYPTOPANIC_API_KEY. Lấy key tại https://cryptopanic.com/developers/api/")
            return 1
        news = fetch_cryptopanic_news(api_key)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(news, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Đã lưu {len(news)} tin vào {output_path}")
    return 0


if __name__ == "__main__":
    exit(main())
