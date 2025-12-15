import os
import requests

# Load API key from .env
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# --------------------------------------------------------
# 1️⃣ Main search function (Serper.dev)
# --------------------------------------------------------
def web_search_serper(query: str, num_results=5):
    """
    Performs a fast Google-like search using Serper.dev.
    Returns JSON results.
    """
    if not SERPER_API_KEY:
        return {"error": "Missing SERPER_API_KEY in environment variables"}

    url = "https://google.serper.dev/search"

    payload = {
        "q": query,
        "num": num_results
    }

    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# --------------------------------------------------------
# 2️⃣ Optional fallback — used because you imported it
# --------------------------------------------------------
def google_search(query: str):
    """
    Simple wrapper that uses Serper for Google-style search.
    """
    return web_search_serper(query)
