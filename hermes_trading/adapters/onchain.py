import httpx
import time
import os

SCHEMA_VERSION = "1.0"


class SchemaError(Exception):
    pass


async def fetch() -> dict:
    glassnode_key = os.getenv("GLASSNODE_API_KEY", "")

    result = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": int(time.time()),
        "source": "unavailable",
        "active_addresses_24h": None,
        "exchange_netflow_btc": None,
    }

    if glassnode_key:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(
                    "https://api.glassnode.com/v1/metrics/addresses/active_count",
                    params={"a": "BTC", "api_key": glassnode_key},
                )
                r.raise_for_status()
                data = r.json()
                if data:
                    result["active_addresses_24h"] = data[-1].get("v")
                    result["source"] = "glassnode"
        except Exception:
            pass

    if "schema_version" not in result:
        raise SchemaError("onchain adapter missing schema_version")

    return result
