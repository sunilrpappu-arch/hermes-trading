import httpx
import time

SCHEMA_VERSION = "1.0"


class SchemaError(Exception):
    pass


async def fetch() -> dict:
    dxy = None
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                "https://query1.finance.yahoo.com/v8/finance/chart/DX-Y.NYB",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            r.raise_for_status()
            data = r.json()
            result_data = data["chart"]["result"]
            if result_data:
                closes = result_data[0]["indicators"]["quote"][0].get("close", [])
                dxy = closes[-1] if closes else None
    except Exception:
        pass

    result = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": int(time.time()),
        "dxy": dxy,
        "source": "yahoo_finance" if dxy else "unavailable",
    }

    if "schema_version" not in result:
        raise SchemaError("macro adapter missing schema_version")

    return result
