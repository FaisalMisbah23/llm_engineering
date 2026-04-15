import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt


@dataclass
class Interaction:
    ts_iso: str
    provider: str
    model: str
    expertise: str
    response_time_s: float
    success: bool
    error: Optional[str] = None


analytics_logs: List[Interaction] = []


def log_interaction(
    provider: str,
    model: str,
    expertise: str,
    response_time_s: float,
    success: bool,
    error: Optional[str] = None,
) -> None:
    analytics_logs.append(
        Interaction(
            ts_iso=datetime.utcnow().isoformat(),
            provider=provider,
            model=model,
            expertise=expertise,
            response_time_s=response_time_s,
            success=success,
            error=error,
        )
    )


def export_json() -> str:
    return json.dumps([asdict(x) for x in analytics_logs], indent=2)


def summarize() -> Dict[str, Any]:
    total = len(analytics_logs)
    successes = sum(1 for x in analytics_logs if x.success)
    avg_rt = (sum(x.response_time_s for x in analytics_logs) / total) if total else 0.0

    by_provider = defaultdict(list)
    by_model = defaultdict(list)
    hourly = Counter()
    daily = Counter()

    for x in analytics_logs:
        by_provider[x.provider].append(x)
        by_model[f"{x.provider}:{x.model}"].append(x)
        dt = datetime.fromisoformat(x.ts_iso)
        hourly[dt.strftime("%H:00")] += 1
        daily[dt.strftime("%Y-%m-%d")] += 1

    provider_stats = {}
    for p, items in by_provider.items():
        provider_stats[p] = {
            "count": len(items),
            "success_rate": sum(1 for i in items if i.success) / len(items),
            "avg_response_time_s": sum(i.response_time_s for i in items) / len(items),
        }

    model_stats = {}
    for m, items in by_model.items():
        model_stats[m] = {
            "count": len(items),
            "success_rate": sum(1 for i in items if i.success) / len(items),
            "avg_response_time_s": sum(i.response_time_s for i in items) / len(items),
        }

    return {
        "total": total,
        "success_rate": (successes / total) if total else 0.0,
        "avg_response_time_s": avg_rt,
        "provider_stats": provider_stats,
        "model_stats": model_stats,
        "hourly_usage": dict(hourly),
        "daily_usage": dict(daily),
    }


def plot_provider_avg_latency():
    s = summarize()["provider_stats"]
    providers = list(s.keys())
    values = [s[p]["avg_response_time_s"] for p in providers]

    fig = plt.figure(figsize=(7, 4))
    plt.bar(providers, values)
    plt.title("Average response time by provider (s)")
    plt.ylabel("seconds")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    return fig


def plot_daily_usage():
    daily = summarize()["daily_usage"]
    keys = sorted(daily.keys())
    vals = [daily[k] for k in keys]
    fig = plt.figure(figsize=(7, 4))
    plt.plot(keys, vals, marker="o")
    plt.title("Daily usage")
    plt.ylabel("count")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    return fig
