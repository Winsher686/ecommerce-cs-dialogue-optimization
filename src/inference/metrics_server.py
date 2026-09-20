"""
Prometheus 指标。

用法:
    from src.inference.metrics_server import (
        REQUEST_COUNT, REQUEST_LATENCY, ACTIVE_ADAPTER,
        record_request,
    )
"""

import time
from contextlib import contextmanager

from src.utils.logger import get_logger

logger = get_logger(__name__)


try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    _HAS_PROM = True
except ImportError:
    _HAS_PROM = False
    logger.warning("prometheus_client not installed, metrics disabled")


if _HAS_PROM:
    REQUEST_COUNT = Counter(
        "cs_api_requests_total",
        "Total requests",
        ["target", "status"],
    )
    REQUEST_LATENCY = Histogram(
        "cs_api_latency_seconds",
        "Request latency",
        ["target"],
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    ACTIVE_ADAPTER = Gauge(
        "cs_api_active_adapter",
        "Currently active LoRA adapter",
        ["name"],
    )
    FINETUNED_RATIO = Gauge(
        "cs_api_finetuned_ratio",
        "Current finetuned traffic ratio",
    )
else:
    REQUEST_COUNT = None
    REQUEST_LATENCY = None
    ACTIVE_ADAPTER = None
    FINETUNED_RATIO = None


@contextmanager
def record_request(target: str):
    start = time.time()
    status = "success"
    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        elapsed = time.time() - start
        if _HAS_PROM:
            REQUEST_COUNT.labels(target=target, status=status).inc()
            REQUEST_LATENCY.labels(target=target).observe(elapsed)


def set_active_adapter(name: str) -> None:
    if _HAS_PROM:
        ACTIVE_ADAPTER.labels(name=name).set(1)


def set_finetuned_ratio(ratio: float) -> None:
    if _HAS_PROM:
        FINETUNED_RATIO.set(ratio)


def start_metrics_server(port: int = 9090) -> None:
    if not _HAS_PROM:
        logger.warning("skip metrics server (prometheus_client not installed)")
        return
    start_http_server(port)
    logger.info(f"metrics server started on :{port}")


if __name__ == "__main__":
    start_metrics_server(9090)
    logger.info("metrics server running, press Ctrl+C to stop")
    import time as _t
    while True:
        _t.sleep(60)
