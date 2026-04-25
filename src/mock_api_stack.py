"""
Production-style mock API stack for support operations.
Includes realistic payloads, transient failures, and latency simulation.
"""

from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Optional

try:
    from faker import Faker
except Exception:  # pragma: no cover - fallback when Faker is unavailable
    Faker = None


@dataclass
class MockApiResponse:
    status: str
    provider: str
    data: Dict
    latency_ms: int
    error_type: Optional[str] = None

    def to_dict(self) -> Dict:
        out = {
            "status": self.status,
            "provider": self.provider,
            "latency_ms": self.latency_ms,
            "data": self.data,
        }
        if self.error_type:
            out["error_type"] = self.error_type
        return out


class MockProviderBase:
    def __init__(self, name: str, failure_rate: float = 0.06, timeout_rate: float = 0.03):
        self.name = name
        self.failure_rate = failure_rate
        self.timeout_rate = timeout_rate
        self._faker = Faker("en_IN") if Faker else None

    def _latency(self) -> int:
        return random.randint(25, 240)

    def _simulate_transient_state(self) -> Optional[MockApiResponse]:
        latency = self._latency()
        # Keep sleep tiny so local runs stay fast but still represent timing.
        time.sleep(latency / 10000.0)
        p = random.random()
        if p < self.timeout_rate:
            return MockApiResponse(
                status="timeout",
                provider=self.name,
                latency_ms=latency,
                error_type="upstream_timeout",
                data={},
            )
        if p < self.timeout_rate + self.failure_rate:
            return MockApiResponse(
                status="error",
                provider=self.name,
                latency_ms=latency,
                error_type="rate_limited",
                data={},
            )
        return None

    def _customer_name(self) -> str:
        if self._faker:
            return self._faker.name()
        return f"Customer-{uuid.uuid4().hex[:6]}"

    def _city(self) -> str:
        if self._faker:
            return self._faker.city()
        return "Bengaluru"


class MockOrderApi(MockProviderBase):
    def __init__(self):
        super().__init__("order_api", failure_rate=0.05, timeout_rate=0.02)

    def get_order(self, order_id: str) -> Dict:
        transient = self._simulate_transient_state()
        if transient:
            return transient.to_dict()
        latency = self._latency()
        payload = {
            "order_id": order_id,
            "status": random.choice(["delayed", "in_transit", "delivered", "exception"]),
            "eta_days": random.randint(0, 5),
            "carrier": random.choice(["Delhivery", "BlueDart", "EcomExpress"]),
            "destination_city": self._city(),
            "customer_name": self._customer_name(),
        }
        return MockApiResponse("ok", self.name, payload, latency).to_dict()


class MockPaymentApi(MockProviderBase):
    def __init__(self):
        super().__init__("payment_api", failure_rate=0.08, timeout_rate=0.03)

    def get_payment(self, payment_id: str) -> Dict:
        transient = self._simulate_transient_state()
        if transient:
            return transient.to_dict()
        latency = self._latency()
        payload = {
            "payment_id": payment_id,
            "state": random.choice(["captured", "pending", "failed", "chargeback"]),
            "amount_inr": random.randint(499, 29999),
            "method": random.choice(["upi", "card", "netbanking", "wallet"]),
            "risk_signal": round(random.uniform(0.05, 0.95), 2),
        }
        return MockApiResponse("ok", self.name, payload, latency).to_dict()


class MockFraudApi(MockProviderBase):
    def __init__(self):
        super().__init__("fraud_api", failure_rate=0.04, timeout_rate=0.02)

    def screen_customer(self, customer_id: str) -> Dict:
        transient = self._simulate_transient_state()
        if transient:
            return transient.to_dict()
        latency = self._latency()
        risk = round(random.uniform(0.05, 0.98), 2)
        payload = {
            "customer_id": customer_id,
            "fraud_check_id": f"fr_{customer_id}_{uuid.uuid4().hex[:4]}",
            "risk_score": risk,
            "recommended_action": "manual_review" if risk > 0.75 else "allow",
        }
        return MockApiResponse("ok", self.name, payload, latency).to_dict()


class MockPolicyApi(MockProviderBase):
    def __init__(self):
        super().__init__("policy_api", failure_rate=0.03, timeout_rate=0.01)

    def lookup(self, policy_key: str) -> Dict:
        transient = self._simulate_transient_state()
        if transient:
            return transient.to_dict()
        latency = self._latency()
        payload = {
            "policy_key": policy_key,
            "policy_reference": f"POL-{policy_key.upper().replace(':', '-')}-v2",
            "clause": random.choice(["clause_4_2", "clause_7_1", "clause_9_3"]),
            "requires_human_review": random.choice([True, False]),
        }
        return MockApiResponse("ok", self.name, payload, latency).to_dict()
