"""
Mock tool hub for policy, fraud, history, and transactional lookups.
"""

import random
from typing import Dict
from src.mock_api_stack import MockOrderApi, MockPaymentApi, MockFraudApi, MockPolicyApi


class MockToolHub:
    def __init__(self):
        self._orders = {"ord_1001": {"status": "delayed", "eta_days": 3}}
        self._payments = {"pay_2001": {"state": "captured", "amount": 4999}}
        self._history = {"cust_1001": {"previous_contacts": 4, "risk_score": 0.71}}
        self._order_api = MockOrderApi()
        self._payment_api = MockPaymentApi()
        self._fraud_api = MockFraudApi()
        self._policy_api = MockPolicyApi()
        self._provider_health = {
            "order_api": "healthy",
            "payment_api": "healthy",
            "fraud_api": "healthy",
            "policy_api": "healthy",
        }

    def _provider_fallback(self, provider_name: str, primary: Dict, fallback: Dict) -> Dict:
        if primary.get("status") == "ok":
            self._provider_health[provider_name] = "healthy"
            return primary
        self._provider_health[provider_name] = "degraded"
        return {
            "status": "fallback",
            "provider": provider_name,
            "degraded_reason": primary.get("error_type", primary.get("status", "unknown")),
            "data": fallback,
            "latency_ms": primary.get("latency_ms", 0),
        }

    def order_lookup(self, order_id: str) -> Dict:
        primary = self._order_api.get_order(order_id)
        fallback = self._orders.get(order_id, {"order_id": order_id, "status": "unknown"})
        return self._provider_fallback("order_api", primary, fallback)

    def payment_lookup(self, payment_id: str) -> Dict:
        primary = self._payment_api.get_payment(payment_id)
        fallback = self._payments.get(payment_id, {"payment_id": payment_id, "state": "unknown"})
        return self._provider_fallback("payment_api", primary, fallback)

    def policy_lookup(self, policy_key: str) -> Dict:
        primary = self._policy_api.lookup(policy_key)
        fallback = {"policy_key": policy_key, "policy_reference": f"POL-{policy_key}-fallback", "clause": "mock_clause_v2", "allowed": True}
        return self._provider_fallback("policy_api", primary, fallback)

    def fraud_screen(self, customer_id: str) -> Dict:
        primary = self._fraud_api.screen_customer(customer_id)
        profile = self._history.get(customer_id, {"risk_score": 0.2})
        fallback = {
            "customer_id": customer_id,
            "fraud_check_id": f"fr_{customer_id}",
            "risk_score": profile["risk_score"],
            "recommended_action": "manual_review" if profile["risk_score"] > 0.7 else "allow",
        }
        return self._provider_fallback("fraud_api", primary, fallback)

    def customer_history(self, customer_id: str) -> Dict:
        base = self._history.get(customer_id, {"previous_contacts": 0, "risk_score": 0.1})
        return {
            "customer_id": customer_id,
            "previous_contacts": base["previous_contacts"],
            "risk_score": base["risk_score"],
            "avg_csats_last_5": round(random.uniform(1.8, 4.9), 2),
            "last_escalation_days_ago": random.randint(1, 180),
        }

    def kyc_verify(self, customer_id: str) -> Dict:
        return {
            "customer_id": customer_id,
            "kyc_verified": True,
            "verification_id": f"kyc_{customer_id}",
            "method": random.choice(["otp", "video_kyc", "doc_match"]),
        }

    def legal_escalation(self, ticket_id: str) -> Dict:
        return {
            "ticket_id": ticket_id,
            "legal_case_id": f"lg_{ticket_id}",
            "status": "created",
            "priority": random.choice(["high", "critical"]),
        }

    def trust_safety_review(self, ticket_id: str) -> Dict:
        return {
            "ticket_id": ticket_id,
            "review_state": "queued",
            "pii_redaction_proof": True,
            "sla_minutes": random.choice([15, 30, 60]),
        }

    def providers_health(self) -> Dict:
        return {"providers": self._provider_health}

