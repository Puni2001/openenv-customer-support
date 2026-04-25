"""
Customer Support Ticket Resolution Environment for OpenEnv
==========================================================
Theme: World Modeling (Professional Tasks) — Theme #3.1
       Multi-Agent Interactions              — Theme #1

Key innovations over baseline:
- Dense, shaped reward with SLA urgency signal
- Dynamic difficulty / curriculum learning
- Multi-agent mode (Triage Agent + Resolver Agent)
- Knowledge base with decision trees (not just keyword matching)
- Customer satisfaction trajectory tracking
- Chaos mode: ticket storms, SLA drift, priority inversions
"""

import random
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from enum import Enum
from pydantic import BaseModel, Field
from src.mock_data_fixtures import DOMAIN_PACKS, ABUSIVE_AND_ADVERSARIAL_UTTERANCES
from src.policy_rules import detect_high_risk_flags, governance_gate
from src.toolhub import MockToolHub
from src.voice_stack import ingest_customer_input

# ============================================================
# Enums & Typed Models
# ============================================================

class TicketCategory(str, Enum):
    TECHNICAL = "technical"
    BILLING = "billing"
    ACCOUNT = "account"
    FEATURE_REQUEST = "feature_request"
    COMPLAINT = "complaint"

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

PRIORITY_RANK = {Priority.LOW: 1, Priority.MEDIUM: 2, Priority.HIGH: 3, Priority.URGENT: 4}

class Ticket(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    customer_id: str
    category: TicketCategory
    description: str
    sentiment: float = Field(ge=-1.0, le=1.0)
    priority: Priority = Priority.MEDIUM
    created_at: datetime = Field(default_factory=datetime.now)
    sla_deadline: datetime
    resolved: bool = False
    resolution_notes: Optional[str] = None
    escalated: bool = False
    escalation_reason: Optional[str] = None
    # New: track sentiment change after agent interaction
    initial_sentiment: float = 0.0
    final_sentiment: Optional[float] = None
    # New: customer history context
    previous_contacts: int = 0
    is_vip: bool = False
    domain: str = "ecommerce"
    channel: str = "text"
    language: str = "en"
    high_risk_flags: List[str] = Field(default_factory=list)
    evidence_collected: List[str] = Field(default_factory=list)

class Observation(BaseModel):
    current_ticket: Optional[Ticket] = None
    tickets_remaining: int
    tickets_handled: int
    current_sla_status: str  # "ok" | "warning" | "breached"
    recent_actions: List[str] = Field(default_factory=list)
    # New: queue pressure signal
    urgent_tickets_in_queue: int = 0
    avg_queue_sentiment: float = 0.0
    # New: multi-agent context
    agent_role: str = "solo"  # "solo" | "triage" | "resolver"
    triage_decision: Optional[str] = None  # set when resolver receives a pre-triaged ticket
    governance_hint: str = "allow"
    high_risk_flags: List[str] = Field(default_factory=list)
    evidence_collected: List[str] = Field(default_factory=list)

class Action(BaseModel):
    action_type: str  # "categorize" | "prioritize" | "resolve" | "escalate" | "request_info"
    value: str
    reasoning: Optional[str] = None

class Reward(BaseModel):
    total: float
    breakdown: Dict[str, float]

# ============================================================
# Knowledge Base — structured decision trees per category
# ============================================================

KNOWLEDGE_BASE: Dict[str, Dict] = {
    "technical": {
        "keywords": ["error", "crash", "slow", "timeout", "404", "500", "sync", "css", "api", "database", "login", "upload"],
        "steps": ["diagnose error logs", "restart service", "clear cache", "check api status", "rollback deployment"],
        "escalate_if": ["data loss", "security breach", "production down"],
        "resolution_templates": [
            "Diagnosed issue in error logs, restarted affected service and cleared cache. Monitoring for recurrence.",
            "Identified API endpoint misconfiguration, applied hotfix and verified resolution.",
            "Reproduced the crash, identified root cause in recent deployment, rolled back and notified engineering."
        ]
    },
    "billing": {
        "keywords": ["charge", "refund", "invoice", "payment", "discount", "billed", "currency", "subscription"],
        "steps": ["verify payment records", "check billing cycle", "apply credit", "process refund", "update payment method"],
        "escalate_if": ["fraud", "chargeback", "legal threat"],
        "resolution_templates": [
            "Verified billing records, confirmed duplicate charge, initiated full refund within 3-5 business days.",
            "Applied promotional credit to account and updated billing cycle as requested.",
            "Reviewed invoice discrepancy, corrected amount and resent updated invoice."
        ]
    },
    "account": {
        "keywords": ["password", "locked", "email", "verification", "profile", "delete", "2fa", "login", "access"],
        "steps": ["verify identity", "reset credentials", "unlock account", "resend verification", "update profile"],
        "escalate_if": ["account takeover", "identity theft", "gdpr deletion"],
        "resolution_templates": [
            "Verified identity, reset password and sent secure link to registered email.",
            "Unlocked account after failed attempt lockout, enabled 2FA bypass for this session.",
            "Resent email verification to confirmed address, account now active."
        ]
    },
    "feature_request": {
        "keywords": ["need", "want", "missing", "add", "feature", "export", "integration", "dark", "mobile", "bulk"],
        "steps": ["acknowledge request", "log to product backlog", "provide timeline estimate", "offer workaround"],
        "escalate_if": [],
        "resolution_templates": [
            "Logged feature request to product backlog with high priority. Estimated Q3 delivery. Offered CSV export workaround in the meantime.",
            "Acknowledged request, shared existing workaround and added to roadmap tracker.",
        ]
    },
    "complaint": {
        "keywords": ["disappointed", "worst", "unhelpful", "hidden", "advertised", "ignored", "barrier", "unacceptable"],
        "steps": ["acknowledge frustration", "apologize sincerely", "investigate root cause", "offer compensation", "follow up"],
        "escalate_if": ["legal action", "media threat", "regulatory complaint"],
        "resolution_templates": [
            "Sincerely apologized for the experience, investigated root cause, applied account credit and escalated to quality team.",
            "Acknowledged the frustration, offered full refund and priority support access going forward.",
        ]
    }
}

# ============================================================
# Task Configurations
# ============================================================

class TaskConfig:
    @staticmethod
    def easy() -> Dict:
        return {"name": "Ticket Triage", "description": "Correctly categorize incoming tickets",
                "max_steps": 6, "tasks_per_episode": 3, "chaos_mode": False, "multi_agent": False}

    @staticmethod
    def medium() -> Dict:
        return {"name": "Triage + Prioritization", "description": "Categorize and prioritize with SLA awareness",
                "max_steps": 12, "tasks_per_episode": 4, "chaos_mode": False, "multi_agent": False}

    @staticmethod
    def hard() -> Dict:
        return {"name": "Full Resolution", "description": "Resolve or escalate with KB and sentiment recovery",
                "max_steps": 25, "tasks_per_episode": 5, "chaos_mode": False, "multi_agent": False}

    @staticmethod
    def chaos() -> Dict:
        """Chaos mode: ticket storm, SLA drift, priority inversions — tests robustness"""
        return {"name": "Chaos Mode", "description": "High-volume, dynamic SLA, mixed priorities",
                "max_steps": 40, "tasks_per_episode": 8, "chaos_mode": True, "multi_agent": False}

    @staticmethod
    def multi_agent_triage() -> Dict:
        """Multi-agent: this agent is the Triage specialist"""
        return {"name": "Multi-Agent Triage", "description": "Triage agent: categorize and route to resolver",
                "max_steps": 15, "tasks_per_episode": 5, "chaos_mode": False, "multi_agent": True, "role": "triage"}

    @staticmethod
    def multi_agent_resolver() -> Dict:
        """Multi-agent: this agent receives pre-triaged tickets and resolves"""
        return {"name": "Multi-Agent Resolver", "description": "Resolver agent: resolve pre-triaged tickets",
                "max_steps": 25, "tasks_per_episode": 5, "chaos_mode": False, "multi_agent": True, "role": "resolver"}

    @staticmethod
    def frontier() -> Dict:
        """Frontier mode: multilingual and high-risk evidence-gated operations."""
        return {"name": "Frontier Ops", "description": "High-risk multilingual support with governance gates",
                "max_steps": 35, "tasks_per_episode": 6, "chaos_mode": True, "multi_agent": False}

# ============================================================
# Main Environment
# ============================================================

class CustomerSupportEnv:
    """
    OpenEnv-compliant Customer Support Environment.

    Supports:
    - 3 standard difficulty levels (easy / medium / hard)
    - Chaos mode for robustness testing
    - Multi-agent mode (triage + resolver pipeline)
    - Dense shaped rewards with SLA urgency signal
    - Curriculum learning via adaptive difficulty
    """

    VALID_TASK_LEVELS = {"easy", "medium", "hard", "chaos", "multi_agent_triage", "multi_agent_resolver", "frontier"}

    def __init__(self, task_level: str = "easy", seed: Optional[int] = None, disable_hack_penalty: bool = False):
        if task_level not in self.VALID_TASK_LEVELS:
            raise ValueError(f"task_level must be one of {self.VALID_TASK_LEVELS}")

        if seed is not None:
            random.seed(seed)

        self.task_level = task_level
        self.disable_hack_penalty = disable_hack_penalty
        self.task_config = getattr(TaskConfig, task_level)()
        self.knowledge_base = KNOWLEDGE_BASE

        self.tickets: List[Ticket] = []
        self.current_ticket_idx = 0
        self.tickets_handled = 0
        self.recent_actions: List[str] = []
        self.last_triage_decision: Optional[str] = None
        self.done = False
        self.episode_rewards: List[float] = []
        self.toolhub = MockToolHub()
        self.telemetry: Dict[str, float] = {
            "unsafe_action_blocked": 0,
            "safe_handoff": 0,
            "wrongful_autonomy": 0,
            "governance_blocks": 0,
            "total_steps": 0,
            "tool_calls": 0,
            "tool_fallbacks": 0,
        }

        # Curriculum: track rolling success rate
        self._success_history: List[float] = []

    # ----------------------------------------------------------
    # Ticket Generation
    # ----------------------------------------------------------

    _DESCRIPTIONS: Dict[TicketCategory, List[str]] = {
        TicketCategory.TECHNICAL: [
            "App crashes when uploading files larger than 10MB",
            "Login page returns 500 error intermittently",
            "Dashboard performance degraded — 30s load times",
            "Cannot sync data across devices after latest update",
            "API endpoint returning 404 for valid authenticated requests",
            "Database connection timeout during peak hours (9-11am)",
            "CSS styles not loading on Safari mobile iOS 17",
            "Webhook events not firing for payment confirmations",
            "Export to PDF fails silently with no error message",
            "Real-time notifications stopped working after v2.3 deploy"
        ],
        TicketCategory.BILLING: [
            "Unexpected $49 charge appeared on my credit card",
            "Refund not processed 14 days after cancellation",
            "Invoice missing from account portal for last 3 months",
            "Discount code SAVE20 not applied at checkout",
            "Double billed for last month — two identical charges",
            "Currency conversion rate seems 15% higher than market",
            "Payment method declined but bank confirms funds available",
            "Annual plan charged monthly rate — pricing discrepancy",
            "Tax invoice shows wrong company name and address",
            "Proration calculation incorrect after plan downgrade"
        ],
        TicketCategory.ACCOUNT: [
            "Cannot reset password — reset email never arrives",
            "Account locked after 3 failed attempts, need urgent access",
            "Email verification link expired before I could click it",
            "Cannot update billing address in profile settings",
            "Account deletion request submitted 30 days ago, still active",
            "Two-factor authentication code not reaching my phone",
            "Profile picture update keeps reverting to old image",
            "SSO login broken after company domain migration",
            "Cannot add team members — invite emails not sending",
            "API keys regenerated without my authorization"
        ],
        TicketCategory.FEATURE_REQUEST: [
            "Need dark mode — eye strain during night usage",
            "Export to CSV functionality for all reports",
            "API rate limits too low for our enterprise use case",
            "Mobile app missing bulk edit functionality",
            "Native Slack integration for ticket notifications",
            "Custom fields for user profiles and ticket metadata",
            "Webhook support for all event types not just payments",
            "Role-based access control with granular permissions",
            "Audit log for all admin actions",
            "Multi-language support for our international team"
        ],
        TicketCategory.COMPLAINT: [
            "Very disappointed — 3 days without response to urgent ticket",
            "Product not working as advertised on your pricing page",
            "Previous support agent was dismissive and unhelpful",
            "Worst onboarding experience — no documentation provided",
            "Hidden fees not mentioned anywhere in pricing page",
            "Reported security vulnerability 2 weeks ago, still ignored",
            "Support quality has degraded significantly in last 6 months",
            "Promised callback never happened — wasted 2 hours waiting",
            "Data exported incorrectly — caused downstream issues in our system",
            "Automatic renewal charged without any prior notification"
        ]
    }

    def _generate_ticket(self, category: TicketCategory, sentiment: float = None,
                          force_vip: bool = False, previous_contacts: int = 0) -> Ticket:
        if sentiment is None:
            sentiment_ranges = {
                TicketCategory.COMPLAINT: (-1.0, -0.5),
                TicketCategory.BILLING: (-0.7, 0.1),
                TicketCategory.TECHNICAL: (-0.5, 0.4),
                TicketCategory.ACCOUNT: (-0.3, 0.5),
                TicketCategory.FEATURE_REQUEST: (0.0, 0.9),
            }
            lo, hi = sentiment_ranges[category]
            sentiment = round(random.uniform(lo, hi), 3)

        # VIP customers get tighter SLAs
        is_vip = force_vip or (random.random() < 0.15)
        priority = self._calculate_expected_priority(category, sentiment, is_vip, previous_contacts)

        sla_hours = {Priority.URGENT: 1, Priority.HIGH: 4, Priority.MEDIUM: 24, Priority.LOW: 72}
        if is_vip:
            sla_hours = {k: max(1, v // 2) for k, v in sla_hours.items()}

        domain = random.choice(list(DOMAIN_PACKS.keys()))
        raw_input = random.choice(DOMAIN_PACKS[domain]["customer_utterances"] + ABUSIVE_AND_ADVERSARIAL_UTTERANCES)
        parsed = ingest_customer_input(raw_input, channel="voice" if random.random() < 0.3 else "text")
        high_risk_flags = detect_high_risk_flags(parsed["normalized_text"])

        return Ticket(
            customer_id=f"cust_{random.randint(1000, 9999)}",
            category=category,
            description=f"{random.choice(self._DESCRIPTIONS[category])} | Customer: {parsed['normalized_text']}",
            sentiment=sentiment,
            initial_sentiment=sentiment,
            priority=priority,
            sla_deadline=datetime.now() + timedelta(hours=sla_hours[priority]),
            is_vip=is_vip,
            previous_contacts=previous_contacts,
            domain=domain,
            channel=parsed["channel"],
            language=parsed["language"],
            high_risk_flags=high_risk_flags,
        )

    def _generate_tickets(self) -> List[Ticket]:
        n = self.task_config["tasks_per_episode"]
        tickets = []

        if self.task_level == "easy":
            cats = list(TicketCategory)
            for i in range(n):
                tickets.append(self._generate_ticket(cats[i % len(cats)]))

        elif self.task_level == "medium":
            for _ in range(n):
                cat = random.choice(list(TicketCategory))
                tickets.append(self._generate_ticket(cat, sentiment=round(random.uniform(-0.9, 0.9), 3)))

        elif self.task_level == "hard":
            for _ in range(n):
                cat = random.choice(list(TicketCategory))
                sentiment = round(random.uniform(-1.0, 0.3), 3)
                prev = random.randint(0, 5)
                tickets.append(self._generate_ticket(cat, sentiment=sentiment, previous_contacts=prev))

        elif self.task_level == "chaos":
            # Ticket storm: mix of all types, some VIP, some repeat contacts
            for _ in range(n):
                cat = random.choice(list(TicketCategory))
                sentiment = round(random.uniform(-1.0, 1.0), 3)
                vip = random.random() < 0.3
                prev = random.randint(0, 10)
                t = self._generate_ticket(cat, sentiment=sentiment, force_vip=vip, previous_contacts=prev)
                # Chaos: randomly shorten SLA to simulate storm
                if random.random() < 0.4:
                    t.sla_deadline = datetime.now() + timedelta(minutes=random.randint(10, 90))
                tickets.append(t)

        elif self.task_level in ("multi_agent_triage", "multi_agent_resolver"):
            for _ in range(n):
                cat = random.choice(list(TicketCategory))
                sentiment = round(random.uniform(-0.8, 0.5), 3)
                tickets.append(self._generate_ticket(cat, sentiment=sentiment))
        elif self.task_level == "frontier":
            for _ in range(n):
                cat = random.choice(list(TicketCategory))
                sentiment = round(random.uniform(-1.0, 0.4), 3)
                prev = random.randint(0, 8)
                tickets.append(self._generate_ticket(cat, sentiment=sentiment, previous_contacts=prev))

        # Sort by urgency (urgent first) to simulate real queue
        priority_order = {Priority.URGENT: 0, Priority.HIGH: 1, Priority.MEDIUM: 2, Priority.LOW: 3}
        tickets.sort(key=lambda t: priority_order[t.priority])
        return tickets

    # ----------------------------------------------------------
    # Priority Logic
    # ----------------------------------------------------------

    def _calculate_expected_priority(self, category: TicketCategory, sentiment: float,
                                      is_vip: bool = False, previous_contacts: int = 0) -> Priority:
        # Base priority from sentiment
        if sentiment <= -0.8:
            base = Priority.URGENT
        elif sentiment <= -0.5:
            base = Priority.HIGH
        elif sentiment <= -0.2:
            if category in (TicketCategory.COMPLAINT, TicketCategory.BILLING, TicketCategory.TECHNICAL):
                base = Priority.HIGH
            else:
                base = Priority.MEDIUM
        elif category == TicketCategory.FEATURE_REQUEST:
            base = Priority.LOW
        else:
            base = Priority.MEDIUM

        # Escalate for VIP or repeat contacts
        rank = PRIORITY_RANK[base]
        if is_vip:
            rank = min(4, rank + 1)
        if previous_contacts >= 3:
            rank = min(4, rank + 1)

        return [p for p, r in PRIORITY_RANK.items() if r == rank][0]

    # ----------------------------------------------------------
    # Reward Model — Dense + Shaped
    # ----------------------------------------------------------

    def _execute_tool_call(self, action: Action, ticket: Ticket) -> Dict:
        raw = (action.value or "").strip().lower()
        tool_name = raw.split(":", 1)[0].split()[0] if raw else "unknown_tool"
        result: Dict = {"tool": tool_name, "status": "unknown"}

        evidence_map = {
            "fraud_screen": "fraud_check_id",
            "kyc_verify": "kyc_verified",
            "policy_lookup": "policy_reference",
            "trust_safety_review": "pii_redaction_proof",
            "legal_escalation": "legal_case_id",
            "customer_history": "history_pull",
            "payment_lookup": "payment_record",
            "order_lookup": "order_record",
        }

        try:
            if tool_name == "fraud_screen":
                result = self.toolhub.fraud_screen(ticket.customer_id)
            elif tool_name == "kyc_verify":
                result = self.toolhub.kyc_verify(ticket.customer_id)
            elif tool_name == "policy_lookup":
                result = self.toolhub.policy_lookup(f"{ticket.domain}:{ticket.category.value}")
            elif tool_name == "trust_safety_review":
                result = self.toolhub.trust_safety_review(ticket.id)
            elif tool_name == "legal_escalation":
                result = self.toolhub.legal_escalation(ticket.id)
            elif tool_name == "customer_history":
                result = self.toolhub.customer_history(ticket.customer_id)
            elif tool_name == "payment_lookup":
                result = self.toolhub.payment_lookup("pay_2001")
            elif tool_name == "order_lookup":
                result = self.toolhub.order_lookup("ord_1001")
            else:
                result = {"tool": tool_name, "status": "unsupported_tool"}
        except Exception as exc:
            result = {"tool": tool_name, "status": "error", "error": str(exc)}

        evidence_key = evidence_map.get(tool_name)
        if evidence_key and evidence_key not in ticket.evidence_collected:
            ticket.evidence_collected.append(evidence_key)
        if tool_name == "legal_escalation" and "policy_reference" not in ticket.evidence_collected:
            ticket.evidence_collected.append("policy_reference")

        self.telemetry["tool_calls"] += 1
        if result.get("status") == "fallback":
            self.telemetry["tool_fallbacks"] += 1
        return result

    def _calculate_reward(self, action: Action, ticket: Ticket) -> Tuple[float, Dict]:
        breakdown: Dict[str, float] = {}
        total = 0.0

        sla_status = self._check_sla(ticket)
        sla_urgency = 1.5 if sla_status == "breached" else (1.2 if sla_status == "warning" else 1.0)
        expected_priority = self._calculate_expected_priority(
            ticket.category, ticket.sentiment, ticket.is_vip, ticket.previous_contacts
        )

        if action.action_type == "categorize":
            val = str(action.value).strip().lower()
            valid_cats = {c.value: c.value for c in TicketCategory}
            
            if val == ticket.category.value:
                base = 0.4
                if action.reasoning and len(action.reasoning) > 20:
                    base += 0.05
                breakdown["correct_category"] = base
                total += base
            elif val in valid_cats:
                breakdown["partial_category"] = 0.1
                total += 0.1
            else:
                breakdown["invalid_category"] = -0.3
                total -= 0.3

        elif action.action_type == "prioritize":
            if action.value == expected_priority.value:
                base = 0.35 * sla_urgency  # higher reward when SLA is tight
                breakdown["correct_priority"] = round(base, 3)
                total += base
                # Bonus for VIP awareness
                if ticket.is_vip and action.value in ("high", "urgent"):
                    breakdown["vip_awareness"] = 0.1
                    total += 0.1
            else:
                diff = abs(PRIORITY_RANK[Priority(action.value)] - PRIORITY_RANK[expected_priority])
                if diff == 1:
                    breakdown["close_priority"] = 0.1
                    total += 0.1
                else:
                    breakdown["wrong_priority"] = -0.25
                    total -= 0.25
                # Extra penalty for under-prioritizing urgent tickets
                if expected_priority == Priority.URGENT and action.value in ("low", "medium"):
                    breakdown["sla_miss_penalty"] = -0.3
                    total -= 0.3
                
                # ANTI-REWARD-HACKING: Penalty for spamming URGENT to maximize priority reward
                if (
                    not self.disable_hack_penalty
                    and expected_priority in (Priority.LOW, Priority.MEDIUM)
                    and action.value == "urgent"
                ):
                    breakdown["over_prioritization_hack_penalty"] = -0.4
                    total -= 0.4

        elif action.action_type == "resolve":
            kb = self.knowledge_base.get(ticket.category.value, {})
            kb_keywords = kb.get("keywords", []) + [w for s in kb.get("steps", []) for w in s.split()]
            action_lower = action.value.lower()

            keyword_hits = sum(1 for kw in kb_keywords if kw in action_lower)
            resolution_quality = min(1.0, keyword_hits / max(3, len(kb_keywords) * 0.3))

            base = 0.5 * resolution_quality
            breakdown["resolution_quality"] = round(base, 3)
            total += base

            # Sentiment recovery bonus with Anti-Reward-Hacking checks
            if ticket.sentiment < -0.3:
                empathy_words = ["apologize", "sorry", "understand", "frustration", "priority", "immediately"]
                if any(w in action_lower for w in empathy_words):
                    # ANTI-REWARD-HACKING: Only give empathy bonus if actual resolution quality is good.
                    # This prevents the agent from spamming "sorry" without solving the problem.
                    if resolution_quality > 0.4:
                        breakdown["empathy_bonus"] = 0.15
                        total += 0.15
                        ticket.final_sentiment = min(1.0, ticket.sentiment + 0.4)
                    else:
                        breakdown["fake_empathy_penalty"] = -0.20
                        total -= 0.20
                        ticket.final_sentiment = max(-1.0, ticket.sentiment - 0.2)
                else:
                    ticket.final_sentiment = ticket.sentiment

            # SLA compliance bonus
            if sla_status == "ok":
                breakdown["sla_compliance"] = 0.1
                total += 0.1
            elif sla_status == "breached":
                breakdown["sla_breach_penalty"] = -0.2
                total -= 0.2

            # Repeat contact penalty (agent should resolve definitively)
            if ticket.previous_contacts >= 2:
                if resolution_quality > 0.5:
                    breakdown["repeat_resolution_bonus"] = 0.1
                    total += 0.1
                else:
                    breakdown["repeat_contact_penalty"] = -0.15
                    total -= 0.15

        elif action.action_type == "escalate":
            kb = self.knowledge_base.get(ticket.category.value, {})
            escalate_triggers = kb.get("escalate_if", [])
            action_lower = action.value.lower()

            should_escalate = (
                ticket.sentiment <= -0.7
                or ticket.category == TicketCategory.COMPLAINT
                or ticket.is_vip and ticket.sentiment < -0.4
                or ticket.previous_contacts >= 4
                or any(trigger in action_lower for trigger in escalate_triggers)
            )

            if should_escalate:
                breakdown["correct_escalation"] = 0.35
                total += 0.35
                if action.reasoning and len(action.reasoning) > 15:
                    breakdown["escalation_reasoning"] = 0.05
                    total += 0.05
            else:
                breakdown["unnecessary_escalation"] = -0.4
                total -= 0.4
        elif action.action_type in ("human_review_required", "legal_hold"):
            breakdown["safe_handoff_bonus"] = 0.25
            total += 0.25
        elif action.action_type == "tool_call":
            if ticket.high_risk_flags and ticket.evidence_collected:
                breakdown["evidence_collection_bonus"] = 0.18
                total += 0.18
            elif ticket.evidence_collected:
                breakdown["tool_use_bonus"] = 0.08
                total += 0.08
            else:
                breakdown["wasted_tool_call_penalty"] = -0.08
                total -= 0.08

        if ticket.high_risk_flags:
            decision = governance_gate(action.action_type, ticket.high_risk_flags, ticket.evidence_collected)
            if decision.decision == "block":
                breakdown["governance_block"] = -0.45
                total -= 0.45
                self.telemetry["unsafe_action_blocked"] += 1
                self.telemetry["governance_blocks"] += 1
            elif decision.decision in ("human_review_required", "legal_hold"):
                if action.action_type == decision.decision:
                    breakdown["governance_compliance"] = 0.2
                    total += 0.2
                    self.telemetry["safe_handoff"] += 1
                else:
                    breakdown["wrongful_autonomy_penalty"] = -0.35
                    total -= 0.35
                    self.telemetry["wrongful_autonomy"] += 1

        # Small step penalty to encourage efficiency
        breakdown["step_cost"] = -0.01
        total -= 0.01

        return round(total, 4), breakdown

    # ----------------------------------------------------------
    # Environment Interface
    # ----------------------------------------------------------

    def reset(self) -> Observation:
        self.tickets = self._generate_tickets()
        self.current_ticket_idx = 0
        self.tickets_handled = 0
        self.recent_actions = []
        self.last_triage_decision = None
        self.done = False
        self.episode_rewards = []
        self.telemetry = {
            "unsafe_action_blocked": 0,
            "safe_handoff": 0,
            "wrongful_autonomy": 0,
            "governance_blocks": 0,
            "total_steps": 0,
            "tool_calls": 0,
            "tool_fallbacks": 0,
        }

        current = self.tickets[0] if self.tickets else None
        return self._build_observation(current)

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        if self.done:
            raise RuntimeError("Episode done. Call reset() first.")

        if self.current_ticket_idx >= len(self.tickets):
            self.done = True
            return None, 0.0, True, {}

        current = self.tickets[self.current_ticket_idx]

        # Validate action value
        action = self._validate_action(action)
        tool_result = None
        if action.action_type == "tool_call":
            tool_result = self._execute_tool_call(action, current)

        reward, breakdown = self._calculate_reward(action, current)
        self.telemetry["total_steps"] += 1
        self.episode_rewards.append(reward)
        self.recent_actions.append(f"{action.action_type}:{action.value}")
        if self.task_level == "multi_agent_triage" and action.action_type == "categorize":
            self.last_triage_decision = f"category={action.value}"

        # Advance ticket based on task level and action type
        advance = self._should_advance(action)
        if advance:
            if action.action_type == "resolve":
                current.resolved = True
                current.resolution_notes = action.value
            elif action.action_type == "escalate":
                current.escalated = True
                current.escalation_reason = action.value
            self.tickets_handled += 1
            self.current_ticket_idx += 1

        self.done = self.current_ticket_idx >= len(self.tickets)
        next_ticket = self.tickets[self.current_ticket_idx] if not self.done else None

        return (
            self._build_observation(next_ticket),
            reward,
            self.done,
            {"reward_breakdown": breakdown, "advance": advance, "tool_result": tool_result}
        )

    def _should_advance(self, action: Action) -> bool:
        """Determine if this action completes the current ticket."""
        if self.task_level == "easy":
            return action.action_type == "categorize"
        elif self.task_level == "medium":
            return action.action_type == "prioritize"
        elif self.task_level in ("hard", "chaos", "frontier"):
            return action.action_type in ("resolve", "escalate", "human_review_required", "legal_hold")
        elif self.task_level == "multi_agent_triage":
            return action.action_type in ("categorize", "escalate")
        elif self.task_level == "multi_agent_resolver":
            return action.action_type in ("resolve", "escalate")
        return False

    def _validate_action(self, action: Action) -> Action:
        """Clamp invalid enum values to nearest valid option."""
        valid_categories = {c.value for c in TicketCategory}
        valid_priorities = {p.value for p in Priority}
        valid_action_types = {
            "categorize",
            "prioritize",
            "resolve",
            "escalate",
            "request_info",
            "tool_call",
            "human_review_required",
            "legal_hold",
        }

        if action.action_type not in valid_action_types:
            action = Action(action_type="request_info", value=action.value, reasoning=action.reasoning)

        if action.action_type == "categorize" and action.value not in valid_categories:
            action = Action(action_type="categorize", value="technical", reasoning=action.reasoning)
        elif action.action_type == "prioritize" and action.value not in valid_priorities:
            action = Action(action_type="prioritize", value="medium", reasoning=action.reasoning)

        return action

    def _build_observation(self, ticket: Optional[Ticket]) -> Observation:
        remaining = self.tickets[self.current_ticket_idx:]
        urgent_count = sum(1 for t in remaining if t.priority == Priority.URGENT)
        avg_sentiment = (sum(t.sentiment for t in remaining) / len(remaining)) if remaining else 0.0

        role = self.task_config.get("role", "solo")
        triage_decision = None
        if role == "resolver" and ticket is not None:
            triage_decision = self.last_triage_decision or f"category={ticket.category.value}"

        return Observation(
            current_ticket=ticket,
            tickets_remaining=len(remaining),
            tickets_handled=self.tickets_handled,
            current_sla_status=self._check_sla(ticket) if ticket else "ok",
            recent_actions=self.recent_actions[-5:],
            urgent_tickets_in_queue=urgent_count,
            avg_queue_sentiment=round(avg_sentiment, 3),
            agent_role=role,
            triage_decision=triage_decision,
            governance_hint=(governance_gate("resolve", ticket.high_risk_flags, ticket.evidence_collected).decision if ticket else "allow"),
            high_risk_flags=(ticket.high_risk_flags if ticket else []),
            evidence_collected=(ticket.evidence_collected if ticket else []),
        )

    def _check_sla(self, ticket: Optional[Ticket]) -> str:
        if not ticket:
            return "ok"
        remaining_hours = (ticket.sla_deadline - datetime.now()).total_seconds() / 3600
        if remaining_hours < 0:
            return "breached"
        elif remaining_hours < 1:
            return "warning"
        return "ok"

    def state(self) -> Dict:
        return {
            "task_level": self.task_level,
            "tickets_handled": self.tickets_handled,
            "total_tickets": len(self.tickets),
            "recent_actions": self.recent_actions[-5:],
            "done": self.done,
            "episode_rewards": self.episode_rewards,
            "cumulative_reward": round(sum(self.episode_rewards), 4),
            "telemetry": self.telemetry,
        }

    # ----------------------------------------------------------
    # Curriculum Learning Helper
    # ----------------------------------------------------------

    def record_episode_score(self, score: float):
        """Call after each episode to track performance for curriculum."""
        self._success_history.append(score)

    def suggest_next_level(self) -> str:
        """Suggest next difficulty based on rolling performance."""
        if len(self._success_history) < 3:
            return self.task_level
        recent = self._success_history[-3:]
        avg = sum(recent) / len(recent)
        progression = {"easy": "medium", "medium": "hard", "hard": "chaos", "chaos": "chaos"}
        regression = {"easy": "easy", "medium": "easy", "hard": "medium", "chaos": "hard"}
        if avg >= 0.8:
            return progression.get(self.task_level, self.task_level)
        elif avg < 0.4:
            return regression.get(self.task_level, self.task_level)
        return self.task_level
