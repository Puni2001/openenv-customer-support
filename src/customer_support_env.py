"""Customer Support Ticket Resolution Environment for OpenEnv"""

import random
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from enum import Enum
from pydantic import BaseModel, Field

# ============================================
# Typed Models (OpenEnv Requirement)
# ============================================

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

class Ticket(BaseModel):
    """Represents a customer support ticket"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    customer_id: str
    category: TicketCategory
    description: str
    sentiment: float = Field(ge=-1.0, le=1.0)  # -1 (very negative) to +1 (very positive)
    priority: Priority = Priority.MEDIUM
    created_at: datetime = Field(default_factory=datetime.now)
    sla_deadline: datetime
    resolved: bool = False
    resolution_notes: Optional[str] = None
    escalated: bool = False
    escalation_reason: Optional[str] = None

class Observation(BaseModel):
    """What the agent sees at each step"""
    current_ticket: Ticket
    tickets_remaining: int
    tickets_handled: int
    current_sla_status: str  # "ok", "warning", "breached"
    recent_actions: List[str] = Field(default_factory=list)

class Action(BaseModel):
    """What the agent can do"""
    action_type: str  # "categorize", "prioritize", "resolve", "escalate", "request_info"
    value: str
    reasoning: Optional[str] = None

class Reward(BaseModel):
    """Reward structure with breakdown"""
    total: float
    breakdown: Dict[str, float]

# ============================================
# Task Configurations (3 difficulty levels)
# ============================================

class TaskConfig:
    """Configuration for different difficulty levels"""
    
    @staticmethod
    def easy() -> Dict:
        return {
            "name": "Ticket Categorization",
            "description": "Correctly categorize incoming support tickets",
            "max_steps": 5,
            "success_criteria": "All tickets correctly categorized",
            "tasks_per_episode": 3
        }
    
    @staticmethod
    def medium() -> Dict:
        return {
            "name": "Categorization + Prioritization with SLA",
            "description": "Categorize tickets AND set appropriate priority based on SLA",
            "max_steps": 10,
            "success_criteria": "Correct category + priority meets SLA constraints",
            "tasks_per_episode": 5
        }
    
    @staticmethod
    def hard() -> Dict:
        return {
            "name": "Full Resolution with KB and Escalation",
            "description": "Resolve tickets using knowledge base, escalate when needed",
            "max_steps": 20,
            "success_criteria": "Tickets resolved correctly or appropriately escalated",
            "tasks_per_episode": 7
        }

# ============================================
# Main Environment
# ============================================

class CustomerSupportEnv:
    """OpenEnv-compliant Customer Support Environment"""
    
    def __init__(self, task_level: str = "easy"):
        """Initialize environment with specified difficulty
        
        Args:
            task_level: "easy", "medium", or "hard"
        """
        self.task_level = task_level
        self.task_config = {
            "easy": TaskConfig.easy(),
            "medium": TaskConfig.medium(),
            "hard": TaskConfig.hard()
        }[task_level]
        
        self.tickets: List[Ticket] = []
        self.current_ticket_idx = 0
        self.tickets_handled = 0
        self.recent_actions: List[str] = []
        self.done = False
        
        # Knowledge base for hard tasks
        self.knowledge_base = self._build_knowledge_base()
        
    def _build_knowledge_base(self) -> Dict[str, str]:
        """Build simulated knowledge base for resolution tasks"""
        return {
            "technical": "Check error logs, restart service, clear cache",
            "billing": "Verify payment method, check billing cycle, apply credits if applicable",
            "account": "Reset password, verify email, check account status",
            "feature_request": "Route to product team, collect requirements",
            "complaint": "Apologize, investigate root cause, offer compensation"
        }
    
    def _generate_ticket(self, category: TicketCategory, sentiment: float = None) -> Ticket:
        """Generate a realistic support ticket"""
        
        descriptions = {
            TicketCategory.TECHNICAL: [
                "App crashes when uploading files",
                "Login page returns 500 error",
                "Slow performance in dashboard",
                "Cannot sync data across devices"
            ],
            TicketCategory.BILLING: [
                "Unexpected charge on my credit card",
                "Refund not processed after cancellation",
                "Invoice missing from account",
                "Discount code not applied"
            ],
            TicketCategory.ACCOUNT: [
                "Cannot reset password",
                "Account locked after multiple attempts",
                "Email verification not received",
                "Cannot update profile information"
            ],
            TicketCategory.FEATURE_REQUEST: [
                "Need dark mode feature",
                "Export to CSV functionality",
                "API rate limits too low",
                "Mobile app missing features"
            ],
            TicketCategory.COMPLAINT: [
                "Very disappointed with support response time",
                "Product not working as advertised",
                "Unhelpful previous agent",
                "Worst experience ever"
            ]
        }
        
        # Set sentiment based on category and random factors
        if sentiment is None:
            if category == TicketCategory.COMPLAINT:
                sentiment = random.uniform(-1.0, -0.5)
            elif category == TicketCategory.BILLING:
                sentiment = random.uniform(-0.5, 0.2)
            else:
                sentiment = random.uniform(-0.2, 0.8)
        
        # Calculate SLA deadline (urgent = 2hr, high = 4hr, medium = 24hr, low = 48hr)
        base_hours = {
            Priority.URGENT: 2,
            Priority.HIGH: 4,
            Priority.MEDIUM: 24,
            Priority.LOW: 48
        }
        
        # Initially set default priority based on category
        default_priority = {
            TicketCategory.COMPLAINT: Priority.HIGH,
            TicketCategory.BILLING: Priority.MEDIUM,
            TicketCategory.TECHNICAL: Priority.MEDIUM,
            TicketCategory.ACCOUNT: Priority.MEDIUM,
            TicketCategory.FEATURE_REQUEST: Priority.LOW
        }[category]
        
        sla_deadline = datetime.now() + timedelta(hours=base_hours[default_priority])
        
        return Ticket(
            customer_id=f"cust_{random.randint(1000, 9999)}",
            category=category,
            description=random.choice(descriptions[category]),
            sentiment=sentiment,
            priority=default_priority,
            sla_deadline=sla_deadline
        )
    
    def _generate_tickets(self) -> List[Ticket]:
        """Generate tickets based on task level"""
        tickets = []
        num_tickets = self.task_config["tasks_per_episode"]
        
        if self.task_level == "easy":
            # Easy: Only categorization needed
            categories = [TicketCategory.TECHNICAL, TicketCategory.BILLING, 
                         TicketCategory.ACCOUNT, TicketCategory.FEATURE_REQUEST]
            for _ in range(num_tickets):
                cat = random.choice(categories)
                tickets.append(self._generate_ticket(cat))
                
        elif self.task_level == "medium":
            # Medium: Mix with sentiment analysis
            for _ in range(num_tickets):
                cat = random.choice(list(TicketCategory))
                sentiment = random.uniform(-0.8, 0.9)
                tickets.append(self._generate_ticket(cat, sentiment))
                
        else:  # hard
            # Hard: Complex cases with escalation paths
            for _ in range(num_tickets):
                cat = random.choice(list(TicketCategory))
                # Hard tasks have more negative sentiment
                sentiment = random.uniform(-1.0, 0.3)
                ticket = self._generate_ticket(cat, sentiment)
                # Add complexity flags
                if random.random() < 0.3:
                    ticket.escalated = False  # Will need agent decision to escalate
                tickets.append(ticket)
        
        return tickets
    
    def _calculate_reward(self, action: Action, ticket: Ticket) -> Tuple[float, Dict]:
        """Calculate reward with partial progress signals"""
        reward_breakdown = {}
        total_reward = 0.0
        
        # Check if action matches required category
        if action.action_type == "categorize":
            if action.value == ticket.category.value:
                reward_breakdown["correct_category"] = 0.4
                total_reward += 0.4
            elif action.value in [c.value for c in TicketCategory]:
                reward_breakdown["partial_category"] = 0.1
                total_reward += 0.1
            else:
                reward_breakdown["wrong_category"] = -0.2
                total_reward -= 0.2
        
        # Check priority (for medium/hard tasks)
        elif action.action_type == "prioritize":
            expected_priority = self._get_expected_priority(ticket)
            if action.value == expected_priority.value:
                reward_breakdown["correct_priority"] = 0.3
                total_reward += 0.3
            else:
                # Partial credit for reasonable priority
                priority_values = {Priority.LOW: 1, Priority.MEDIUM: 2, 
                                  Priority.HIGH: 3, Priority.URGENT: 4}
                if abs(priority_values[Priority(action.value)] - 
                       priority_values[expected_priority]) <= 1:
                    reward_breakdown["close_priority"] = 0.1
                    total_reward += 0.1
                else:
                    reward_breakdown["wrong_priority"] = -0.2
                    total_reward -= 0.2
        
        # Check resolution quality (for hard tasks)
        elif action.action_type == "resolve":
            if self.task_level == "hard":
                # Check if resolution matches knowledge base
                kb_solution = self.knowledge_base.get(ticket.category.value, "")
                if action.value and any(word in action.value.lower() for word in kb_solution.split()):
                    reward_breakdown["good_resolution"] = 0.5
                    total_reward += 0.5
                    
                    # Bonus for negative sentiment recovery
                    if ticket.sentiment < 0:
                        reward_breakdown["sentiment_recovery"] = 0.2
                        total_reward += 0.2
                else:
                    reward_breakdown["poor_resolution"] = -0.3
                    total_reward -= 0.3
        
        # Check escalation appropriateness
        elif action.action_type == "escalate":
            # Escalate if sentiment < -0.5 or category is complaint
            should_escalate = ticket.sentiment < -0.5 or ticket.category == TicketCategory.COMPLAINT
            if should_escalate and action.value:
                reward_breakdown["correct_escalation"] = 0.3
                total_reward += 0.3
            elif not should_escalate and action.value:
                reward_breakdown["unnecessary_escalation"] = -0.5
                total_reward -= 0.5
        
        return total_reward, reward_breakdown
    
    def _get_expected_priority(self, ticket: Ticket) -> Priority:
        """Determine expected priority based on ticket attributes"""
        if ticket.sentiment < -0.7:
            return Priority.URGENT
        elif ticket.sentiment < -0.3:
            return Priority.HIGH
        elif ticket.category == TicketCategory.COMPLAINT:
            return Priority.HIGH
        elif ticket.category == TicketCategory.BILLING and ticket.sentiment < -0.2:
            return Priority.HIGH
        else:
            return Priority.MEDIUM
    
    def reset(self) -> Observation:
        """Reset environment for new episode"""
        self.tickets = self._generate_tickets()
        self.current_ticket_idx = 0
        self.tickets_handled = 0
        self.recent_actions = []
        self.done = False
        
        if not self.tickets:
            self.done = True
            
        current_ticket = self.tickets[0] if self.tickets else None
        return Observation(
            current_ticket=current_ticket,
            tickets_remaining=len(self.tickets),
            tickets_handled=0,
            current_sla_status=self._check_sla(current_ticket) if current_ticket else "ok",
            recent_actions=[]
        )
    
    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        """Execute action and return next state"""
        
        if self.done:
            raise RuntimeError("Episode already done. Call reset() first.")
        
        if self.current_ticket_idx >= len(self.tickets):
            self.done = True
            return None, 0.0, True, {}
        
        current_ticket = self.tickets[self.current_ticket_idx]
        
        # Calculate reward
        reward, breakdown = self._calculate_reward(action, current_ticket)
        
        # Record action
        self.recent_actions.append(f"{action.action_type}: {action.value}")
        
        # Handle different action types
        if action.action_type == "resolve":
            current_ticket.resolved = True
            current_ticket.resolution_notes = action.value
            self.tickets_handled += 1
            self.current_ticket_idx += 1
            
        elif action.action_type == "escalate":
            current_ticket.escalated = True
            current_ticket.escalation_reason = action.value
            self.tickets_handled += 1
            self.current_ticket_idx += 1
            
        elif action.action_type in ["categorize", "prioritize", "request_info"]:
            # Non-terminal actions - stay on same ticket
            pass
        
        # Check if episode is complete
        if self.current_ticket_idx >= len(self.tickets):
            self.done = True
        
        # Get next ticket
        next_ticket = self.tickets[self.current_ticket_idx] if self.current_ticket_idx < len(self.tickets) else None
        
        # Build observation
        observation = Observation(
            current_ticket=next_ticket,
            tickets_remaining=len(self.tickets) - self.current_ticket_idx,
            tickets_handled=self.tickets_handled,
            current_sla_status=self._check_sla(next_ticket) if next_ticket else "ok",
            recent_actions=self.recent_actions[-5:]  # Keep last 5 actions
        )
        
        return observation, reward, self.done, {"reward_breakdown": breakdown}
    
    def _check_sla(self, ticket: Ticket) -> str:
        """Check if ticket is within SLA"""
        if not ticket:
            return "ok"
            
        time_remaining = (ticket.sla_deadline - datetime.now()).total_seconds() / 3600
        
        if time_remaining < 0:
            return "breached"
        elif time_remaining < 1:  # Less than 1 hour
            return "warning"
        else:
            return "ok"
    
    def state(self) -> Dict:
        """Return current environment state"""
        return {
            "task_level": self.task_level,
            "tickets_handled": self.tickets_handled,
            "total_tickets": len(self.tickets),
            "recent_actions": self.recent_actions[-5:],
            "done": self.done
        }
