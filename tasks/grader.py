"""Graders for each difficulty level"""

from typing import List, Dict
from src.customer_support_env import Ticket, Priority, TicketCategory

class TaskGrader:
    """Programmatic graders for each task difficulty"""
    
    @staticmethod
    def grade_easy(agent_actions: List[Dict], tickets: List[Ticket]) -> float:
        """
        Grade easy task: correct categorization only
        
        Returns score 0.0 - 1.0
        """
        if not tickets:
            return 0.0
            
        correct = 0
        for i, action in enumerate(agent_actions):
            if i >= len(tickets):
                break
                
            expected_category = tickets[i].category.value
            if action.get("categorization") == expected_category:
                correct += 1
        
        return correct / len(tickets)
    
    @staticmethod
    def grade_medium(agent_actions: List[Dict], tickets: List[Ticket]) -> float:
        """
        Grade medium task: categorization + priority with SLA
        
        Returns score 0.0 - 1.0
        """
        if not tickets:
            return 0.0
            
        category_score = 0
        priority_score = 0
        
        for i, action in enumerate(agent_actions):
            if i >= len(tickets):
                break
                
            # Category accuracy (50% weight)
            if action.get("categorization") == tickets[i].category.value:
                category_score += 1
            
            # Priority accuracy (50% weight)
            expected_priority = TaskGrader._get_expected_priority(tickets[i])
            if action.get("priority") == expected_priority.value:
                priority_score += 1
        
        category_accuracy = category_score / len(tickets)
        priority_accuracy = priority_score / len(tickets)
        
        # Weighted average
        return (category_accuracy * 0.5) + (priority_accuracy * 0.5)
    
    @staticmethod
    def grade_hard(agent_actions: List[Dict], tickets: List[Ticket], 
                   knowledge_base: Dict[str, str]) -> float:
        """
        Grade hard task: full resolution with KB and escalation
        
        Returns score 0.0 - 1.0
        """
        if not tickets:
            return 0.0
            
        resolution_score = 0
        escalation_score = 0
        sentiment_recovery_score = 0
        
        for i, action in enumerate(agent_actions):
            if i >= len(tickets):
                break
                
            ticket = tickets[i]
            
            # Check resolution quality
            if action.get("resolution"):
                # Check if resolution aligns with knowledge base
                kb_solution = knowledge_base.get(ticket.category.value, "")
                if any(word in action["resolution"].lower() for word in kb_solution.split()):
                    resolution_score += 1
                
                # Check sentiment recovery
                if ticket.sentiment < -0.5 and action.get("sentiment_recovered", False):
                    sentiment_recovery_score += 1
            
            # Check escalation appropriateness
            should_escalate = ticket.sentiment < -0.5 or ticket.category == TicketCategory.COMPLAINT
            if action.get("escalated", False) == should_escalate:
                escalation_score += 1
        
        resolution_accuracy = resolution_score / len(tickets)
        escalation_accuracy = escalation_score / len(tickets)
        sentiment_recovery = sentiment_recovery_score / len(tickets)
        
        # Weighted combination
        return (resolution_accuracy * 0.5) + (escalation_accuracy * 0.3) + (sentiment_recovery * 0.2)
    
    @staticmethod
    def _get_expected_priority(ticket: Ticket) -> Priority:
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
