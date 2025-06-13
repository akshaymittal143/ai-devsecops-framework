"""
Security Orchestrator - Manages component interactions and policy enforcement
through a message bus architecture with conflict resolution strategies.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

class ConflictResolutionStrategy(Enum):
    CONSERVATIVE = "conservative"
    MAJORITY_BASED = "majority"
    CONFIDENCE_WEIGHTED = "confidence"

@dataclass
class SecuritySignal:
    component: str
    threat_type: str
    confidence: float
    severity: str
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class PolicyAction:
    action_type: str
    target: str
    parameters: Dict[str, Any]
    confidence: float

class SecurityOrchestrator:
    """
    Orchestrates security components and resolves conflicts between
    different security signals using configurable strategies.
    """
    
    def __init__(self, strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.CONFIDENCE_WEIGHTED):
        self.strategy = strategy
        self.active_signals: List[SecuritySignal] = []
        self.policy_actions: List[PolicyAction] = []
        self.component_weights = {
            "lstm_detector": 0.8,
            "zero_day_detector": 0.9,
            "api_validator": 0.7,
            "business_logic_validator": 0.6
        }
    
    async def process_security_signal(self, signal: SecuritySignal) -> Optional[PolicyAction]:
        """Process incoming security signal and determine appropriate action."""
        logger.info(f"Processing security signal from {signal.component}: {signal.threat_type}")
        
        self.active_signals.append(signal)
        
        # Apply conflict resolution strategy
        action = await self._resolve_conflicts()
        
        if action:
            self.policy_actions.append(action)
            await self._execute_policy_action(action)
        
        return action
    
    async def _resolve_conflicts(self) -> Optional[PolicyAction]:
        """Resolve conflicts between multiple security signals."""
        if not self.active_signals:
            return None
        
        if self.strategy == ConflictResolutionStrategy.CONSERVATIVE:
            return await self._conservative_resolution()
        elif self.strategy == ConflictResolutionStrategy.MAJORITY_BASED:
            return await self._majority_resolution()
        elif self.strategy == ConflictResolutionStrategy.CONFIDENCE_WEIGHTED:
            return await self._confidence_weighted_resolution()
        
        return None
    
    async def _conservative_resolution(self) -> Optional[PolicyAction]:
        """Apply the strictest security measure."""
        highest_severity = max(self.active_signals, key=lambda s: self._severity_score(s.severity))
        
        return PolicyAction(
            action_type="isolate_pod",
            target=highest_severity.metadata.get("pod_name", "unknown"),
            parameters={"reason": f"Conservative policy: {highest_severity.threat_type}"},
            confidence=highest_severity.confidence
        )
    
    async def _majority_resolution(self) -> Optional[PolicyAction]:
        """Follow majority component decision."""
        threat_votes = {}
        for signal in self.active_signals:
            threat_votes[signal.threat_type] = threat_votes.get(signal.threat_type, 0) + 1
        
        if not threat_votes:
            return None
        
        majority_threat = max(threat_votes, key=threat_votes.get)
        majority_signals = [s for s in self.active_signals if s.threat_type == majority_threat]
        avg_confidence = sum(s.confidence for s in majority_signals) / len(majority_signals)
        
        return PolicyAction(
            action_type="restrict_network",
            target=majority_signals[0].metadata.get("service_name", "unknown"),
            parameters={"threat_type": majority_threat, "votes": threat_votes[majority_threat]},
            confidence=avg_confidence
        )
    
    async def _confidence_weighted_resolution(self) -> Optional[PolicyAction]:
        """Use highest confidence signal with component weighting."""
        if not self.active_signals:
            return None
        
        # Calculate weighted confidence scores
        weighted_signals = []
        for signal in self.active_signals:
            component_weight = self.component_weights.get(signal.component, 0.5)
            weighted_confidence = signal.confidence * component_weight
            weighted_signals.append((signal, weighted_confidence))
        
        # Select signal with highest weighted confidence
        best_signal, best_score = max(weighted_signals, key=lambda x: x[1])
        
        # Determine action based on threat type and confidence
        if best_score > 0.8:
            action_type = "isolate_pod"
        elif best_score > 0.6:
            action_type = "restrict_network"
        else:
            action_type = "monitor_enhanced"
        
        return PolicyAction(
            action_type=action_type,
            target=best_signal.metadata.get("target", "unknown"),
            parameters={
                "threat_type": best_signal.threat_type,
                "weighted_confidence": best_score,
                "original_confidence": best_signal.confidence,
                "component": best_signal.component
            },
            confidence=best_score
        )
    
    def _severity_score(self, severity: str) -> int:
        """Convert severity string to numeric score."""
        severity_map = {
            "low": 1,
            "medium": 2,
            "high": 3,
            "critical": 4
        }
        return severity_map.get(severity.lower(), 0)
    
    async def _execute_policy_action(self, action: PolicyAction):
        """Execute the determined policy action."""
        logger.info(f"Executing policy action: {action.action_type} on {action.target}")
        
        try:
            if action.action_type == "isolate_pod":
                await self._isolate_pod(action.target, action.parameters)
            elif action.action_type == "restrict_network":
                await self._restrict_network(action.target, action.parameters)
            elif action.action_type == "monitor_enhanced":
                await self._enhance_monitoring(action.target, action.parameters)
            
            logger.info(f"Policy action {action.action_type} executed successfully")
            
        except Exception as e:
            logger.error(f"Failed to execute policy action {action.action_type}: {e}")
    
    async def _isolate_pod(self, target: str, parameters: Dict[str, Any]):
        """Isolate a pod by applying network policies."""
        # Implementation would integrate with Kubernetes API
        logger.info(f"Isolating pod {target}: {parameters.get('reason', 'Security threat detected')}")
    
    async def _restrict_network(self, target: str, parameters: Dict[str, Any]):
        """Restrict network access for a service."""
        logger.info(f"Restricting network access for {target}: {parameters.get('threat_type', 'Unknown threat')}")
    
    async def _enhance_monitoring(self, target: str, parameters: Dict[str, Any]):
        """Enhance monitoring for suspicious activity."""
        logger.info(f"Enhancing monitoring for {target}: confidence {parameters.get('weighted_confidence', 0)}")
    
    def get_active_signals(self) -> List[SecuritySignal]:
        """Get currently active security signals."""
        return self.active_signals.copy()
    
    def clear_signals(self):
        """Clear all active signals."""
        self.active_signals.clear()
        logger.info("Cleared all active security signals")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            "active_signals": len(self.active_signals),
            "total_actions": len(self.policy_actions),
            "strategy": self.strategy.value,
            "component_weights": self.component_weights
        } 