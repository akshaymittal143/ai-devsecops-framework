from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class SecurityOrchestrator:
    def __init__(self, config: Dict):
        self.config = config
        self.components = {}
        self.message_bus = SecurityMessageBus()
        
    def resolve_conflicts(self, signals: Dict) -> Dict:
        """Resolve conflicting security signals"""
        weighted_signals = {
            src: self._calculate_confidence(signal)
            for src, signal in signals.items()
        }
        
        if self._has_contradictions(weighted_signals):
            return self._apply_resolution_policy(
                weighted_signals,
                self.config['resolution_strategy']
            )
            
        return self._merge_signals(weighted_signals)
    
    def _calculate_confidence(self, signal: Dict) -> float:
        """Calculate confidence score for a signal"""
        base_confidence = signal.get('confidence', 0.5)
        source_weight = self.config['source_weights'].get(
            signal['source'], 1.0
        )
        return base_confidence * source_weight

    def _has_contradictions(self, signals: Dict) -> bool:
        """Check for contradicting security signals"""
        actions = [s['proposed_action'] for s in signals.values()]
        return len(set(actions)) > 1

    def _apply_resolution_policy(self, signals: Dict, strategy: str) -> Dict:
        """Apply configured resolution strategy"""
        policies = {
            'conservative': self._take_strictest_action,
            'majority': self._take_majority_vote,
            'confidence': self._highest_confidence
        }
        return policies[strategy](signals)