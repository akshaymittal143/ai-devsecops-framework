from typing import Dict, List
import logging

class SecurityOrchestrator:
    def __init__(self, config: Dict):
        self.config = config
        self.components = {}
        self.logger = logging.getLogger(__name__)
        
    def register_component(self, component_id: str, capabilities: List[str]):
        self.components[component_id] = {
            'capabilities': capabilities,
            'status': 'active'
        }
        
    def handle_security_event(self, event: Dict):
        signals = self._collect_component_signals(event)
        if self._has_conflicts(signals):
            resolution = self._resolve_conflicts(signals)
            self.logger.info(f"Resolved conflict with strategy: {resolution['strategy']}")
            return resolution
        return signals[0]  # No conflicts
    
    def _collect_component_signals(self, event: Dict) -> List[Dict]:
        """Collect security signals from all registered components for the given event"""
        signals = []
        for component_id, component in self.components.items():
            if component['status'] == 'active':
                signal = self._simulate_component_signal(component_id, event)
                signals.append(signal)
        return signals

    def _simulate_component_signal(self, component_id: str, event: Dict) -> Dict:
        """Simulate the generation of a security signal from a component"""
        return {
            'source': component_id,
            'event': event,
            'confidence': 0.8,  # Simulated confidence level
            'proposed_action': 'alert'  # Simulated action
        }
    
    def _has_conflicts(self, signals: List[Dict]) -> bool:
        """Determine if there are conflicts among the collected signals"""
        actions = [signal['proposed_action'] for signal in signals]
        return len(set(actions)) > 1

    def _resolve_conflicts(self, signals: List[Dict]) -> Dict:
        """Resolve conflicts among signals using the configured strategy"""
        strategy = self.config.get('default_resolution_strategy', 'majority')
        if strategy == 'majority':
            return self._majority_resolution(signals)
        elif strategy == 'strict':
            return self._strictest_resolution(signals)
        elif strategy == 'confidence':
            return self._highest_confidence_resolution(signals)
        return signals[0]  # Default to first signal if no strategy matches

    def _majority_resolution(self, signals: List[Dict]) -> Dict:
        """Resolve conflicts by majority vote"""
        from collections import Counter
        action_counts = Counter(signal['proposed_action'] for signal in signals)
        most_common_action, _ = action_counts.most_common(1)[0]
        return next(signal for signal in signals if signal['proposed_action'] == most_common_action)

    def _strictest_resolution(self, signals: List[Dict]) -> Dict:
        """Resolve conflicts by taking the strictest action"""
        return min(signals, key=lambda s: s.get('confidence', 1))

    def _highest_confidence_resolution(self, signals: List[Dict]) -> Dict:
        """Resolve conflicts by selecting the signal with the highest confidence"""
        return max(signals, key=lambda s: s.get('confidence', 0))