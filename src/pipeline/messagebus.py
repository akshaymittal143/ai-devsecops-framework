import time
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class SecurityMessageBus:
    def __init__(self):
        self.subscribers = {}
        
    def publish_signal(self, signal: Dict) -> None:
        """Publish security signal to subscribers"""
        enriched = self._enrich_signal(signal)
        
        for subscriber in self._get_subscribers(signal['type']):
            try:
                subscriber.process_signal(enriched)
            except Exception as e:
                self._handle_delivery_failure(e, subscriber)
                
    def _enrich_signal(self, signal: Dict) -> Dict:
        """Enrich signal with metadata"""
        return {
            'timestamp': time.time(),
            'source': signal['source'],
            'confidence': signal.get('confidence', 1.0),
            'affected_services': self._resolve_affects(),
            'proposed_action': signal['action'],
            'context': self._gather_context()
        }
        
    def subscribe(self, signal_type: str, subscriber: object) -> None:
        """Register subscriber for signal type"""
        if signal_type not in self.subscribers:
            self.subscribers[signal_type] = []
        self.subscribers[signal_type].append(subscriber)