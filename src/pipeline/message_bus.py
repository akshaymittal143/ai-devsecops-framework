from typing import Dict
import threading
from queue import Queue
import logging
import time

class SecurityMessageBus:
    def __init__(self):
        self.subscribers = {}
        self.queue = Queue()
        self.logger = logging.getLogger(__name__)
        self._start_worker()
        
    def _start_worker(self):
        """Starts the message processing worker thread"""
        def worker():
            while True:
                message = self.queue.get()
                self._process_message(message)
                self.queue.task_done()
                
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
    
    def _process_message(self, message: Dict):
        """Processes incoming security messages"""
        topic = message.get('topic', 'default')
        if topic in self.subscribers:
            for callback in self.subscribers[topic]:
                try:
                    callback(message)
                except Exception as e:
                    self.logger.error(f"Error processing message: {e}")
        
    def publish(self, signal: Dict):
        enriched = self._enrich_signal(signal)
        self.queue.put(enriched)
        
    def subscribe(self, topic: str, callback):
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)
        
    def _enrich_signal(self, signal: Dict) -> Dict:
        return {
            'timestamp': time.time(),
            'source': signal['source'],
            'context': self._gather_context()
        }
    
    def _gather_context(self):
        # Placeholder for context gathering logic
        return {}