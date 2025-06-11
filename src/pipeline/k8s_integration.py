from kubernetes import client, config
from typing import Dict
import yaml
import logging

logger = logging.getLogger(__name__)

class KubernetesIntegration:
    def __init__(self):
        config.load_incluster_config()
        self.v1 = client.CoreV1Api()
        self.networking = client.NetworkingV1Api()
        
    def apply_network_policy(self, policy_spec: Dict) -> None:
        """Apply NetworkPolicy to cluster"""
        try:
            self.networking.create_namespaced_network_policy(
                namespace=policy_spec['metadata']['namespace'],
                body=policy_spec
            )
            logger.info(f"Applied NetworkPolicy: {policy_spec['metadata']['name']}")
        except Exception as e:
            logger.error(f"Failed to apply NetworkPolicy: {str(e)}")
            raise

    def isolate_pod(self, pod_name: str, namespace: str) -> None:
        """Create isolation policy for suspicious pod"""
        policy = {
            'apiVersion': 'networking.k8s.io/v1',
            'kind': 'NetworkPolicy',
            'metadata': {
                'name': f'isolate-{pod_name}',
                'namespace': namespace
            },
            'spec': {
                'podSelector': {
                    'matchLabels': {
                        'app': pod_name
                    }
                },
                'policyTypes': ['Egress'],
                'egress': []
            }
        }
        self.apply_network_policy(policy)