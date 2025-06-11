import yaml
import logging
from typing import Dict, List
from kubernetes import client, config
from .detection_engine import DetectionEngine

logger = logging.getLogger(__name__)

class SecurityGate:
    """Security gate for CI/CD pipeline"""
    
    def __init__(self):
        self.detector = DetectionEngine()
        self.risk_threshold = 0.8
        
    def evaluate_deployment(self, manifest_path: str) -> Dict:
        """Evaluate Kubernetes deployment security"""
        try:
            with open(manifest_path) as f:
                deploy = yaml.safe_load(f)
            
            risks = self._check_security_risks(deploy)
            score = self._calculate_risk_score(risks)
            
            return {
                "pass": score < self.risk_threshold,
                "score": score,
                "risks": risks
            }
        except Exception as e:
            logger.error(f"Security gate failed: {str(e)}")
            return {
                "pass": False,
                "error": str(e)
            }
    
    def _check_security_risks(self, deployment: Dict) -> List[Dict]:
        """Check for security risks in deployment"""
        risks = []
        
        # Check for privileged containers
        containers = deployment.get('spec', {}).get('template', {}).get('spec', {}).get('containers', [])
        for container in containers:
            if container.get('securityContext', {}).get('privileged'):
                risks.append({
                    "severity": "HIGH",
                    "message": f"Privileged container found: {container['name']}"
                })
        
        return risks
    
    def _calculate_risk_score(self, risks: List[Dict]) -> float:
        """Calculate overall risk score"""
        severity_weights = {
            "HIGH": 1.0,
            "MEDIUM": 0.5,
            "LOW": 0.2
        }
        
        total_weight = sum(severity_weights[risk['severity']] for risk in risks)
        return min(total_weight / 3.0, 1.0)  # Normalize to [0,1]