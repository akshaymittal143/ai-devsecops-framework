from typing import Dict, List
import logging
from openapi_spec_validator import validate_spec
import requests
from datetime import datetime
from .utils import generate_payload

logger = logging.getLogger(__name__)

class APISecurityValidator:
    """Advanced API security validation and fuzzing component"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.attack_patterns = self._load_attack_patterns()
        self.results_cache = {}
        
    def fuzz_endpoints(self, spec: Dict) -> List[Dict]:
        """Execute security fuzzing on API endpoints"""
        try:
            validate_spec(spec)
            results = []
            
            for path, methods in spec['paths'].items():
                for method, details in methods.items():
                    vulnerabilities = self._test_endpoint_security(
                        path=path,
                        method=method,
                        spec=details
                    )
                    results.extend(vulnerabilities)
                    
            return self._generate_security_report(results)
        except Exception as e:
            self.logger.error(f"Fuzzing failed: {str(e)}")
            raise
    
    def _test_endpoint_security(self, path: str, method: str, spec: Dict) -> List[Dict]:
        """Test individual endpoint for security vulnerabilities"""
        vulnerabilities = []
        test_cases = self._generate_test_cases(spec)
        
        for test in test_cases:
            try:
                response = self._execute_test(method, path, test)
                if security_issue := self._analyze_response(response, test):
                    vulnerabilities.append({
                        'endpoint': path,
                        'method': method,
                        'payload': test['payload'],
                        'vulnerability': security_issue['type'],
                        'severity': security_issue['severity'],
                        'timestamp': datetime.now().isoformat(),
                        'mitigation': self._suggest_mitigation(security_issue)
                    })
            except Exception as e:
                self.logger.warning(f"Test failed for {method} {path}: {str(e)}")
                
        return vulnerabilities
    
    def _generate_test_cases(self, spec: Dict) -> List[Dict]:
        """Generate intelligent test cases based on endpoint specification"""
        test_cases = []
        
        # Generate valid test cases
        test_cases.extend(self._generate_valid_cases(spec))
        
        # Generate security-focused test cases
        test_cases.extend(self._generate_attack_cases(spec))
        
        return test_cases
    
    def _suggest_mitigation(self, security_issue: Dict) -> str:
        """Suggest mitigation strategies for identified vulnerabilities"""
        mitigations = {
            'injection': 'Implement input validation and parameterized queries',
            'xss': 'Add Content-Security-Policy headers and encode outputs',
            'auth_bypass': 'Strengthen authentication mechanisms and session management',
            'rate_limit': 'Implement rate limiting and request throttling'
        }
        return mitigations.get(security_issue['type'], 'Review security controls')