# AI-Augmented DevSecOps Framework - API Reference

## Overview

This document provides comprehensive API reference for the AI-Augmented DevSecOps Framework components mentioned in the IEEE SOSE 2025 paper.

## Core Components

### 1. LSTM Threat Detection Engine

#### `LSTMDetector`

Real-time threat detection using LSTM neural networks with 64-unit architecture.

```python
from src.detection.lstm_detector import LSTMDetector

detector = LSTMDetector(
    units=64,
    window_size=10,
    threshold=0.95
)
```

**Methods:**

- `train(data, epochs=100)` - Train the LSTM model
- `predict(sequence)` - Predict threat probability
- `detect_anomaly(data)` - Real-time anomaly detection
- `get_model_metrics()` - Return performance metrics

**Performance:**
- Latency: Sub-2 seconds at 10k events/sec
- Accuracy: 95% detection rate
- False Positive Rate: <5%

### 2. Security Orchestrator

#### `SecurityOrchestrator`

Manages component interactions and policy enforcement through message bus architecture.

```python
from src.orchestration.orchestrator import SecurityOrchestrator, ConflictResolutionStrategy

orchestrator = SecurityOrchestrator(
    strategy=ConflictResolutionStrategy.CONFIDENCE_WEIGHTED
)
```

**Conflict Resolution Strategies:**
- `CONSERVATIVE` - Apply strictest security measure
- `MAJORITY_BASED` - Follow majority component decision  
- `CONFIDENCE_WEIGHTED` - Use highest confidence signal

**Methods:**

- `process_security_signal(signal)` - Process incoming security signals
- `get_active_signals()` - Get currently active security signals
- `clear_signals()` - Clear all active signals
- `get_statistics()` - Get orchestrator statistics

### 3. Zero-Day Attack Detection

#### `ZeroDayDetector`

Behavioral analysis for detecting previously unseen attack patterns.

```python
from src.detection.zero_day_detector import ZeroDayDetector

detector = ZeroDayDetector()
detector.train_baseline(benign_data)
result = detector.analyze_behavior(sample)
```

**Performance Metrics:**
- Detection Rate: 85% for unseen patterns
- False Positive Rate: 3%
- Average Detection Time: 1.8 seconds

### 4. API Security Validator

#### `APISecurityValidator`

Intelligent fuzzing and security testing for REST APIs.

```python
from src.api.security_validator import APISecurityValidator

validator = APISecurityValidator()
results = validator.validate_api(endpoint_url, openapi_spec)
```

**Capabilities:**
- OpenAPI specification support
- Custom test case generation
- BOLA vulnerability detection
- Excessive data exposure identification

### 5. Business Logic Validator

#### `BusinessLogicValidator`

Detection of business logic vulnerabilities and workflow violations.

```python
from src.validation.business_logic_validator import BusinessLogicValidator

validator = BusinessLogicValidator()
results = validator.validate_workflow(workflow_data)
```

**Detection Methods:**
- Race Conditions: 92% success rate
- State Violations: 89% success rate  
- Logic Bypasses: 87% success rate

## Configuration

### Settings Configuration

```yaml
# config/settings.yaml
detection:
  model:
    type: lstm
    units: 64
    window_size: 10
  threshold:
    anomaly: 0.95
    confidence: 0.85

orchestration:
  strategy: confidence-weighted
  retry_attempts: 3
  timeout: 30s

monitoring:
  prometheus:
    scrape_interval: 15s
  logging:
    level: info
    retention: 30d
```

### Policy Configuration

Network policies are automatically generated and stored in `config/policies/`:

- `network-policies.yaml` - Kubernetes NetworkPolicy templates
- `security-rules.yaml` - Custom security rule definitions
- `alert-rules.yaml` - Prometheus alert configurations

## Deployment

### Kubernetes Deployment

```bash
# Deploy using Helm
helm install ai-devsecops ./helm/ai-devsecops-framework

# Deploy using kubectl
kubectl apply -f deploy/kubernetes/
```

### Docker Compose

```bash
# Development environment
docker-compose -f docker-compose.dev.yml up

# Production environment  
docker-compose up -d
```

## Monitoring & Metrics

### Prometheus Metrics

The framework exposes the following metrics:

- `ai_devsecops_threats_detected_total` - Total threats detected
- `ai_devsecops_detection_latency_seconds` - Detection latency histogram
- `ai_devsecops_model_accuracy` - Current model accuracy
- `ai_devsecops_false_positives_total` - False positive count

### Performance Benchmarks

Based on evaluation against baseline systems:

| System | Precision | Recall | Latency(s) | Throughput(eps) |
|--------|-----------|--------|------------|-----------------|
| **Our Framework** | **0.95±0.02** | **0.94±0.03** | **1.5±0.2** | **10K** |
| Suricata | 0.88±0.04 | 0.85±0.05 | 2.3±0.3 | 8K |
| Prisma Cloud | 0.92±0.02 | 0.89±0.03 | 1.9±0.3 | 9K |

## Examples

### Basic Usage

```python
from src.main import AIDevSecOpsFramework

# Initialize framework
framework = AIDevSecOpsFramework()

# Start monitoring
framework.start_monitoring()

# Process security event
event = {
    "timestamp": time.time(),
    "source_ip": "192.168.1.100",
    "event_type": "api_call",
    "metadata": {...}
}

result = framework.process_event(event)
print(f"Threat detected: {result.is_threat}")
```

### Advanced Configuration

```python
from src.detection.lstm_detector import LSTMDetector
from src.orchestration.orchestrator import SecurityOrchestrator
from src.pipeline.policy_engine import PolicyEngine

# Custom detector configuration
detector = LSTMDetector(
    units=128,  # Increased capacity
    window_size=15,
    dropout_rate=0.3
)

# Custom orchestrator
orchestrator = SecurityOrchestrator(
    strategy=ConflictResolutionStrategy.CONSERVATIVE
)

# Policy engine with custom rules
policy_engine = PolicyEngine(
    enforcement_mode="strict",
    auto_remediation=True
)
```

## Error Handling

### Common Exceptions

- `ModelNotTrainedException` - Raised when using untrained models
- `InvalidConfigurationException` - Configuration validation errors
- `PolicyEnforcementException` - Policy application failures
- `DetectionTimeoutException` - Detection timeout exceeded

### Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('ai_devsecops')
```

## Integration

### CI/CD Integration

```yaml
# Jenkins pipeline integration
stages:
  - name: Security Scan
    script: |
      python -m src.cli scan --target ./app
      python -m src.cli validate --policies ./policies
```

### Kubernetes Integration

```yaml
# Custom Resource Definition
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: securitypolicies.ai-devsecops.io
spec:
  group: ai-devsecops.io
  versions:
  - name: v1
    served: true
    storage: true
```

## Support

For issues and questions:

- GitHub Issues: https://github.com/akshaymittal143/ai-devsecops-framework/issues
- Documentation: https://github.com/akshaymittal143/ai-devsecops-framework/docs
- Paper Reference: IEEE SOSE 2025 - "AI-Augmented DevSecOps Pipelines"

## License

MIT License - See LICENSE file for details. 