import click
import logging
from pathlib import Path
from .telemetry.collector import TelemetryCollector
from .detection.engine import DetectionEngine
from .pipeline.security_gate import SecurityGate

@click.group()
def cli():
    """AI-Augmented DevSecOps Framework CLI"""
    pass

@cli.command()
@click.option('--namespace', default='default', help='Kubernetes namespace')
def monitor(namespace):
    """Start telemetry monitoring"""
    collector = TelemetryCollector()
    click.echo(f"Starting telemetry collection for namespace: {namespace}")
    result = collector.collect_pod_metrics()
    click.echo(f"Collection status: {result['status']}")

@cli.command()
@click.argument('manifest_path', type=click.Path(exists=True))
def evaluate(manifest_path):
    """Evaluate deployment security"""
    gate = SecurityGate()
    result = gate.evaluate_deployment(manifest_path)
    
    if result['pass']:
        click.echo(click.style("✓ Security check passed", fg='green'))
    else:
        click.echo(click.style("✗ Security check failed", fg='red'))
    
    click.echo(f"\nRisk Score: {result['score']:.2f}")
    
    if 'risks' in result:
        click.echo("\nIdentified Risks:")
        for risk in result['risks']:
            click.echo(f"- [{risk['severity']}] {risk['message']}")

if __name__ == '__main__':
    cli()