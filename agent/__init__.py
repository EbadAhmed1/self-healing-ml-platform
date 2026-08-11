"""
agent
─────
Autonomous rule-based diagnosis agent and LLM reasoning package.

Modules:
  db_models   — ORM models for `deployments`, `incidents`, and `llm_usage` tables
  evidence    — EvidenceGatherer querying deployments, simulation events, drift & accuracy
  diagnosis   — Rule-based DiagnosisEngine with priority rules & confidence scores
  remediation — RemediationEngine with confidence gates & pointer file rollback
  dedup       — Alert deduplication helper (15m window)
  llm         — LLMReasoningEngine for escalated incident enrichment via Groq API
  run_agent   — Standalone CLI runner service
"""

from agent.db_models import Deployment, Incident, LLMUsage
from agent.dedup import attach_duplicate_alert, find_duplicate_incident
from agent.diagnosis import DiagnosisEngine, DiagnosisResult
from agent.evidence import EvidenceGatherer
from agent.llm import LLMDiagnosisResponse, LLMReasoningEngine
from agent.remediation import RemediationEngine
from agent.run_agent import process_unprocessed_alerts, run_agent_loop

__all__ = [
    "Deployment",
    "DiagnosisEngine",
    "DiagnosisResult",
    "EvidenceGatherer",
    "Incident",
    "LLMDiagnosisResponse",
    "LLMReasoningEngine",
    "LLMUsage",
    "RemediationEngine",
    "attach_duplicate_alert",
    "find_duplicate_incident",
    "process_unprocessed_alerts",
    "run_agent_loop",
]
