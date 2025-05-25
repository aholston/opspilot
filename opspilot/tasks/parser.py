"""
OpsPilot Task Parser Module
Interprets user tasks and builds appropriate context for AI agents.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage


class TaskCategory(Enum):
    """High-level task categories for operations teams"""
    INCIDENT_RESPONSE = "incident_response"
    TROUBLESHOOTING = "troubleshooting" 
    CODE_REVIEW = "code_review"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    SECURITY = "security"
    DOCUMENTATION = "documentation"
    ANALYSIS = "analysis"
    GENERAL = "general"


class TaskUrgency(Enum):
    """Task urgency levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ParsedTask:
    """Structured representation of a parsed task"""
    original_query: str
    category: TaskCategory
    urgency: TaskUrgency
    intent: str  # What the user wants to accomplish
    entities: Dict[str, List[str]]  # Extracted entities (services, errors, etc.)
    context_hints: List[str]  # What kind of docs/info would be helpful
    search_queries: List[str]  # Optimized queries for vector search
    constraints: Dict[str, Any]  # Time bounds, scope limits, etc.
    confidence: float  # How confident we are in the parsing


class TaskParser(ABC):
    """Abstract base for task parsers"""
    
    @abstractmethod
    def parse(self, task_description: str, additional_context: Optional[str] = None) -> ParsedTask:
        pass


class RuleBasedParser(TaskParser):
    """Fast rule-based parser for common patterns"""
    
    def __init__(self):
        # Category detection patterns
        self.category_patterns = {
            TaskCategory.INCIDENT_RESPONSE: [
                r'\b(incident|outage|down|emergency|critical|urgent|p[0-9])\b',
                r'\b(fire|escalat|page|alert|sev[0-9])\b'
            ],
            TaskCategory.TROUBLESHOOTING: [
                r'\b(debug|troubleshoot|investigate|diagnose|why|failing|broken)\b',
                r'\b(error|exception|timeout|connection|slow)\b'
            ],
            TaskCategory.CODE_REVIEW: [
                r'\b(review|pr|pull request|merge|code|diff)\b',
                r'\b(approve|feedback|comments|changes)\b'
            ],
            TaskCategory.DEPLOYMENT: [
                r'\b(deploy|release|rollout|launch|ship)\b',
                r'\b(staging|production|canary|blue.green)\b'
            ],
            TaskCategory.MONITORING: [
                r'\b(monitor|metrics|dashboard|alerts|observability)\b',
                r'\b(grafana|prometheus|datadog|newrelic)\b'
            ],
            TaskCategory.SECURITY: [
                r'\b(security|vulnerability|cve|patch|exploit)\b',
                r'\b(auth|permission|access|breach|compliance)\b'
            ],
            TaskCategory.DOCUMENTATION: [
                r'\b(document|readme|wiki|guide|howto|explain)\b',
                r'\b(onboard|training|knowledge|runbook)\b'
            ],
            TaskCategory.ANALYSIS: [
                r'\b(analyze|report|metrics|trends|performance)\b',
                r'\b(usage|cost|capacity|optimization)\b'
            ]
        }
        
        # Urgency detection patterns
        self.urgency_patterns = {
            TaskUrgency.CRITICAL: [
                r'\b(critical|emergency|urgent|asap|immediately)\b',
                r'\b(down|outage|sev[01]|p[01])\b'
            ],
            TaskUrgency.HIGH: [
                r'\b(high|important|priority|soon|blocking)\b',
                r'\b(sev2|p2|escalated)\b'
            ],
            TaskUrgency.MEDIUM: [
                r'\b(medium|normal|standard|regular)\b',
                r'\b(sev3|p3)\b'
            ],
            TaskUrgency.LOW: [
                r'\b(low|nice.to.have|when.possible|backlog)\b',
                r'\b(sev4|p4)\b'
            ]
        }
        
        # Entity extraction patterns
        self.entity_patterns = {
            'services': r'\b(api|service|microservice|backend|frontend|database|db|redis|kafka|nginx|apache)\b',
            'environments': r'\b(prod|production|staging|dev|development|test|qa|canary)\b',
            'technologies': r'\b(kubernetes|k8s|docker|aws|gcp|azure|terraform|ansible|jenkins|gitlab|github)\b',
            'error_types': r'\b(timeout|exception|error|failure|crash|hang|deadlock|memory|cpu|disk)\b',
            'time_refs': r'\b(today|yesterday|last\s+\w+|since\s+\w+|\d+\s*(?:hour|day|week|month)s?\s+ago)\b'
        }
    
    def parse(self, task_description: str, additional_context: Optional[str] = None) -> ParsedTask:
        """Parse task using rule-based patterns"""
        text = task_description.lower()
        if additional_context:
            text += " " + additional_context.lower()
        
        # Detect category
        category = self._detect_category(text)
        
        # Detect urgency
        urgency = self._detect_urgency(text)
        
        # Extract entities
        entities = self._extract_entities(text)
        
        # Generate intent
        intent = self._generate_intent(task_description, category, entities)
        
        # Generate context hints
        context_hints = self._generate_context_hints(category, entities)
        
        # Generate search queries
        search_queries = self._generate_search_queries(task_description, category, entities)
        
        # Extract constraints
        constraints = self._extract_constraints(text)
        
        return ParsedTask(
            original_query=task_description,
            category=category,
            urgency=urgency,
            intent=intent,
            entities=entities,
            context_hints=context_hints,
            search_queries=search_queries,
            constraints=constraints,
            confidence=0.8  # Rule-based has decent confidence
        )
    
    def _detect_category(self, text: str) -> TaskCategory:
        """Detect task category from patterns"""
        scores = {}
        
        for category, patterns in self.category_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text))
                score += matches
            scores[category] = score
        
        # Return category with highest score, or GENERAL if none
        if not scores or max(scores.values()) == 0:
            return TaskCategory.GENERAL
        
        return max(scores, key=scores.get)
    
    def _detect_urgency(self, text: str) -> TaskUrgency:
        """Detect urgency from patterns"""
        for urgency, patterns in self.urgency_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    return urgency
        
        return TaskUrgency.MEDIUM  # Default
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract relevant entities from text"""
        entities = {}
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                entities[entity_type] = list(set(matches))  # Remove duplicates
        
        return entities
    
    def _generate_intent(self, original_text: str, category: TaskCategory, entities: Dict[str, List[str]]) -> str:
        """Generate a clear intent statement"""
        if category == TaskCategory.INCIDENT_RESPONSE:
            return f"Respond to and resolve an incident or outage affecting system availability"
        elif category == TaskCategory.TROUBLESHOOTING:
            services = ", ".join(entities.get('services', ['the system']))
            return f"Troubleshoot and diagnose issues with {services}"
        elif category == TaskCategory.CODE_REVIEW:
            return "Review code changes for quality, security, and best practices"
        elif category == TaskCategory.DEPLOYMENT:
            envs = ", ".join(entities.get('environments', ['target environment']))
            return f"Plan and execute deployment to {envs}"
        elif category == TaskCategory.MONITORING:
            return "Set up monitoring, analyze metrics, or investigate alerts"
        elif category == TaskCategory.SECURITY:
            return "Address security concerns, vulnerabilities, or compliance requirements"
        elif category == TaskCategory.DOCUMENTATION:
            return "Create, update, or find documentation and knowledge resources"
        elif category == TaskCategory.ANALYSIS:
            return "Analyze system performance, usage patterns, or generate reports"
        else:
            return f"Assist with: {original_text}"
    
    def _generate_context_hints(self, category: TaskCategory, entities: Dict[str, List[str]]) -> List[str]:
        """Generate hints about what documentation would be helpful"""
        hints = []
        
        # Category-specific hints
        if category == TaskCategory.INCIDENT_RESPONSE:
            hints.extend(["runbooks", "incident procedures", "escalation guides", "monitoring dashboards"])
        elif category == TaskCategory.TROUBLESHOOTING:
            hints.extend(["troubleshooting guides", "error logs", "system architecture", "known issues"])
        elif category == TaskCategory.CODE_REVIEW:
            hints.extend(["coding standards", "security guidelines", "architecture decisions", "test requirements"])
        elif category == TaskCategory.DEPLOYMENT:
            hints.extend(["deployment procedures", "rollback plans", "environment configs", "release notes"])
        elif category == TaskCategory.MONITORING:
            hints.extend(["monitoring setup", "alert configurations", "metric definitions", "dashboard configs"])
        elif category == TaskCategory.SECURITY:
            hints.extend(["security policies", "vulnerability reports", "compliance docs", "access controls"])
        elif category == TaskCategory.DOCUMENTATION:
            hints.extend(["existing documentation", "templates", "style guides", "knowledge base"])
        elif category == TaskCategory.ANALYSIS:
            hints.extend(["metrics documentation", "reporting templates", "data schemas", "analysis tools"])
        
        # Entity-specific hints
        for service in entities.get('services', []):
            hints.append(f"{service} documentation")
            hints.append(f"{service} configuration")
        
        for tech in entities.get('technologies', []):
            hints.append(f"{tech} guides")
        
        return list(set(hints))  # Remove duplicates
    
    def _generate_search_queries(self, original_text: str, category: TaskCategory, entities: Dict[str, List[str]]) -> List[str]:
        """Generate optimized search queries for vector retrieval"""
        queries = []
        
        # Always include the original text (cleaned)
        clean_original = re.sub(r'\b(please|help|can you|could you|i need)\b', '', original_text.lower()).strip()
        if clean_original:
            queries.append(clean_original)
        
        # Category-specific queries
        if category == TaskCategory.INCIDENT_RESPONSE:
            queries.extend([
                "incident response procedures",
                "emergency escalation",
                "system outage troubleshooting"
            ])
        elif category == TaskCategory.TROUBLESHOOTING:
            queries.extend([
                "troubleshooting guide",
                "error diagnosis",
                "system debugging"
            ])
        elif category == TaskCategory.CODE_REVIEW:
            queries.extend([
                "code review guidelines",
                "pull request process",
                "code quality standards"
            ])
        
        # Entity-enhanced queries
        for service in entities.get('services', []):
            queries.append(f"{service} troubleshooting")
            queries.append(f"{service} configuration")
        
        for error_type in entities.get('error_types', []):
            queries.append(f"{error_type} resolution")
        
        # Combine entities for more specific queries
        services = entities.get('services', [])
        error_types = entities.get('error_types', [])
        if services and error_types:
            for service in services[:2]:  # Limit combinations
                for error in error_types[:2]:
                    queries.append(f"{service} {error}")
        
        return queries[:8]  # Limit to 8 queries max
    
    def _extract_constraints(self, text: str) -> Dict[str, Any]:
        """Extract time, scope, and other constraints"""
        constraints = {}
        
        # Time constraints
        time_matches = re.findall(r'\b(?:within|in|by)\s+(\d+)\s*(hour|day|week|month)s?\b', text)
        if time_matches:
            num, unit = time_matches[0]
            constraints['time_limit'] = f"{num} {unit}s"
        
        # Environment constraints
        env_matches = re.findall(r'\b(prod|production|staging|dev|development)\s+only\b', text)
        if env_matches:
            constraints['environment'] = env_matches[0]
        
        # Priority constraints
        if 'read.only' in text or 'no changes' in text:
            constraints['read_only'] = True
        
        return constraints


class LLMTaskParser(TaskParser):
    """LLM-powered parser for complex task understanding"""
    
    def __init__(self, 
                 provider: str = "openai",
                 model: str = "gpt-4",
                 api_key: Optional[str] = None):
        
        if provider == "openai":
            self.llm = ChatOpenAI(
                model=model,
                temperature=0.1,
                openai_api_key=api_key
            )
        elif provider == "anthropic":
            self.llm = ChatAnthropic(
                model=model,
                temperature=0.1,
                anthropic_api_key=api_key
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        self.system_prompt = """You are an expert at parsing operational tasks for DevOps, SRE, and SecOps teams.

Analyze the user's task and respond with a JSON object containing:
{
  "category": "incident_response|troubleshooting|code_review|deployment|monitoring|security|documentation|analysis|general",
  "urgency": "low|medium|high|critical",
  "intent": "clear statement of what the user wants to accomplish",
  "entities": {
    "services": ["list of services/systems mentioned"],
    "environments": ["prod", "staging", etc.],
    "technologies": ["kubernetes", "docker", etc.],
    "error_types": ["timeout", "exception", etc.],
    "time_refs": ["today", "last week", etc.]
  },
  "context_hints": ["types of documentation that would be helpful"],
  "search_queries": ["optimized queries for finding relevant docs"],
  "constraints": {"any limitations or requirements"},
  "confidence": 0.95
}

Focus on operational context. Be specific about what documentation would help."""
    
    def parse(self, task_description: str, additional_context: Optional[str] = None) -> ParsedTask:
        """Parse task using LLM"""
        full_context = task_description
        if additional_context:
            full_context += f"\n\nAdditional context: {additional_context}"
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Task: {full_context}")
        ]
        
        try:
            response = self.llm(messages)
            
            # Parse JSON response
            import json
            parsed_data = json.loads(response.content)
            
            return ParsedTask(
                original_query=task_description,
                category=TaskCategory(parsed_data["category"]),
                urgency=TaskUrgency(parsed_data["urgency"]),
                intent=parsed_data["intent"],
                entities=parsed_data["entities"],
                context_hints=parsed_data["context_hints"],
                search_queries=parsed_data["search_queries"],
                constraints=parsed_data["constraints"],
                confidence=parsed_data["confidence"]
            )
            
        except Exception as e:
            print(f"LLM parsing failed: {e}")
            # Fallback to rule-based parsing
            fallback_parser = RuleBasedParser()
            result = fallback_parser.parse(task_description, additional_context)
            result.confidence = 0.3  # Lower confidence for fallback
            return result


class HybridTaskParser(TaskParser):
    """Combines rule-based and LLM parsing for best results"""
    
    def __init__(self, 
                 use_llm: bool = True,
                 llm_provider: str = "openai",
                 llm_model: str = "gpt-4",
                 api_key: Optional[str] = None):
        
        self.rule_parser = RuleBasedParser()
        self.llm_parser = None
        
        if use_llm:
            try:
                self.llm_parser = LLMTaskParser(llm_provider, llm_model, api_key)
            except Exception as e:
                print(f"Warning: LLM parser unavailable: {e}")
    
    def parse(self, task_description: str, additional_context: Optional[str] = None) -> ParsedTask:
        """Parse using hybrid approach"""
        
        # Start with rule-based parsing
        rule_result = self.rule_parser.parse(task_description, additional_context)
        
        # Enhance with LLM if available
        if self.llm_parser:
            try:
                llm_result = self.llm_parser.parse(task_description, additional_context)
                
                # Use LLM result if it has higher confidence
                if llm_result.confidence > rule_result.confidence:
                    # But merge in rule-based entities that might be missed
                    for entity_type, entities in rule_result.entities.items():
                        if entity_type not in llm_result.entities:
                            llm_result.entities[entity_type] = entities
                        else:
                            # Merge entity lists
                            combined = list(set(llm_result.entities[entity_type] + entities))
                            llm_result.entities[entity_type] = combined
                    
                    return llm_result
                    
            except Exception as e:
                print(f"LLM parsing failed, using rule-based: {e}")
        
        return rule_result


# Usage example
if __name__ == "__main__":
    parser = HybridTaskParser(use_llm=False)  # Set to True with API key
    
    test_tasks = [
        "Help me troubleshoot the API gateway timeout errors in production",
        "Review the PR for the new authentication service",
        "The Kubernetes cluster is down and users can't access the app",
        "Generate a report on database performance over the last month",
        "Document the deployment process for new team members"
    ]
    
    for task in test_tasks:
        print(f"Task: {task}")
        result = parser.parse(task)
        print(f"Category: {result.category.value}")
        print(f"Urgency: {result.urgency.value}")
        print(f"Intent: {result.intent}")
        print(f"Entities: {result.entities}")
        print(f"Search queries: {result.search_queries}")
        print(f"Confidence: {result.confidence}")
        print("---")