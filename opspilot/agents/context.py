"""
OpsPilot Context Constructor
Builds specialized system prompts and retrieval context for AI agents.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from opspilot.tasks.parser import ParsedTask, TaskCategory
from opspilot.storage.vector_store import VectorStore
from opspilot.embedding.chunker import Chunk


@dataclass 
class AgentContext:
    """Complete context package for an AI agent"""
    system_prompt: str
    task_context: str
    relevant_docs: List[Tuple[Chunk, float]]
    search_summary: str
    constraints: Dict[str, Any]
    metadata: Dict[str, Any]


class ContextConstructor:
    """Builds specialized context for different types of operational tasks"""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        
        # Base system prompts for different task categories
        self.base_prompts = {
            TaskCategory.INCIDENT_RESPONSE: """You are an expert Site Reliability Engineer (SRE) specializing in incident response. Your role is to help resolve system outages and critical issues quickly and effectively.

PRIORITIES:
1. Restore service availability immediately
2. Minimize customer impact
3. Follow established incident procedures
4. Document actions for post-mortem

APPROACH:
- Start with the most likely causes based on symptoms
- Use monitoring data and logs to guide investigation
- Escalate appropriately when needed
- Keep stakeholders informed with clear status updates

GUIDELINES:
- Be decisive but methodical
- Focus on restoration first, root cause analysis second
- Consider rollback options early
- Always check for similar past incidents""",

            TaskCategory.TROUBLESHOOTING: """You are an expert Systems Engineer with deep knowledge of distributed systems, infrastructure, and application debugging.

APPROACH:
- Start with systematic problem isolation
- Use data-driven investigation methods
- Consider the full system context
- Apply the scientific method to hypothesis testing

METHODOLOGY:
1. Gather symptoms and reproduce the issue
2. Form hypotheses based on system knowledge
3. Test hypotheses with minimal impact
4. Document findings and solutions

GUIDELINES:
- Always consider recent changes as potential causes
- Use logs, metrics, and traces systematically
- Think about dependencies and upstream/downstream effects
- Provide actionable recommendations""",

            TaskCategory.CODE_REVIEW: """You are a senior software engineer and architect with expertise in code quality, security, and best practices.

FOCUS AREAS:
- Code correctness and logic
- Security vulnerabilities and best practices
- Performance implications
- Maintainability and readability
- Architecture and design patterns
- Testing coverage and quality

REVIEW APPROACH:
- Understand the business context and requirements
- Check for potential edge cases and error handling
- Verify compliance with team standards
- Suggest improvements with clear rationale
- Balance perfectionism with practicality

GUIDELINES:
- Be constructive and educational in feedback
- Prioritize security and correctness issues
- Consider the skill level of the author
- Suggest specific improvements, not just problems""",

            TaskCategory.DEPLOYMENT: """You are a DevOps Engineer with expertise in continuous integration, deployment strategies, and infrastructure management.

DEPLOYMENT PHASES:
1. Pre-deployment validation
2. Deployment execution
3. Post-deployment verification
4. Rollback planning

CONSIDERATIONS:
- Environment-specific configurations
- Database migrations and dependencies
- Feature flags and gradual rollouts
- Monitoring and alerting setup
- Rollback procedures and criteria

GUIDELINES:
- Always have a rollback plan ready
- Use phased deployment strategies for risk reduction
- Verify deployment health at each stage
- Communicate status to stakeholders
- Document any issues or deviations""",

            TaskCategory.MONITORING: """You are an Observability Engineer with expertise in monitoring, alerting, and system performance analysis.

MONITORING PILLARS:
- Metrics: Key performance indicators and business metrics
- Logs: Application and system event data
- Traces: Request flow and distributed system behavior
- Alerts: Actionable notifications for anomalies

APPROACH:
- Focus on user-impacting metrics first
- Build dashboards that tell a story
- Set up alerts that are actionable and not noisy
- Use SLIs/SLOs for service level management

GUIDELINES:
- Monitor the customer experience, not just system health
- Use the four golden signals: latency, traffic, errors, saturation
- Implement proper alert fatigue prevention
- Provide context in alerts and dashboards""",

            TaskCategory.SECURITY: """You are a Security Engineer with expertise in application security, infrastructure hardening, and compliance frameworks.

SECURITY DOMAINS:
- Authentication and authorization
- Data protection and encryption
- Network security and segmentation
- Vulnerability management
- Compliance and governance

APPROACH:
- Apply defense-in-depth principles
- Follow the principle of least privilege
- Use threat modeling for risk assessment
- Implement security controls that don't break usability

GUIDELINES:
- Security is everyone's responsibility
- Balance security with business requirements
- Use automated tools but don't rely solely on them
- Stay current with threat landscape and best practices
- Document security decisions and trade-offs""",

            TaskCategory.DOCUMENTATION: """You are a Technical Writer and Knowledge Management specialist with expertise in creating clear, useful documentation for technical teams.

DOCUMENTATION TYPES:
- API documentation and integration guides
- Runbooks and operational procedures
- Architecture decision records (ADRs)
- Troubleshooting guides and FAQs
- Onboarding and training materials

PRINCIPLES:
- Write for your audience's context and skill level
- Use examples and practical scenarios
- Keep documentation up-to-date and maintainable
- Make information discoverable and searchable

GUIDELINES:
- Start with the user's goal or problem
- Use clear headings and logical structure
- Include code examples and screenshots where helpful
- Test procedures and verify accuracy
- Gather feedback and iterate""",

            TaskCategory.ANALYSIS: """You are a Data Engineer and System Analyst with expertise in performance analysis, capacity planning, and operational intelligence.

ANALYSIS AREAS:
- Performance trends and capacity planning
- Cost optimization and resource utilization
- User behavior and system usage patterns
- Incident patterns and reliability metrics
- Business impact and operational efficiency

METHODOLOGY:
1. Define clear objectives and success metrics
2. Gather relevant data from multiple sources
3. Apply statistical analysis and visualization
4. Draw actionable insights and recommendations
5. Present findings in business context

GUIDELINES:
- Let data drive decisions, not assumptions
- Consider statistical significance and confidence intervals
- Account for external factors and seasonality
- Present findings clearly with visualizations
- Provide specific, actionable recommendations""",

            TaskCategory.GENERAL: """You are an experienced DevOps Engineer with broad expertise in software development, system operations, and infrastructure management.

CORE COMPETENCIES:
- System architecture and design
- Infrastructure as code and automation
- CI/CD pipeline development
- Performance optimization
- Security best practices
- Team collaboration and communication

APPROACH:
- Understand the business context and requirements
- Apply engineering best practices and industry standards
- Consider scalability, reliability, and maintainability
- Balance technical excellence with practical constraints

GUIDELINES:
- Ask clarifying questions when requirements are unclear
- Provide step-by-step guidance for complex tasks
- Share relevant best practices and lessons learned
- Consider the impact on team productivity and system reliability"""
        }
    
    def build_context(self, parsed_task: ParsedTask, max_chunks: int = 10) -> AgentContext:
        """Build complete context for an AI agent based on parsed task"""
        
        # Retrieve relevant documentation
        relevant_docs = self._retrieve_relevant_docs(parsed_task, max_chunks)
        
        # Build system prompt
        system_prompt = self._build_system_prompt(parsed_task, relevant_docs)
        
        # Build task-specific context
        task_context = self._build_task_context(parsed_task, relevant_docs)
        
        # Create search summary
        search_summary = self._create_search_summary(parsed_task, relevant_docs)
        
        # Package metadata
        metadata = {
            "task_category": parsed_task.category.value,
            "task_urgency": parsed_task.urgency.value,
            "confidence": parsed_task.confidence,
            "num_docs_retrieved": len(relevant_docs),
            "search_queries_used": parsed_task.search_queries,
            "timestamp": datetime.now().isoformat()
        }
        
        return AgentContext(
            system_prompt=system_prompt,
            task_context=task_context,
            relevant_docs=relevant_docs,
            search_summary=search_summary,
            constraints=parsed_task.constraints,
            metadata=metadata
        )
    
    def _retrieve_relevant_docs(self, parsed_task: ParsedTask, max_chunks: int) -> List[Tuple[Chunk, float]]:
        """Retrieve relevant documentation chunks using multiple search strategies"""
        all_results = []
        seen_chunk_ids = set()
        
        # Execute each search query
        for query in parsed_task.search_queries:
            try:
                # Basic search
                results = self.vector_store.search(query, k=min(max_chunks, 5))
                
                # Add to results if not already seen
                for chunk, score in results:
                    if chunk.chunk_id not in seen_chunk_ids:
                        all_results.append((chunk, score))
                        seen_chunk_ids.add(chunk.chunk_id)
                
                # Filter by document type if we have relevant entities
                if parsed_task.entities.get('services') or parsed_task.entities.get('technologies'):
                    # Try filtered search for more specific results
                    doc_type_filters = self._build_doc_type_filters(parsed_task)
                    for filter_dict in doc_type_filters:
                        filtered_results = self.vector_store.search(query, k=3, filter_metadata=filter_dict)
                        for chunk, score in filtered_results:
                            if chunk.chunk_id not in seen_chunk_ids:
                                all_results.append((chunk, score * 1.1))  # Boost filtered results
                                seen_chunk_ids.add(chunk.chunk_id)
                
            except Exception as e:
                print(f"Warning: Search failed for query '{query}': {e}")
                continue
        
        # Sort by score and return top results
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:max_chunks]
    
    def _build_doc_type_filters(self, parsed_task: ParsedTask) -> List[Dict[str, Any]]:
        """Build metadata filters based on task entities"""
        filters = []
        
        # Filter by document type based on task category
        if parsed_task.category in [TaskCategory.INCIDENT_RESPONSE, TaskCategory.TROUBLESHOOTING]:
            filters.append({"doc_type": "log"})
            filters.append({"doc_type": "yaml"})  # Config files often help with troubleshooting
        
        elif parsed_task.category == TaskCategory.CODE_REVIEW:
            filters.append({"doc_type": "markdown"})  # Code standards, guidelines
        
        elif parsed_task.category == TaskCategory.DEPLOYMENT:
            filters.append({"doc_type": "yaml"})  # Deployment configs
            filters.append({"doc_type": "markdown"})  # Deployment guides
        
        # Filter by specific technologies mentioned
        for tech in parsed_task.entities.get('technologies', []):
            filters.append({"title": tech})
        
        return filters[:3]  # Limit number of filters
    
    def _build_system_prompt(self, parsed_task: ParsedTask, relevant_docs: List[Tuple[Chunk, float]]) -> str:
        """Build comprehensive system prompt"""
        
        # Start with base prompt for task category
        base_prompt = self.base_prompts.get(parsed_task.category, self.base_prompts[TaskCategory.GENERAL])
        
        # Add task-specific context
        task_specific = f"""

CURRENT TASK: {parsed_task.intent}

URGENCY LEVEL: {parsed_task.urgency.value.upper()}"""
        
        # Add urgency-specific guidance
        if parsed_task.urgency.value in ['high', 'critical']:
            task_specific += """
- Focus on immediate resolution over perfect solutions
- Escalate quickly if initial attempts don't work
- Keep stakeholders informed of progress"""
        
        # Add entity context
        if parsed_task.entities:
            task_specific += "\n\nRELEVANT SYSTEMS/TECHNOLOGIES:"
            for entity_type, entities in parsed_task.entities.items():
                if entities:
                    task_specific += f"\n- {entity_type.title()}: {', '.join(entities)}"
        
        # Add constraints
        if parsed_task.constraints:
            task_specific += "\n\nCONSTRAINTS:"
            for constraint, value in parsed_task.constraints.items():
                task_specific += f"\n- {constraint}: {value}"
        
        # Add documentation context
        if relevant_docs:
            doc_context = "\n\nAVAILABLE DOCUMENTATION:"
            doc_context += "\nYou have access to relevant documentation that has been retrieved based on the task context."
            doc_context += "\nUse this documentation to provide accurate, specific guidance."
            doc_context += "\nAlways cite specific documentation when making recommendations."
        else:
            doc_context = "\n\nNOTE: No specific documentation was found for this task. Provide general best practices and suggest what documentation would be helpful."
        
        return base_prompt + task_specific + doc_context
    
    def _build_task_context(self, parsed_task: ParsedTask, relevant_docs: List[Tuple[Chunk, float]]) -> str:
        """Build detailed task context with documentation"""
        
        context = f"# Task Analysis\n\n"
        context += f"**Original Request:** {parsed_task.original_query}\n\n"
        context += f"**Interpreted Intent:** {parsed_task.intent}\n\n"
        context += f"**Category:** {parsed_task.category.value.replace('_', ' ').title()}\n\n"
        context += f"**Urgency:** {parsed_task.urgency.value.title()}\n\n"
        
        if parsed_task.entities:
            context += "**Identified Components:**\n"
            for entity_type, entities in parsed_task.entities.items():
                if entities:
                    context += f"- {entity_type.replace('_', ' ').title()}: {', '.join(entities)}\n"
            context += "\n"
        
        # Add relevant documentation
        if relevant_docs:
            context += "# Relevant Documentation\n\n"
            for i, (chunk, score) in enumerate(relevant_docs[:5], 1):  # Show top 5
                context += f"## Document {i}: {chunk.metadata.get('title', 'Unknown')}\n"
                context += f"**Source:** {chunk.metadata.get('doc_type', 'unknown')} | **Relevance:** {score:.2f}\n\n"
                
                # Truncate content if too long
                content = chunk.content
                if len(content) > 500:
                    content = content[:500] + "...\n\n[Content truncated - full document available]"
                
                context += f"```\n{content}\n```\n\n"
        
        return context
    
    def _create_search_summary(self, parsed_task: ParsedTask, relevant_docs: List[Tuple[Chunk, float]]) -> str:
        """Create a summary of what documentation was found"""
        
        if not relevant_docs:
            return "No relevant documentation found. Agent will provide general guidance based on best practices."
        
        doc_types = {}
        total_score = 0
        
        for chunk, score in relevant_docs:
            doc_type = chunk.metadata.get('doc_type', 'unknown')
            if doc_type not in doc_types:
                doc_types[doc_type] = {'count': 0, 'total_score': 0, 'titles': set()}
            
            doc_types[doc_type]['count'] += 1
            doc_types[doc_type]['total_score'] += score
            doc_types[doc_type]['titles'].add(chunk.metadata.get('title', 'Unknown'))
            total_score += score
        
        summary = f"Found {len(relevant_docs)} relevant documentation chunks "
        summary += f"(avg relevance: {total_score/len(relevant_docs):.2f}):\n\n"
        
        for doc_type, info in doc_types.items():
            avg_score = info['total_score'] / info['count']
            summary += f"- {info['count']} {doc_type} documents (avg relevance: {avg_score:.2f})\n"
            if len(info['titles']) <= 3:
                summary += f"  Titles: {', '.join(list(info['titles']))}\n"
            else:
                titles_list = list(info['titles'])
                summary += f"  Titles: {', '.join(titles_list[:3])} and {len(titles_list)-3} others\n"
        
        return summary


# Usage example
if __name__ == "__main__":
    from opspilot.tasks.parser import HybridTaskParser
    from opspilot.storage.vector_store import create_vector_store
    from opspilot.embedding.chunker import EmbeddingGenerator
    
    # Create components (mock for demo)
    vector_store = create_vector_store("faiss", embedding_generator=EmbeddingGenerator())
    context_constructor = ContextConstructor(vector_store)
    task_parser = HybridTaskParser(use_llm=False)
    
    # Example task
    task = task_parser.parse("The API gateway is timing out and users can't login")
    
    # Build context
    agent_context = context_constructor.build_context(task)
    
    print("System Prompt Preview:")
    print(agent_context.system_prompt[:500] + "...")
    print(f"\nSearch Summary: {agent_context.search_summary}")
    print(f"\nMetadata: {agent_context.metadata}")