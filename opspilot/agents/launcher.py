"""
OpsPilot Agent Launcher
Creates and manages task-aware AI agents with full context.
"""

from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass
import json
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.callbacks.base import BaseCallbackHandler

from opspilot.tasks.parser import ParsedTask, HybridTaskParser
from opspilot.agents.context import ContextConstructor, AgentContext
from opspilot.storage.vector_store import VectorStore


@dataclass
class AgentResponse:
    """Response from an OpsPilot agent"""
    content: str
    confidence: float
    sources_used: List[str]
    reasoning: Optional[str] = None
    follow_up_suggestions: List[str] = None
    metadata: Dict[str, Any] = None


class OpsPilotCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for OpsPilot agents"""
    
    def __init__(self):
        self.tokens_used = 0
        self.reasoning_steps = []
        self.sources_referenced = []
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Called when LLM starts generating"""
        print("🤖 Agent thinking...")
    
    def on_llm_end(self, response, **kwargs) -> None:
        """Called when LLM finishes generating"""
        if hasattr(response, 'llm_output') and response.llm_output:
            if 'token_usage' in response.llm_output:
                self.tokens_used += response.llm_output['token_usage'].get('total_tokens', 0)
        print("✅ Agent response ready")


class OpsPilotAgent:
    """A task-aware AI agent with specialized context"""
    
    def __init__(self, 
                 agent_context: AgentContext,
                 llm_provider: str = "openai",
                 llm_model: str = "gpt-4",
                 temperature: float = 0.1,
                 api_key: Optional[str] = None):
        
        self.context = agent_context
        self.conversation_history: List[Dict[str, str]] = []
        
        # Initialize LLM
        if llm_provider == "openai":
            self.llm = ChatOpenAI(
                model=llm_model,
                temperature=temperature,
                openai_api_key=api_key,
                callbacks=[OpsPilotCallbackHandler()]
            )
        elif llm_provider == "anthropic":
            self.llm = ChatAnthropic(
                model=llm_model,
                temperature=temperature,
                anthropic_api_key=api_key,
                callbacks=[OpsPilotCallbackHandler()]
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
        
        # Agent metadata
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "task_category": agent_context.metadata.get("task_category"),
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "context_docs": len(agent_context.relevant_docs)
        }
    
    def respond(self, user_message: str, include_reasoning: bool = False) -> AgentResponse:
        """Generate a response to user input"""
        
        # Build messages for LLM
        messages = self._build_messages(user_message, include_reasoning)
        
        try:
            # Get LLM response
            response = self.llm(messages)
            
            # Parse response if it includes structured reasoning
            if include_reasoning and self._is_structured_response(response.content):
                parsed_response = self._parse_structured_response(response.content)
                content = parsed_response.get("response", response.content)
                reasoning = parsed_response.get("reasoning")
                confidence = parsed_response.get("confidence", 0.8)
                follow_ups = parsed_response.get("follow_up_suggestions", [])
            else:
                content = response.content
                reasoning = None
                confidence = 0.8
                follow_ups = self._generate_follow_ups(user_message, content)
            
            # Extract sources used
            sources_used = self._extract_sources_from_response(content)
            
            # Store in conversation history
            self.conversation_history.append({
                "role": "user",
                "content": user_message,
                "timestamp": datetime.now().isoformat()
            })
            self.conversation_history.append({
                "role": "assistant", 
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "sources": sources_used
            })
            
            return AgentResponse(
                content=content,
                confidence=confidence,
                sources_used=sources_used,
                reasoning=reasoning,
                follow_up_suggestions=follow_ups,
                metadata={
                    "response_length": len(content),
                    "conversation_turn": len(self.conversation_history) // 2
                }
            )
            
        except Exception as e:
            # Return error response
            error_content = f"I encountered an error while processing your request: {str(e)}\n\n"
            error_content += "Please try rephrasing your question or contact support if the issue persists."
            
            return AgentResponse(
                content=error_content,
                confidence=0.0,
                sources_used=[],
                metadata={"error": str(e)}
            )
    
    def _build_messages(self, user_message: str, include_reasoning: bool = False) -> List:
        """Build message history for LLM"""
        messages = []
        
        # System prompt
        system_content = self.context.system_prompt
        
        # Add reasoning instructions if requested
        if include_reasoning:
            system_content += """

RESPONSE FORMAT:
Structure your response as JSON with the following format:
{
  "reasoning": "Step-by-step thinking process",
  "response": "Your main response to the user",
  "confidence": 0.95,
  "follow_up_suggestions": ["suggestion 1", "suggestion 2"]
}

If the task is urgent or the user asks for a quick response, skip the JSON format and respond directly."""
        
        messages.append(SystemMessage(content=system_content))
        
        # Add task context as first user message
        if not self.conversation_history:  # Only on first interaction
            messages.append(HumanMessage(content=self.context.task_context))
        
        # Add conversation history
        for turn in self.conversation_history:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            else:
                messages.append(AIMessage(content=turn["content"]))
        
        # Add current user message
        messages.append(HumanMessage(content=user_message))
        
        return messages
    
    def _is_structured_response(self, content: str) -> bool:
        """Check if response is in structured JSON format"""
        content = content.strip()
        return content.startswith('{') and content.endswith('}')
    
    def _parse_structured_response(self, content: str) -> Dict[str, Any]:
        """Parse structured JSON response"""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"response": content}
    
    def _extract_sources_from_response(self, content: str) -> List[str]:
        """Extract document sources referenced in the response"""
        sources = []
        
        # Look for references to documentation chunks
        for chunk, score in self.context.relevant_docs:
            doc_title = chunk.metadata.get('title', 'Unknown')
            
            # Check if the document title or content is referenced
            if (doc_title.lower() in content.lower() or 
                any(word in content.lower() for word in doc_title.lower().split()[:3])):
                sources.append(doc_title)
        
        return list(set(sources))  # Remove duplicates
    
    def _generate_follow_ups(self, user_message: str, response: str) -> List[str]:
        """Generate helpful follow-up suggestions"""
        follow_ups = []
        
        task_category = self.context.metadata.get("task_category")
        
        if task_category == "incident_response":
            follow_ups = [
                "What should be the next escalation step?",
                "How do I communicate this to stakeholders?",
                "What monitoring should I check next?"
            ]
        elif task_category == "troubleshooting":
            follow_ups = [
                "What logs should I examine next?",
                "Are there any related known issues?",
                "How can I prevent this in the future?"
            ]
        elif task_category == "code_review":
            follow_ups = [
                "Are there any security concerns I missed?",
                "What testing should be added?",
                "How does this affect performance?"
            ]
        elif task_category == "deployment":
            follow_ups = [
                "What's the rollback procedure?",
                "How do I verify the deployment succeeded?",
                "What should I monitor after deployment?"
            ]
        else:
            follow_ups = [
                "Can you provide more specific guidance?",
                "What documentation would help with this?",
                "What are the potential risks to consider?"
            ]
        
        return follow_ups[:3]  # Limit to 3 suggestions
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of the conversation"""
        return {
            "total_turns": len(self.conversation_history) // 2,
            "task_category": self.context.metadata.get("task_category"),
            "docs_available": len(self.context.relevant_docs),
            "created_at": self.metadata["created_at"],
            "last_interaction": self.conversation_history[-1]["timestamp"] if self.conversation_history else None
        }


class AgentLauncher:
    """Factory for creating task-aware OpsPilot agents"""
    
    def __init__(self,
                 vector_store: VectorStore,
                 task_parser: Optional[HybridTaskParser] = None,
                 context_constructor: Optional[ContextConstructor] = None,
                 default_llm_provider: str = "openai",
                 default_llm_model: str = "gpt-4",
                 api_key: Optional[str] = None):
        
        self.vector_store = vector_store
        self.task_parser = task_parser or HybridTaskParser(use_llm=api_key is not None)
        self.context_constructor = context_constructor or ContextConstructor(vector_store)
        self.default_llm_provider = default_llm_provider
        self.default_llm_model = default_llm_model
        self.api_key = api_key
        
        # Track created agents
        self.active_agents: Dict[str, OpsPilotAgent] = {}
    
    def launch_agent(self, 
                    task_description: str,
                    additional_context: Optional[str] = None,
                    agent_id: Optional[str] = None,
                    llm_provider: Optional[str] = None,
                    llm_model: Optional[str] = None,
                    max_context_docs: int = 10) -> OpsPilotAgent:
        """Launch a new task-aware agent"""
        
        print(f"🚀 Launching OpsPilot agent for: {task_description}")
        
        # Parse the task
        print("📋 Parsing task...")
        parsed_task = self.task_parser.parse(task_description, additional_context)
        print(f"   Category: {parsed_task.category.value}")
        print(f"   Urgency: {parsed_task.urgency.value}")
        print(f"   Confidence: {parsed_task.confidence:.2f}")
        
        # Build context
        print("🔍 Building context...")
        agent_context = self.context_constructor.build_context(parsed_task, max_context_docs)
        print(f"   Retrieved {len(agent_context.relevant_docs)} relevant documents")
        
        # Create agent
        print("🤖 Creating agent...")
        agent = OpsPilotAgent(
            agent_context=agent_context,
            llm_provider=llm_provider or self.default_llm_provider,
            llm_model=llm_model or self.default_llm_model,
            api_key=self.api_key
        )
        
        # Store agent if ID provided
        if agent_id:
            self.active_agents[agent_id] = agent
        
        print("✅ Agent ready!")
        return agent
    
    def get_agent(self, agent_id: str) -> Optional[OpsPilotAgent]:
        """Get an existing agent by ID"""
        return self.active_agents.get(agent_id)
    
    def list_active_agents(self) -> Dict[str, Dict[str, Any]]:
        """List all active agents with their summaries"""
        return {
            agent_id: agent.get_conversation_summary() 
            for agent_id, agent in self.active_agents.items()
        }
    
    def terminate_agent(self, agent_id: str) -> bool:
        """Terminate an active agent"""
        if agent_id in self.active_agents:
            del self.active_agents[agent_id]
            return True
        return False


class InteractiveSession:
    """Interactive session manager for OpsPilot agents"""
    
    def __init__(self, agent: OpsPilotAgent):
        self.agent = agent
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def start_session(self):
        """Start interactive session with the agent"""
        print(f"\n🎯 OpsPilot Agent Session Started")
        print(f"📋 Task Category: {self.agent.context.metadata['task_category']}")
        print(f"📚 Documentation: {len(self.agent.context.relevant_docs)} relevant docs loaded")
        print(f"🔧 Agent: {self.agent.metadata['llm_model']} ({self.agent.metadata['llm_provider']})")
        print("\n" + "="*60)
        print("💬 Type your questions below. Type 'exit' to end session.")
        print("💡 Type 'help' for available commands.")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() == 'exit':
                    print("\n👋 Session ended. Thank you for using OpsPilot!")
                    break
                
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                elif user_input.lower() == 'context':
                    self._show_context()
                    continue
                
                elif user_input.lower() == 'docs':
                    self._show_docs()
                    continue
                
                elif user_input.lower() == 'history':
                    self._show_history()
                    continue
                
                elif user_input.lower().startswith('reasoning'):
                    include_reasoning = True
                    if len(user_input.split()) > 1:
                        user_input = " ".join(user_input.split()[1:])
                    else:
                        user_input = input("Question (with reasoning): ").strip()
                else:
                    include_reasoning = False
                
                if not user_input:
                    continue
                
                # Get agent response
                print("\n🤖 Agent:")
                response = self.agent.respond(user_input, include_reasoning=include_reasoning)
                
                print(response.content)
                
                # Show reasoning if available
                if response.reasoning:
                    print(f"\n🧠 Reasoning: {response.reasoning}")
                
                # Show sources if available
                if response.sources_used:
                    print(f"\n📚 Sources: {', '.join(response.sources_used)}")
                
                # Show follow-up suggestions
                if response.follow_up_suggestions:
                    print("\n💡 Suggested follow-ups:")
                    for i, suggestion in enumerate(response.follow_up_suggestions, 1):
                        print(f"   {i}. {suggestion}")
                
                print("\n" + "-"*60 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Session ended. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again or type 'exit' to end session.\n")
    
    def _show_help(self):
        """Show available commands"""
        print("\n📖 Available Commands:")
        print("  help     - Show this help message")
        print("  context  - Show task context and agent info")
        print("  docs     - Show loaded documentation")
        print("  history  - Show conversation history")
        print("  reasoning <question> - Ask with step-by-step reasoning")
        print("  exit     - End session")
        print()
    
    def _show_context(self):
        """Show agent context"""
        print("\n📋 Agent Context:")
        print(f"  Task Category: {self.agent.context.metadata['task_category']}")
        print(f"  Urgency: {self.agent.context.metadata.get('task_urgency', 'unknown')}")
        print(f"  Confidence: {self.agent.context.metadata.get('confidence', 'unknown')}")
        print(f"  Documents: {len(self.agent.context.relevant_docs)} loaded")
        print(f"  Created: {self.agent.metadata['created_at']}")
        print()
    
    def _show_docs(self):
        """Show loaded documentation"""
        print("\n📚 Loaded Documentation:")
        if not self.agent.context.relevant_docs:
            print("  No documents loaded")
        else:
            for i, (chunk, score) in enumerate(self.agent.context.relevant_docs[:5], 1):
                title = chunk.metadata.get('title', 'Unknown')
                doc_type = chunk.metadata.get('doc_type', 'unknown')
                print(f"  {i}. {title} ({doc_type}) - Relevance: {score:.2f}")
        print()
    
    def _show_history(self):
        """Show conversation history"""
        print("\n💬 Conversation History:")
        if not self.agent.conversation_history:
            print("  No conversation yet")
        else:
            for turn in self.agent.conversation_history[-6:]:  # Show last 6 messages
                role = "You" if turn["role"] == "user" else "Agent"
                content = turn["content"][:100] + "..." if len(turn["content"]) > 100 else turn["content"]
                print(f"  {role}: {content}")
        print()


# Complete usage example
def main():
    """Example usage of the complete OpsPilot system"""
    import os
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("⚠️  No API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        return
    
    try:
        # Initialize components
        from opspilot.storage.vector_store import create_vector_store
        from opspilot.embedding.chunker import EmbeddingGenerator
        
        print("🔧 Initializing OpsPilot...")
        
        # Create vector store
        embedding_gen = EmbeddingGenerator(api_key=api_key)
        vector_store = create_vector_store("faiss", embedding_generator=embedding_gen)
        
        # Create launcher
        launcher = AgentLauncher(
            vector_store=vector_store,
            api_key=api_key
        )
        
        # Example tasks
        example_tasks = [
            "The API gateway is returning 504 errors and users can't login",
            "Review the security changes in the authentication service PR",
            "Help me deploy the new microservice to staging environment",
            "Create documentation for our incident response procedure"
        ]
        
        print("\n🎯 Example Tasks:")
        for i, task in enumerate(example_tasks, 1):
            print(f"  {i}. {task}")
        
        # Get user task
        print("\n💬 Enter your task description:")
        task_description = input("Task: ").strip()
        
        if not task_description:
            task_description = example_tasks[0]  # Use first example
            print(f"Using example: {task_description}")
        
        # Launch agent
        agent = launcher.launch_agent(task_description)
        
        # Start interactive session
        session = InteractiveSession(agent)
        session.start_session()
        
    except Exception as e:
        print(f"❌ Error initializing OpsPilot: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()