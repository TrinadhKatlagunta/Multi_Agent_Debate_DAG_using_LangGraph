import os
import asyncio
import warnings
from typing import Dict, List, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from dotenv import load_dotenv
import logging
import graphviz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rich.console import Console
from rich.panel import Panel
import textwrap
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
warnings.filterwarnings("ignore")

# Set event loop policy for Windows to avoid ProactorEventLoop issues
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Set up logging
logging.basicConfig(
    filename="debate_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# Initialize Rich console
console = Console()

# State definition
class DebateState(TypedDict):
    topic: str
    round: int
    arguments: List[Dict[str, str]]
    current_speaker: str
    summary: str
    winner: str
    winner_reason: str

# Initialize OpenAI model
try:
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
except Exception as e:
    logging.error(f"Failed to initialize OpenAI model: {str(e)}")
    raise

# Prompt templates
AGENT_PROMPT = PromptTemplate(
    input_variables=["topic", "agent_role", "history"],
    template="""
You are a {agent_role} debating the topic: "{topic}".
Based on the debate history, provide a concise, logical argument (2-3 sentences) that has not been made before.
History: {history}
Your argument:
"""
)

JUDGE_PROMPT = PromptTemplate(
    input_variables=["topic", "arguments"],
    template="""
You are a neutral judge evaluating a debate on the topic: "{topic}".
Here are the arguments made:
{arguments}
Provide a summary of the debate (3-4 sentences).
Declare a single winner, strictly either Scientist or Philosopher, based on the strength, clarity, and relevance of their arguments. Ties are not allowed.
Provide a logical justification (2-3 sentences) explaining why one outperformed the other.
Format your response exactly as:
Summary: [Your summary]
Winner: [Scientist or Philosopher]
Reason: [Your reason]
"""
)

# Cosine similarity check
def is_unique_argument(new_arg: str, past_args: List[str]) -> bool:
    if not past_args:
        return True
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([new_arg] + past_args)
    similarity = cosine_similarity(vectors[0:1], vectors[1:])[0]
    return all(sim < 0.9 for sim in similarity)

# Format text for CLI
def format_output(text: str, width: int = 80) -> str:
    return "\n".join(textwrap.wrap(text, width=width, replace_whitespace=False))

# Nodes
async def user_input_node(state: DebateState) -> DebateState:
    topic = input("Enter topic for debate: ")
    logging.info(f"Debate topic: {topic}")
    console.print(f"[bold cyan]🚀 Starting debate between Scientist and Philosopher...[/bold cyan]")
    return {"topic": topic, "round": 1, "arguments": [], "current_speaker": "Scientist"}

async def agent_node(state: DebateState, agent_role: str) -> DebateState:
    try:
        await asyncio.sleep(2)
        history = "\n".join([f"{arg['speaker']}: {arg['argument']}" for arg in state["arguments"]])
        prompt = AGENT_PROMPT.format(topic=state["topic"], agent_role=agent_role, history=history)
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        argument = response.content.strip()
        
        # Validate uniqueness
        past_args = [arg["argument"] for arg in state["arguments"] if arg["speaker"] == agent_role]
        if not is_unique_argument(argument, past_args):
            argument = f"{agent_role} reiterates a similar point, but with nuance: {argument}"
        
        # Format and display
        emoji = "🧬" if agent_role == "Scientist" else "🤔"
        color = "green" if agent_role == "Scientist" else "yellow"
        formatted_arg = format_output(argument)
        console.print(Panel(
            f"{formatted_arg}",
            title=f"[bold {color}]{emoji} [Round {state['round']}] {agent_role}[/bold {color}]",
            border_style=color,
            width=90
        ))
        logging.info(f"[Round {state['round']}] {agent_role}: {argument}")
        
        # Update state
        new_arguments = state["arguments"] + [{"speaker": agent_role, "argument": argument}]
        next_speaker = "Philosopher" if agent_role == "Scientist" else "Scientist"
        return {
            "arguments": new_arguments,
            "round": state["round"] + (1 if agent_role == "Philosopher" else 0),
            "current_speaker": next_speaker
        }
    except Exception as e:
        logging.error(f"Error in {agent_role} node: {str(e)}")
        raise

async def scientist_node(state: DebateState) -> DebateState:
    return await agent_node(state, "Scientist")

async def philosopher_node(state: DebateState) -> DebateState:
    return await agent_node(state, "Philosopher")

async def memory_node(state: DebateState) -> DebateState:
    try:
        summary = "\n".join([f"{arg['speaker']}: {arg['argument']}" for arg in state["arguments"]])
        logging.info(f"Memory updated for round {state['round']-1}: {summary}")
        return {"summary": summary}
    except Exception as e:
        logging.error(f"Error in Memory node: {str(e)}")
        raise

@retry(stop=stop_after_attempt(3), wait=wait_fixed(5), retry=retry_if_exception_type(Exception))
async def invoke_with_retry(llm, messages):
    return await llm.ainvoke(messages)

async def judge_node(state: DebateState) -> DebateState:
    try:
        await asyncio.sleep(5)
        arguments = "\n".join([f"{arg['speaker']}: {arg['argument']}" for arg in state["arguments"]])
        prompt = JUDGE_PROMPT.format(topic=state["topic"], arguments=arguments)
        logging.info(f"Judge prompt: {prompt}")
        response = await invoke_with_retry(llm, [HumanMessage(content=prompt)])
        logging.info(f"Judge response: {response.content}")
        judge_output = response.content.strip()
        
        # Robust parsing
        summary = "Summary not provided"
        winner = "Winner not declared"
        reason = "Reason not provided"
        lines = judge_output.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("Summary:"):
                summary = line.replace("Summary:", "").strip()
            elif line.startswith("Winner:"):
                winner = line.replace("Winner:", "").strip()
            elif line.startswith("Reason:"):
                reason = line.replace("Reason:", "").strip()
        
        # Fallback if parsing fails
        if not summary or not winner or not reason:
            logging.warning("Incomplete judge response, using fallback")
            summary = summary or "The debate covered various aspects of the topic."
            winner = winner if winner in ["Scientist", "Philosopher"] else "Scientist"
            reason = reason or "Winner chosen based on argument clarity."
        
        # Format and display
        formatted_summary = format_output(summary)
        formatted_reason = format_output(reason)
        console.print(Panel(
            f"[bold]Summary:[/bold]\n{formatted_summary}\n\n[bold]Winner:[/bold] {winner}\n\n[bold]Reason:[/bold]\n{formatted_reason}",
            title="[bold magenta]⚖️ Judge's Verdict[/bold magenta]",
            border_style="magenta",
            width=90
        ))
        logging.info(f"[Judge] Summary: {summary}")
        logging.info(f"[Judge] Winner: {winner}")
        logging.info(f"[Judge] Reason: {reason}")
        
        return {"summary": summary, "winner": winner, "winner_reason": reason}
    except Exception as e:
        logging.error(f"Error in Judge node: {str(e)}")
        raise

# Conditional routing
def scientist_router(state: DebateState) -> str:
    if state["round"] > 8:
        return "judge"
    return "philosopher"

def philosopher_router(state: DebateState) -> str:
    if state["round"] > 8:
        return "judge"
    return "memory" if state["round"] % 2 == 1 else "scientist"

# Build the graph
workflow = StateGraph(DebateState)
workflow.add_node("user_input", user_input_node)
workflow.add_node("scientist", scientist_node)
workflow.add_node("philosopher", philosopher_node)
workflow.add_node("memory", memory_node)
workflow.add_node("judge", judge_node)

# Define edges
workflow.set_entry_point("user_input")
workflow.add_edge("user_input", "scientist")
workflow.add_conditional_edges(
    "scientist",
    scientist_router,
    {"philosopher": "philosopher", "judge": "judge"}
)
workflow.add_conditional_edges(
    "philosopher",
    philosopher_router,
    {"scientist": "scientist", "memory": "memory", "judge": "judge"}
)
workflow.add_edge("memory", "scientist")
workflow.add_edge("judge", END)

# Compile the graph
app = workflow.compile()

# Generate DAG diagram
def generate_dag_diagram():
    try:
        dot = graphviz.Digraph(comment="Debate DAG")
        dot.node("user_input", "User Input")
        dot.node("scientist", "Scientist")
        dot.node("philosopher", "Philosopher")
        dot.node("memory", "Memory")
        dot.node("judge", "Judge")
        dot.edge("user_input", "scientist")
        dot.edge("scientist", "philosopher", label="Round <= 8")
        dot.edge("philosopher", "memory", label="End of Round")
        dot.edge("philosopher", "scientist", label="Round <= 8")
        dot.edge("memory", "scientist")
        dot.edge("scientist", "judge", label="Round > 8")
        dot.edge("philosopher", "judge", label="Round > 8")
        dot.render("debate_dag", format="png", cleanup=True)
        logging.info("Debate DAG diagram generated: debate_dag.png")
    except Exception as e:
        logging.error(f"Failed to generate DAG diagram: {str(e)}")
        raise

# Run the debate
async def run_debate():
    try:
        generate_dag_diagram()
        initial_state = DebateState(
            topic="",
            round=1,
            arguments=[],
            current_speaker="Scientist",
            summary="",
            winner="",
            winner_reason=""
        )
        await app.ainvoke(initial_state)
    except Exception as e:
        logging.error(f"Error running debate: {str(e)}")
        raise
    finally:
        loop = asyncio.get_event_loop()
        if not loop.is_closed():
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

# Execute
if __name__ == "__main__":
    asyncio.run(run_debate())