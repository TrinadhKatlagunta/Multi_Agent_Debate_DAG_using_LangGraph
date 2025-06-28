# Multi-Agent Debate System

## Overview
This project is a multi-agent debate system built using LangGraph and OpenAI's `gpt-4o-mini` model. Two agents, Scientist and Philosopher, debate a user-provided topic over 8 rounds (4 arguments each), ensuring unique arguments through cosine similarity checks. A memory node logs the debate transcript, and a judge node evaluates the arguments, declaring a single winner with a summary (3-4 sentences) and justification (2-3 sentences). The CLI is enhanced with colorful, emoji-based formatting (🧬 Scientist, 🤔 Philosopher, ⚖️ Judge) for a visually engaging experience.

## Files
- `debate_simulation.py`: Main script implementing the debate workflow.
- `debate_log.txt`: Logs debate arguments, memory updates, and judge verdict.
- `debate_dag.png`: Visual representation of the workflow (DAG).
- `requirements.txt`: Lists Python dependencies.
- `.env`: Stores `OPENAI_API_KEY` (not submitted).

## Setup
1. **Install Graphviz**:
   - Download from `graphviz.org`.
   - Add `C:\Program Files\Graphviz\bin` to Windows PATH.
   - Verify: `dot -version`.

2. **Set Up Virtual Environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Dependencies include `langgraph`, `langchain`, `langchain_openai`, `python-dotenv`, `graphviz`, `scikit-learn`, `rich`, and `tenacity`.

4. **Configure API Key**:
   - Create `.env` in the project root (`C:\Users\katla\Desktop\Pro`).
   - Add: `OPENAI_API_KEY=your-openai-api-key-here`.

5. **Run the Script**:
   ```bash
   python debate_simulation.py
   ```
   - Enter a topic (e.g., “Should sex education be imparted in school?”).
   - Output: 8 rounds of debate, judge verdict, logged to `debate_log.txt`, and visualized in `debate_dag.png`.

## Output
- **Console**: Displays 8 rounds (4 per agent) with emoji-enhanced formatting (🧬 Scientist, 🤔 Philosopher), followed by the judge’s verdict (⚖️) with a 3-4 sentence summary, single winner, and 2-3 sentence justification.
- **debate_log.txt**: Records all arguments, memory updates, and judge output.
- **debate_dag.png**: Workflow diagram showing node connections (user input → scientist → philosopher → memory → judge).

## Notes
- The CLI uses `rich` for colorful, boxed output with emojis (🧬 Scientist, 🤔 Philosopher, ⚖️ Judge) to enhance readability.
- `tenacity` ensures robust API calls for the judge node, retrying up to 3 times on failure.
- The script uses OpenAI’s `gpt-4o-mini` for cost-efficient, reliable performance, with arguments checked for uniqueness (cosine similarity < 0.9).
