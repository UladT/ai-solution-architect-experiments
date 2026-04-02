# ai-solution-architect-experiments
EPAM AI Solution Architect Program - Practical Tasks
# AI Solution Architect - Practical Tasks

## Task 1: Prompt Engineering

### Acceptance Criteria Coverage
| Criteria | Implementation |
|---|---|
| AC-1: Functional ASRs | ReAct prompt + meta-prompting |
| AC-2: Scalable | Template variables, multiple scenarios |
| AC-3: Evaluation Metric | PromptEvaluator with 6 metrics |
| AC-4: Improving Quality | Self-reflection loop + A/B testing |
| AC-5: Security | SecurityGuard input/output validation |

### Setup
```bash
pip install -r requirements.txt
cp .env.example .env  # add your API key
cd task1_prompt_engineering
python main.py