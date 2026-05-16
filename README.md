Multi-Agent Educational AI System — README 

Overview 
This project is a Collaborative Multi-Agent Educational AI System designed to simulate 
how different educational experts work together to help students solve problems, understand 
concepts, receive feedback, and improve learning outcomes. 
The system uses multiple specialized AI agents coordinated by an Orchestrator Agent to 
create a structured educational workflow. 

The architecture supports multiple LLM providers including: 
• Groq (Free) 
• Google Gemini (Free) 
• OpenAI 
• Anthropic 
The system is modular, extensible, and designed for educational problem solving, tutoring, 
evaluation, and learning-plan generation. 

Features 
Multi-Agent Collaboration 
The system includes six specialized agents: 

1. TutorAgent 
• Explains topics in student-friendly language 
• Identifies likely knowledge gaps 
• Uses analogies and simplified explanations 
• Encourages deeper understanding with clarifying questions

2. ProblemSolverAgent 
• Solves problems step-by-step 
• Explicitly states assumptions 
• Shows complete reasoning 
• Provides detailed solutions with explanations
 
3. EvaluatorAgent 
• Critically reviews student answers 
• Detects errors and logical gaps 
• Verifies correctness independently 
• Rates responses as: 
o CORRECT 
o PARTIALLY CORRECT 
o INCORRECT

4. FeedbackAgent 
• Provides supportive and actionable feedback 
• Summarizes strengths and weaknesses 
• Encourages student improvement 
• Suggests practical next steps

5. PlannerAgent 
• Generates a structured 3-step learning plan 
• Aligns recommendations with curriculum context 
• Provides measurable learning objectives 
• Suggests resources and activities

6. OrchestratorAgent 
• Coordinates the entire pipeline 
• Monitors quality of outputs 
• Detects emergent behaviors 
• Handles retries and quality checks 
• Generates emergence reports

System Architecture 
Student Input 
│ 
▼ 
Tutor Agent 
│ 
▼ 
Problem Solver Agent 
│ 
▼ 
Evaluator Agent 
│ 
▼ 
Feedback Agent 
│ 
▼ 
Planner Agent 
│ 
▼ 
Orchestrator Monitoring & Emergence Detection 

Technologies Used 
Programming Language 
• Python 3.10+ 
AI Providers Supported 
• Groq 
• Google Gemini 
• OpenAI 
• Anthropic 
Core Python Libraries 
• dataclasses 
• datetime 
• typing 
• textwrap 
• re 
• os 

Installation 
1. Clone the Repository 
git clone <repository-url> 
cd <repository-folder> 
2. Install Dependencies 
For Groq 
pip install groq 
For Gemini 
pip install google-generativeai 
For OpenAI 
pip install openai 
For Anthropic 
pip install anthropic

API Key Setup 
Set the required API key as an environment variable. 
Groq 
export GROQ_API_KEY="your_api_key" 
Gemini 
export GEMINI_API_KEY="your_api_key" 
OpenAI 
export OPENAI_API_KEY="your_api_key" 
Anthropic 
export ANTHROPIC_API_KEY="your_api_key"
 
Running the Project 
Execute the Python file: python Code.py

User Workflow 
When the program starts: 
1. Choose an LLM provider 
2. Enter: 
o Equation or problem 
o Question about the problem 
o Student's proposed answer 
o Curriculum context 
3. The orchestrator runs the full multi-agent pipeline 
4. Results are displayed sequentially
   
Example Input 
Equation/problem: 
x^2 + 5x + 6 = 0 
Question: 
How do I solve it? 
Student Answer: 
x = -2, x = -3 
Curriculum: 
Grade 10 Algebra, Unit 3 

Example Pipeline Output 
The system generates: 
• Topic explanation 
• Step-by-step solution 
• Evaluation of student answer 
• Constructive feedback 
• Personalized learning plan 
• Emergence analysis report 

Shared Session Memory 
The system uses a shared blackboard architecture through the SessionMemory class. 
It stores: 
• Student question 
• Topic 
• Student answer 
• Curriculum hints 
• Outputs from all agents 
• Emergence detection logs 
• Pipeline history 
This allows all agents to collaborate using shared context. 

Emergence Detection 
The OrchestratorAgent includes advanced monitoring capabilities. 

Positive Emergence Detection 
Detects: 
• Strong reasoning 
• Clear explanations 
• Creative solutions 
• Well-aligned responses 

Negative Emergence Detection 
Detects: 
• Topic drift 
• Unsupported reasoning 
• Blind agreement 
• Low-quality outputs 
• Error leakage 

Quality Control Features 
Automatic Quality Gates 
Each agent output is checked for: 
• Minimum word count 
• Suspicious patterns 
• Missing reasoning 
• Invalid responses 

Retry Mechanism 
If an agent fails quality checks: 
• The orchestrator retries execution 
• Maximum retries configurable 

Configuration 
The LLMConfig class controls: 
provider 
model 
temperature 
max_tokens 
api_key 
Per-agent temperature tuning is also supported. 
Example: 
agent_temps = { 
"tutor": 0.75, 
"problem_solver": 0.30, 
"evaluator": 0.40, 
"feedback": 0.80, 
"planner": 0.50, 
} 

Supported Models 
Groq 
• llama-3.3-70b-versatile 
Gemini 
• gemini-1.5 
OpenAI 
• gpt-4o 
Anthropic 
• claude-3-5-sonnet-latest 

Project Structure 
Code.py 
│ 
├── LLMConfig 
├── LLMBackbone 
├── SessionMemory 
├── BaseAgent 
│ 
├── TutorAgent 
├── ProblemSolverAgent 
├── EvaluatorAgent 
├── FeedbackAgent 
├── PlannerAgent 
│ 
└── OrchestratorAgent 

Educational Benefits 
This system helps students by: 
• Encouraging critical thinking 
• Providing transparent reasoning 
• Supporting self-correction 
• Giving curriculum-aligned guidance 
• Promoting growth mindset learning 

Future Improvements 
Possible enhancements: 
• Web UI integration 
• Persistent memory storage 
• Multi-turn conversations 
• PDF report generation 
• Voice interaction 
• LMS integration 
• Performance analytics dashboard

Conclusion 
This project demonstrates how multiple specialized AI agents can collaborate to create a 
robust educational support system. 
