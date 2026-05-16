"""
Multi-Agent Educational AI System — Collaborative Problem Solving
=================================================================
Agents
------
1. TutorAgent        – breaks topic into student-friendly language, finds gaps, uses analogies
2. ProblemSolverAgent– solves problems step-by-step, states every assumption explicitly
3. EvaluatorAgent    – critical reviewer: finds errors, gaps, unsupported claims; never rubber-stamps
4. FeedbackAgent     – synthesises the full discussion into encouraging, actionable feedback
5. PlannerAgent      – creates a structured 3-step next-learning plan
6. OrchestratorAgent – coordinates the pipeline, monitors agent outputs, detects emergence

Shared backbone: Groq (FREE), Google Gemini (FREE), OpenAI (paid), or Anthropic (paid) — swap via config.

Free API keys:
  Groq   → https://console.groq.com  (no credit card)
  Gemini → https://aistudio.google.com (no credit card)
"""

from __future__ import annotations

import os
import re
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    """Central configuration for the shared LLM backbone."""
    # provider: "groq" (free) | "gemini" (free) | "openai" (paid) | "anthropic" (paid)
    provider: str = "groq"
    model: str = "llama-3.3-70b-versatile"   # default: Groq free model
    temperature: float = 0.7
    max_tokens: int = 1024
    api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))

    # Per-agent temperature overrides (tighter = more consistent role-play)
    agent_temps: dict[str, float] = field(default_factory=lambda: {
        "tutor":          0.75,
        "problem_solver": 0.30,   # low → deterministic step-by-step reasoning
        "evaluator":      0.40,   # low → reliable critical judgment
        "feedback":       0.80,
        "planner":        0.50,
    })


# ─────────────────────────────────────────────────────────────────────────────
# SHARED LLM BACKBONE
# ─────────────────────────────────────────────────────────────────────────────

class LLMBackbone:
    """
    Thin wrapper around Groq, Gemini, OpenAI, or Anthropic clients.
    All agents share a single instance, ensuring a single API-key / model policy.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client: Any = None
        self._init_client()

    def _init_client(self) -> None:
        if self.config.provider == "groq":
            try:
                from groq import Groq  # type: ignore
                self._client = Groq(api_key=self.config.api_key)
            except ImportError:
                raise ImportError("Install groq: pip install groq")
        elif self.config.provider == "gemini":
            try:
                import google.generativeai as genai  # type: ignore
                genai.configure(api_key=self.config.api_key)
                self._client = genai  # store the module; model created per-call
            except ImportError:
                raise ImportError("Install google-generativeai: pip install google-generativeai")
        elif self.config.provider == "openai":
            try:
                import openai  # type: ignore
                self._client = openai.OpenAI(api_key=self.config.api_key)
            except ImportError:
                raise ImportError("Install openai: pip install openai")
        elif self.config.provider == "anthropic":
            try:
                import anthropic  # type: ignore
                self._client = anthropic.Anthropic(
                    api_key=os.getenv("ANTHROPIC_API_KEY", "")
                )
            except ImportError:
                raise ImportError("Install anthropic: pip install anthropic")
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")

    def call(self, system_prompt: str, user_message: str, agent_name: str = "") -> str:
        """Send a single-turn message with an optional system prompt."""
        temp = self.config.agent_temps.get(agent_name, self.config.temperature)

        if self.config.provider == "groq":
            # Groq uses the same interface as OpenAI
            response = self._client.chat.completions.create(
                model=self.config.model,
                temperature=temp,
                max_tokens=self.config.max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
            )
            return response.choices[0].message.content.strip()

        elif self.config.provider == "gemini":
            # Gemini: system prompt is passed as system_instruction
            try:
                model = self._client.GenerativeModel(
                    model_name=self.config.model,
                    system_instruction=system_prompt,
                )
                response = model.generate_content(
                    user_message,
                    generation_config={
                        "temperature": temp,
                        "max_output_tokens": self.config.max_tokens,
                    },
                )
                return response.text.strip()
            except Exception as exc:
                msg = str(exc).lower()
                if "not found" in msg or "unsupported for generatecontent" in msg:
                    raise RuntimeError(
                        f"Gemini model '{self.config.model}' is not supported by the current API. "
                        "Please choose a supported Gemini model name, e.g. 'gemini-1.5'."
                    ) from exc
                raise

        elif self.config.provider == "openai":
            response = self._client.chat.completions.create(
                model=self.config.model,
                temperature=temp,
                max_tokens=self.config.max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
            )
            return response.choices[0].message.content.strip()

        elif self.config.provider == "anthropic":
            response = self._client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=temp,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            return response.content[0].text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# SESSION MEMORY  (shared blackboard between agents)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SessionMemory:
    """Shared blackboard that every agent reads from and writes to."""
    topic:            str = ""
    student_question: str = ""
    student_answer:   str = ""
    curriculum_hints: str = ""

    # Agent outputs (filled in during the pipeline run)
    tutor_output:    str = ""
    solver_output:   str = ""
    evaluator_output:str = ""
    feedback_output: str = ""
    plan_output:     str = ""

    # Orchestrator tracking
    pipeline_log:       list[dict] = field(default_factory=list)
    iteration:          int = 0
    emergence_positive: list[str] = field(default_factory=list)
    emergence_negative: list[str] = field(default_factory=list)
    emergence_flags:    list[str] = field(default_factory=list)

    def log(self, agent: str, output: str) -> None:
        self.pipeline_log.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "agent":     agent,
            "output":    output,
        })


# ─────────────────────────────────────────────────────────────────────────────
# BASE AGENT
# ─────────────────────────────────────────────────────────────────────────────

class BaseAgent:
    """Abstract base for all agents in the system."""

    name: str = "base"
    system_prompt_template: str = ""

    def __init__(self, llm: LLMBackbone):
        self.llm = llm

    def run(self, memory: SessionMemory) -> str:
        raise NotImplementedError

    def _call(self, user_message: str) -> str:
        return self.llm.call(
            system_prompt=self.system_prompt_template,
            user_message=user_message,
            agent_name=self.name,
        )

    @staticmethod
    def _wrap(text: str, width: int = 100) -> str:
        """Soft-wrap long lines for readable console output."""
        return "\n".join(
            textwrap.fill(line, width) if len(line) > width else line
            for line in text.splitlines()
        )


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 1 — TUTOR
# ─────────────────────────────────────────────────────────────────────────────

class TutorAgent(BaseAgent):
    """
    Breaks down the topic into clear, student-friendly language.
    Identifies likely knowledge gaps and uses concrete analogies.
    """

    name = "tutor"
    system_prompt_template = textwrap.dedent("""\
        You are an expert, empathetic tutor. Your job is to make complex topics accessible.

        RULES you MUST follow:
        1. Reframe the topic in plain, jargon-free language suitable for a curious student.
        2. Identify at least TWO likely knowledge gaps a student might have on this topic.
        3. Provide at least ONE concrete, memorable analogy that illuminates the core concept.
        4. Keep your explanation focused and engaging — no unnecessary padding.
        5. End with a single clarifying question that invites the student to deepen their understanding.

        Structure your response with clear headings:
        ## Topic Explained
        ## Knowledge Gaps to Watch
        ## Analogy
        ## Clarifying Question
    """)

    def run(self, memory: SessionMemory) -> str:
        user_msg = (
            f"Topic: {memory.topic}\n"
            f"Student's Question: {memory.student_question}\n"
            f"Student's Proposed Answer: {memory.student_answer or 'None provided'}\n"
            f"Curriculum Context: {memory.curriculum_hints or 'Not provided'}"
        )
        output = self._call(user_msg)
        memory.tutor_output = output
        memory.log(self.name, output)
        return output


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 2 — PROBLEM SOLVER
# ─────────────────────────────────────────────────────────────────────────────

class ProblemSolverAgent(BaseAgent):
    """
    Works through the problem methodically.
    States every assumption, shows every step, arrives at a clearly labelled answer.
    """

    name = "problem_solver"
    system_prompt_template = textwrap.dedent("""\
        You are a meticulous, systematic problem solver. Accuracy and transparency matter most.

        RULES you MUST follow:
        1. List ALL assumptions you are making before you start solving.
        2. Number every step. Never skip steps — even trivial arithmetic must be shown.
        3. At each step, state WHAT you are doing and WHY.
        4. If multiple valid approaches exist, note them briefly; then choose the clearest one.
        5. Label your final answer clearly as "FINAL ANSWER:" on its own line.
        6. After the final answer, note any limitations or edge cases.

        Structure:
        ## Assumptions
        ## Step-by-Step Solution
        ## Final Answer
        ## Limitations / Edge Cases
    """)

    def run(self, memory: SessionMemory) -> str:
        user_msg = (
            f"Topic Context (from Tutor):\n{memory.tutor_output}\n\n"
            f"Student's Question to Solve: {memory.student_question}\n"
            f"Student's Proposed Answer: {memory.student_answer or 'None provided'}"
        )
        output = self._call(user_msg)
        memory.solver_output = output
        memory.log(self.name, output)
        return output


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 3 — EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────

class EvaluatorAgent(BaseAgent):
    """
    Critically reviews the solver's answer.
    Identifies errors, logical gaps, and unsupported claims. Never agrees blindly.
    """

    name = "evaluator"
    system_prompt_template = textwrap.dedent("""\
        You are a rigorous academic evaluator with zero tolerance for hand-waving.

        RULES you MUST follow:
        1. Never accept any claim without checking it against the problem and the steps shown.
        2. Compare the student's proposed answer with the solver's solution.
        3. Identify EVERY mathematical error, logical gap, or unsupported leap in the student's answer.
        4. Rate the student's answer: CORRECT / PARTIALLY CORRECT / INCORRECT, with justification.
        5. If the student's answer is incorrect, explain why and point out the precise step(s) that are wrong.
        6. If the student's answer is correct, still verify the solver's steps independently.

        Structure:
        ## Student Answer Review
        ## Step-by-Step Verification
        ## Errors & Gaps Found
        ## Overall Rating  [CORRECT | PARTIALLY CORRECT | INCORRECT]
        ## Suggested Corrections
    """)

    def run(self, memory: SessionMemory) -> str:
        user_msg = (
            f"Original Question: {memory.student_question}\n"
            f"Student's Proposed Answer: {memory.student_answer or 'None provided'}\n\n"
            f"Proposed Solution:\n{memory.solver_output}"
        )
        output = self._call(user_msg)
        memory.evaluator_output = output
        memory.log(self.name, output)
        return output


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 4 — FEEDBACK
# ─────────────────────────────────────────────────────────────────────────────

class FeedbackAgent(BaseAgent):
    """
    Synthesises the full discussion — tutor, solver and evaluator outputs —
    into encouraging, actionable feedback for the student.
    """

    name = "feedback"
    system_prompt_template = textwrap.dedent("""\
        You are a supportive learning coach who gives honest, constructive feedback.

        RULES you MUST follow:
        1. Read the Tutor explanation, the Solver's solution, and the Evaluator's critique.
        2. Synthesise these into ONE cohesive feedback message addressed directly to the student ("you").
        3. Open with genuine encouragement — celebrate what the student got right or attempted well.
        4. Be specific about what needs improvement — never vague ("study more").
        5. Give at least TWO concrete, actionable next steps the student can do TODAY.
        6. Close on a motivating note that reinforces a growth mindset.
        7. Keep the tone warm, honest, and never condescending.

        Structure:
        ## What Went Well
        ## Areas for Improvement
        ## Actionable Steps for Today
        ## Keep Going!
    """)

    def run(self, memory: SessionMemory) -> str:
        user_msg = (
            f"--- Tutor Explanation ---\n{memory.tutor_output}\n\n"
            f"--- Solver's Solution ---\n{memory.solver_output}\n\n"
            f"--- Evaluator's Critique ---\n{memory.evaluator_output}\n\n"
            f"Student's original question: {memory.student_question}\n"
            f"Student's Proposed Answer: {memory.student_answer or 'None provided'}"
        )
        output = self._call(user_msg)
        memory.feedback_output = output
        memory.log(self.name, output)
        return output


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 5 — PLANNER
# ─────────────────────────────────────────────────────────────────────────────

class PlannerAgent(BaseAgent):
    """
    Creates a structured 3-step next-learning plan based on the session outcome
    and the provided curriculum context.
    """

    name = "planner"
    system_prompt_template = textwrap.dedent("""\
        You are an expert curriculum designer and study coach.

        RULES you MUST follow:
        1. Base the plan on THREE inputs: what went well, what went wrong, and the curriculum context.
        2. Produce EXACTLY 3 concrete learning steps, numbered Step 1 / Step 2 / Step 3.
        3. Each step must include:
           a. A clear learning objective (what the student will be able to do).
           b. A specific resource or activity type (e.g., "watch a 10-min video on X", "practice 5 problems of type Y").
           c. A success criterion (how the student knows they've mastered it).
        4. Steps must be sequenced — each builds on the previous.
        5. Align steps with the curriculum so nothing is out of scope.

        Structure:
        ## Learning Plan
        ### Step 1 — [Title]
        ### Step 2 — [Title]
        ### Step 3 — [Title]
        ## Estimated Time Investment
    """)

    def run(self, memory: SessionMemory) -> str:
        user_msg = (
            f"Curriculum Context: {memory.curriculum_hints or 'General mathematics / science'}\n\n"
            f"What Went Well (from Feedback):\n{memory.feedback_output}\n\n"
            f"Evaluator Findings (what went wrong):\n{memory.evaluator_output}\n\n"
            f"Student's Proposed Answer: {memory.student_answer or 'None provided'}\n"
            f"Original Topic: {memory.topic}"
        )
        output = self._call(user_msg)
        memory.plan_output = output
        memory.log(self.name, output)
        return output


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 6 — ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class OrchestratorAgent:
    """
    Coordinates the full multi-agent pipeline.
    Monitors each agent's output, detects emergent behaviour, and can trigger
    a re-evaluation loop when quality signals fall below thresholds.
    """

    # Patterns that signal problematic outputs (emergence / drift detection)
    _RED_FLAGS = [
        r"\bi don'?t know\b",
        r"\bcannot (help|assist)\b",
        r"\brefuse\b",
        r"N/?A",
        r"\bno (steps?|solution)\b",
        r"error|exception",          # raw Python errors leaking into LLM output
    ]
    _FLAG_RE = re.compile("|".join(_RED_FLAGS), re.IGNORECASE)

    _POSITIVE_FLAGS = [
        r"\bclear(?:ly)?\b",
        r"\belegant\b",
        r"\bdeep insight\b",
        r"\bstrong reasoning\b",
        r"\bwell explained\b",
        r"\bexcellent\b",
        r"\bcreative\b",
        r"\bnovel approach\b",
    ]
    _POSITIVE_RE = re.compile("|".join(_POSITIVE_FLAGS), re.IGNORECASE)

    # Quality gate: minimum word count per agent output
    _MIN_WORDS = {
        "tutor":          80,
        "problem_solver": 100,
        "evaluator":      60,
        "feedback":       60,
        "planner":        80,
    }

    MAX_RETRIES = 2   # how many times to retry a failing agent

    def __init__(self, llm: LLMBackbone):
        self.llm = llm
        self.agents: dict[str, BaseAgent] = {
            "tutor":          TutorAgent(llm),
            "problem_solver": ProblemSolverAgent(llm),
            "evaluator":      EvaluatorAgent(llm),
            "feedback":       FeedbackAgent(llm),
            "planner":        PlannerAgent(llm),
        }

    # ── private helpers ──────────────────────────────────────────────────────

    def _check_output(self, memory: SessionMemory, agent_name: str, output: str) -> list[str]:
        """Return a list of quality issues (empty = OK)."""
        issues: list[str] = []
        word_count = len(output.split())
        min_words   = self._MIN_WORDS.get(agent_name, 50)

        if word_count < min_words:
            issues.append(f"Output too short: {word_count} words (min {min_words})")

        flags = self._FLAG_RE.findall(output)
        if flags:
            issues.append(f"Suspicious patterns detected: {set(flags)}")

        return issues

    def _run_agent_with_retry(
        self, agent_name: str, memory: SessionMemory
    ) -> tuple[str, bool]:
        """Run an agent, retrying up to MAX_RETRIES times if quality checks fail."""
        agent = self.agents[agent_name]
        for attempt in range(1, self.MAX_RETRIES + 1):
            output = agent.run(memory)
            issues = self._check_output(memory, agent_name, output)
            if not issues:
                return output, True
            warn = f"[Orchestrator] {agent_name} attempt {attempt} flagged: {issues}"
            print(f"\n⚠  {warn}")
            memory.emergence_negative.append(warn)
            memory.emergence_flags.append(warn)
            if attempt == self.MAX_RETRIES:
                failure_msg = (
                    f"[Orchestrator] {agent_name} FAILED quality gate after {self.MAX_RETRIES} attempts."
                )
                memory.emergence_negative.append(failure_msg)
                memory.emergence_flags.append(failure_msg)
                return output, False   # return best effort
        return "", False

    def _detect_emergence(self, memory: SessionMemory) -> None:
        """
        Look for emergent cross-agent patterns:
        - Evaluator unexpectedly agreeing with a clearly wrong solution
        - Planner plan contradicting Evaluator findings
        - Severe topic drift between Tutor framing and Solver approach
        """
        positive_issues: list[str] = list(memory.emergence_positive)
        negative_issues: list[str] = list(memory.emergence_negative)

        # 1. Evaluator blind-agreement check
        if re.search(r"\bCORRECT\b", memory.evaluator_output, re.I):
            if re.search(r"\bno (errors?|issues?|problems?|gaps?)\b", memory.evaluator_output, re.I):
                negative_issues.append(
                    "EMERGENCE: Evaluator declared solution fully correct with no issues found "
                    "— verify this is not a blind pass-through."
                )
            elif self._POSITIVE_RE.search(memory.evaluator_output):
                positive_issues.append(
                    "POSITIVE EMERGENCE: Evaluator offered a clear, confident validation of the solution."
                )

        # 2. Plan vs evaluator contradiction check
        evaluator_has_errors = bool(
            re.search(r"(error|incorrect|gap|wrong|mistake|unsupported)", memory.evaluator_output, re.I)
        )
        plan_ignores_errors  = not bool(
            re.search(r"(correct|fix|review|revisit|practice)", memory.plan_output, re.I)
        )
        if evaluator_has_errors and plan_ignores_errors:
            negative_issues.append(
                "EMERGENCE: Evaluator found errors but Planner's plan does not address remediation."
            )
        elif evaluator_has_errors and not plan_ignores_errors:
            positive_issues.append(
                "POSITIVE EMERGENCE: Planner appropriately addressed the evaluator's concerns."
            )

        # 3. Topic drift: tutor topic vs solver topic (simple keyword check)
        tutor_keywords  = set(re.findall(r"\b[a-zA-Z]{5,}\b", memory.topic.lower()))
        solver_text_kw  = set(re.findall(r"\b[a-zA-Z]{5,}\b", memory.solver_output.lower()))
        overlap = tutor_keywords & solver_text_kw
        if tutor_keywords and len(overlap) / len(tutor_keywords) < 0.2:
            negative_issues.append(
                f"EMERGENCE: Possible topic drift — solver output shares only "
                f"{len(overlap)}/{len(tutor_keywords)} topic keywords with the stated topic."
            )
        elif tutor_keywords and len(overlap) / len(tutor_keywords) >= 0.5:
            positive_issues.append(
                "POSITIVE EMERGENCE: Solver output remains well-aligned with the stated topic."
            )

        # 4. Positive signal detection in high-quality agent outputs
        for label, text in [
            ("Solver", memory.solver_output),
            ("Feedback", memory.feedback_output),
            ("Planner", memory.plan_output),
        ]:
            if self._POSITIVE_RE.search(text):
                positive_issues.append(
                    f"POSITIVE EMERGENCE: {label} produced an especially clear or insightful response."
                )

        memory.emergence_negative = negative_issues
        memory.emergence_positive = positive_issues
        memory.emergence_flags = negative_issues + positive_issues

        self._print_emergence_report(positive_issues, negative_issues)

    def _print_emergence_report(self, positive_issues: list[str], negative_issues: list[str]) -> None:
        print("\n" + "═" * 70)
        print("  ORCHESTRATOR — EMERGENCE REPORT")
        print("═" * 70)
        if positive_issues:
            print("  Positive emergence:")
            for item in positive_issues:
                print(f"   • {item}")
            print("─" * 70)
        if negative_issues:
            print("  Negative emergence:")
            for item in negative_issues:
                print(f"   • {item}")
            print("─" * 70)
        if not positive_issues and not negative_issues:
            print("  No emergence detected.")
            print("─" * 70)
        print("═" * 70)

    # ── public interface ─────────────────────────────────────────────────────

    def run_session(self, memory: SessionMemory) -> SessionMemory:
        """
        Execute the full collaborative pipeline in order:
        Tutor → Problem Solver → Evaluator → Feedback → Planner
        With orchestrator checks and emergence detection at the end.
        """
        memory.iteration += 1
        pipeline = [
            ("tutor",          "TUTOR — Topic Breakdown"),
            ("problem_solver", "PROBLEM SOLVER — Step-by-Step Solution"),
            ("evaluator",      "EVALUATOR — Critical Review"),
            ("feedback",       "FEEDBACK AGENT — Student Feedback"),
            ("planner",        "PLANNER — Next 3-Step Learning Plan"),
        ]

        print("\n" + "═" * 70)
        print(f"  ORCHESTRATOR — SESSION {memory.iteration} START")
        print(f"  Topic   : {memory.topic}")
        print(f"  Question: {memory.student_question}")
        print(f"  Student's Proposed Answer: {memory.student_answer or 'None provided'}")
        print("═" * 70)

        for agent_name, label in pipeline:
            print(f"\n{'─' * 70}")
            print(f"  AGENT: {label}")
            print(f"{'─' * 70}")
            output, passed = self._run_agent_with_retry(agent_name, memory)
            status = "✓" if passed else "⚠ (quality gate not fully met)"
            print(f"\n{output}\n")
            print(f"  [Orchestrator] {agent_name} → {status}")

        # Post-pipeline: emergence / anomaly detection
        self._detect_emergence(memory)

        print("\n" + "═" * 70)
        print("  ORCHESTRATOR — SESSION COMPLETE")
        if memory.emergence_positive or memory.emergence_negative:
            print(f"  Positive emergence items: {len(memory.emergence_positive)}")
            print(f"  Negative emergence items: {len(memory.emergence_negative)}")
            print(f"  Total emergence items: {len(memory.emergence_flags)}")
        else:
            print("  No emergence detected.")
        print("═" * 70 + "\n")

        return memory


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLI
# ─────────────────────────────────────────────────────────────────────────────

def _banner() -> None:
    print(textwrap.dedent("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║        Multi-Agent Educational AI — Collaborative Solver        ║
    ║  Agents: Tutor | Solver | Evaluator | Feedback | Planner        ║
    ║  Coordinated by: Orchestrator                                   ║
    ╚══════════════════════════════════════════════════════════════════╝
    """))


def main() -> None:
    _banner()

    # ── LLM Configuration ────────────────────────────────────────────────────
    print("Configure LLM backbone:")
    print("  [1] Groq   — llama-3.3-70b-versatile  ✓ FREE  → https://console.groq.com")
    print("  [2] Gemini — gemini-1.5              ✓ FREE  → https://aistudio.google.com")
    print("  [3] OpenAI — gpt-4o                    $ PAID  (set OPENAI_API_KEY)")
    print("  [4] Anthropic — claude-3-5-sonnet      $ PAID  (set ANTHROPIC_API_KEY)")
    choice = input("Choose [1/2/3/4] (default 1 — Groq free): ").strip() or "1"

    if choice == "2":
        api_key = os.getenv("GEMINI_API_KEY", "") or input("Paste your Gemini API key: ").strip()
        model_name = input("Gemini model name (default: gemini-1.5): ").strip() or "gemini-1.5"
        cfg = LLMConfig(
            provider  = "gemini",
            model     = model_name,
            api_key   = api_key,
        )
    elif choice == "3":
        api_key = os.getenv("OPENAI_API_KEY", "") or input("Paste your OpenAI API key: ").strip()
        cfg = LLMConfig(
            provider  = "openai",
            model     = "gpt-4o",
            api_key   = api_key,
        )
    elif choice == "4":
        api_key = os.getenv("ANTHROPIC_API_KEY", "") or input("Paste your Anthropic API key: ").strip()
        cfg = LLMConfig(
            provider  = "anthropic",
            model     = "claude-3-5-sonnet-latest",
            api_key   = api_key,
        )
    else:
        api_key = os.getenv("GROQ_API_KEY", "") or input("Paste your Groq API key: ").strip()
        cfg = LLMConfig(
            provider  = "groq",
            model     = "llama-3.3-70b-versatile",
            api_key   = api_key,
        )

    llm          = LLMBackbone(cfg)
    orchestrator = OrchestratorAgent(llm)

    # ── Session Input ─────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("Enter session details (press Enter to use examples):")
    equation = input(
        "Equation/problem to solve (e.g. 'x^2 + 5x + 6 = 0'): "
    ).strip() or "x^2 + 5x + 6 = 0"

    topic = equation

    question = input(
        "What question are you asking about this equation? (e.g. 'How do I solve it?'): "
    ).strip() or f"How do I solve {equation}?"

    student_answer = input(
        "Your answer to the question (what you think is correct): "
    ).strip() or "x = -2, x = -3"

    curriculum = input(
        "Curriculum context (e.g. 'Grade 10 Algebra, unit 3'): "
    ).strip() or "Grade 10 Algebra, Unit 3 — Polynomial Expressions"

    # ── Build shared memory & run ─────────────────────────────────────────────
    memory = SessionMemory(
        topic            = topic,
        student_question = question,
        student_answer   = student_answer,
        curriculum_hints = curriculum,
    )

    orchestrator.run_session(memory)

    # ── Optional: print full pipeline log ────────────────────────────────────
    show_log = input("Show full pipeline JSON log? [y/N]: ").strip().lower()
    if show_log == "y":
        import json
        print(json.dumps(memory.pipeline_log, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
