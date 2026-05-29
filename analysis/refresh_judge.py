"""
Re-run the LLM-as-judge GWA loadings on a set of OLMES items.

Reads:
  data/items.jsonl          — produced by analysis/fetch_olmes_items.py.
                              One row per item with {task, doc_id, prompt, choices}.
Writes:
  data/judge_loadings_refreshed.jsonl  — one row per (item, judge, perspective).

Costs real money. Default behaviour is to do ONE call so you can verify your
API keys work before committing to the full run.

Run:
  source .env
  python -m analysis.refresh_judge                # 1 call (smoke test)
  python -m analysis.refresh_judge --n 10         # 10 items × 2 perspectives × 1 judge
  python -m analysis.refresh_judge --judge claude --perspective automation --n 50
  python -m analysis.refresh_judge --all          # full re-run on all items in items.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


# ── Models ──────────────────────────────────────────────────────────────────

MODELS = {
    "claude": "claude-opus-4-5-20251101",
    "openai": "gpt-5.2-2025-12-11",
}


# ── O*NET GWAs (27 cognitive activities, Shao et al. 2025 Appendix E.6) ─────

@dataclass(frozen=True)
class Concept:
    id: str
    name: str
    definition: str


CONCEPTS_GWA: list[Concept] = [
    Concept("organizing_planning_prioritizing", "Organizing, Planning, and Prioritizing Work",
            "Developing specific goals and plans to prioritize, organize, and accomplish your work."),
    Concept("training_teaching", "Training and Teaching Others",
            "Identifying the educational needs of others, developing formal educational or training programs or classes, and teaching or instructing others."),
    Concept("staffing_organizational_units", "Staffing Organizational Units",
            "Recruiting, interviewing, selecting, hiring, and promoting employees in an organization."),
    Concept("updating_using_knowledge", "Updating and Using Relevant Knowledge",
            "Keeping up-to-date technically and applying new knowledge to your job."),
    Concept("developing_objectives_strategies", "Developing Objectives and Strategies",
            "Establishing long-range objectives and specifying the strategies and actions to achieve them."),
    Concept("guiding_directing_motivating", "Guiding, Directing, and Motivating Subordinates",
            "Providing guidance and direction to subordinates, including setting performance standards and monitoring performance."),
    Concept("judging_qualities", "Judging the Qualities of Objects, Services, or People",
            "Assessing the value, importance, or quality of things or people."),
    Concept("communicating_internally", "Communicating with Supervisors, Peers, or Subordinates",
            "Providing information to supervisors, coworkers, and subordinates by telephone, in written form, e-mail, or in person."),
    Concept("providing_consultation", "Providing Consultation and Advice to Others",
            "Providing guidance and expert advice to management or other groups on technical, systems-, or process-related topics."),
    Concept("thinking_creatively", "Thinking Creatively",
            "Developing, designing, or creating new applications, ideas, relationships, systems, or products, including artistic contributions."),
    Concept("interpreting_information", "Interpreting the Meaning of Information for Others",
            "Translating or explaining what information means and how it can be used."),
    Concept("making_decisions_solving_problems", "Making Decisions and Solving Problems",
            "Analyzing information and evaluating results to choose the best solution and solve problems."),
    Concept("monitoring_processes", "Monitoring Processes, Materials, or Surroundings",
            "Monitoring and reviewing information from materials, events, or the environment, to detect or assess problems."),
    Concept("assisting_caring", "Assisting and Caring for Others",
            "Providing personal assistance, medical attention, emotional support, or other personal care to others such as coworkers, customers, or patients."),
    Concept("getting_information", "Getting Information",
            "Observing, receiving, and otherwise obtaining information from all relevant sources."),
    Concept("monitoring_controlling_resources", "Monitoring and Controlling Resources",
            "Monitoring and controlling resources and overseeing the spending of money."),
    Concept("analyzing_data", "Analyzing Data or Information",
            "Identifying the underlying principles, reasons, or facts of information by breaking down information or data into separate parts."),
    Concept("selling_influencing", "Selling or Influencing Others",
            "Convincing others to buy merchandise/goods or to otherwise change their minds or actions."),
    Concept("documenting_recording", "Documenting/Recording Information",
            "Entering, transcribing, recording, storing, or maintaining information in written or electronic/magnetic form."),
    Concept("evaluating_compliance", "Evaluating Information to Determine Compliance with Standards",
            "Using relevant information and individual judgment to determine whether events or processes comply with laws, regulations, or standards."),
    Concept("communicating_externally", "Communicating with People Outside the Organization",
            "Communicating with people outside the organization, representing the organization to customers, the public, government, and other external sources. The information can be exchanged in person, in writing, or by telephone or e-mail."),
    Concept("processing_information", "Processing Information",
            "Compiling, coding, categorizing, calculating, tabulating, auditing, or verifying information or data."),
    Concept("estimating_quantifiable_characteristics", "Estimating the Quantifiable Characteristics of Products, Events, or Information",
            "Estimating sizes, distances, and quantities; or determining time, costs, resources, or materials needed to perform a work activity."),
    Concept("performing_administrative", "Performing Administrative Activities",
            "Performing day-to-day administrative tasks such as maintaining information files and processing paperwork."),
    Concept("performing_for_public", "Performing for or Working Directly with the Public",
            "Performing for people or dealing directly with the public. This includes serving customers in restaurants and stores, and receiving clients or guests."),
    Concept("scheduling_work", "Scheduling Work and Activities",
            "Scheduling events, programs, and activities, as well as the work of others."),
    Concept("establishing_maintaining_relationships", "Establishing and Maintaining Interpersonal Relationships",
            "Developing constructive and cooperative working relationships with others, and maintaining them over time."),
]


# ── Prompt templates ────────────────────────────────────────────────────────
# Two-step (sub_steps → ranked_gwa_ids) 3-shot reasoning prompts. The
# metacognitive decomposition forces the judge to think from first principles
# about the underlying cognitive work, not the surface MCQ format.

PROMPT_AUTOMATION = """\
An AI system is deployed to handle this task autonomously, with no human \
involvement. Which work activities must the AI exercise to complete this \
item end-to-end?

Work in two steps inside the JSON output below:

STEP 1 — In `sub_steps`, decompose the item from first principles into the \
granular sub-tasks an AI would have to do to arrive at the correct answer. \
Think about the underlying problem, not the surface format (e.g. that it's \
multiple choice).

STEP 2 — In `ranked_gwa_ids`, rank up to 5 GWAs from the list below that \
benefit from being able to do those sub-tasks, ordered from most to least \
central.

Benchmark Item:
{task_text}

---

GWA List:
{gwa_list}

---

Example:

Benchmark Item:
The students in a class would like to make 20 paper sailboats for a race. \
The students will select one design and collect the materials they need to \
construct the boats. Which of the following is the best way for the students \
to be sure the paper sailboats will float without tipping over in the water?
  A) construct a prototype of a boat for testing
  B) calculate the total mass of all of the finished boats
  C) determine the total amount of weight each boat can carry
  D) test the strength of each material used to construct the boats

JSON output:
{{
  "sub_steps": [
    "identify what is being asked: choose the best method to verify a design's seaworthiness before scaling to 20 boats",
    "recognize that the goal is to validate floating and stability of one chosen design, not material or aggregate properties",
    "recall standard scientific methodology: build a small-scale prototype and observe it under realistic conditions before scaling up",
    "evaluate each option as a method (build-and-test, compute total mass, load-capacity test, material-strength test) and check which actually predicts floating-without-tipping",
    "rule out total mass, capacity, and material strength as indirect proxies; only the prototype tests the integrated behavior",
    "select option A"
  ],
  "ranked_gwa_ids": [
    "organizing_planning_prioritizing",
    "updating_using_knowledge",
    "estimating_quantifiable_characteristics",
    "analyzing_data",
    "making_decisions_solving_problems"
  ],
  "rationale": "The decisive sub-task is recalling the standard prototyping methodology (updating/applying knowledge) and recognizing it as the best plan among the alternatives (organizing/prioritizing). Some sub-steps require estimating and analyzing physical properties (mass, capacity) of the candidate methods, and the final selection is a problem-solving step that picks the best of four imperfect proxies."
}}

---

Example:

Benchmark Item:
Three bells ring at intervals of 36 seconds, 40 seconds and 48 seconds, \
respectively. They start ringing together at a particular time. When will \
they ring together again?
  A) After 6 minutes
  B) After 12 minutes
  C) After 18 minutes
  D) After 24 minutes
  E) none

JSON output:
{{
  "sub_steps": [
    "recognize that all three bells ring together again at the least common multiple (LCM) of their intervals",
    "factorize each interval: 36 = 2^2 * 3^2, 40 = 2^3 * 5, 48 = 2^4 * 3",
    "compute the LCM by taking the maximum power of each prime: 2^4 * 3^2 * 5 = 16 * 9 * 5 = 720 seconds",
    "convert 720 seconds to minutes: 720 / 60 = 12 minutes",
    "match against the answer choices: 12 minutes corresponds to option B",
    "select option B"
  ],
  "ranked_gwa_ids": [
    "making_decisions_solving_problems",
    "processing_information",
    "analyzing_data",
    "scheduling_work",
    "estimating_quantifiable_characteristics"
  ],
  "rationale": "The decisive sub-task is computing the LCM of three intervals (processing/analyzing the prime factorizations) and selecting the correct multiple (problem-solving). The problem is fundamentally about scheduling repeating events, which makes that GWA salient even though the format is arithmetic. Estimating quantitative characteristics supports the conversion from seconds to minutes."
}}

---

Example:

Benchmark Item:
Self-efficacy (the belief that one has control over one's situation) as it \
related to job satisfaction was studied. When a group of teachers rated their \
ability to control their situation and their satisfaction with their job, the \
two variables had a correlation of 0.30. Which statement follows from this \
correlation?
  A) If you want teachers to be happy with their job, give them more control over their situation.
  B) If you want teachers to take more control over their situation, make them happier at their jobs.
  C) These two variables show a moderate negative correlation.
  D) Higher self-efficacy and higher job satisfaction tend to occur together, but neither causes the other.

JSON output:
{{
  "sub_steps": [
    "parse the setup: self-efficacy and job satisfaction were measured in a group of teachers, with r = 0.30 between them",
    "recall the meaning of the Pearson correlation coefficient: 0.30 is a positive linear association of moderate-to-weak strength, not a causal claim",
    "evaluate option A (efficacy implies satisfaction causal claim): unsupported by correlation alone, eliminate",
    "evaluate option B (satisfaction implies efficacy causal claim): also unsupported, eliminate",
    "evaluate option C (negative correlation): factually wrong because 0.30 is positive, eliminate",
    "evaluate option D (co-occurrence without causation): correctly restates what r = 0.30 implies",
    "select option D"
  ],
  "ranked_gwa_ids": [
    "analyzing_data",
    "making_decisions_solving_problems",
    "interpreting_information",
    "processing_information",
    "updating_using_knowledge"
  ],
  "rationale": "The decisive sub-task is correctly interpreting what a correlation coefficient does and does not imply. Eliminating each answer choice against that interpretation is a problem-solving step over the option set, and recalling the standard 'correlation does not imply causation' rule grounds the analysis in statistical knowledge."
}}

---

Return exactly this JSON (no other text):
{{
  "sub_steps": ["<sub-task>", "<sub-task>", ...],
  "ranked_gwa_ids": ["<gwa_id_1>", "<gwa_id_2>", ...],
  "rationale": "<2–3 sentences linking the ranked GWAs back to the sub-steps>"
}}
"""

PROMPT_AUGMENTATION = """\
An AI system is deployed alongside a human worker to jointly tackle this \
task — the human remains involved throughout, ranging from light oversight \
to close collaboration. Which work activities does the AI most need to be \
capable of in this collaborative setting?

Work in two steps inside the JSON output below:

STEP 1 — In `sub_steps`, imagine interacting with a user on this item and \
walk through the exchange concretely: what the user would ask, what the AI \
would do, how they go back and forth.

STEP 2 — In `ranked_gwa_ids`, rank up to 5 GWAs from the list below that \
correlate with the AI being good at this kind of interaction, ordered from \
most to least central.

Benchmark Item:
{task_text}

---

GWA List:
{gwa_list}

---

Example:

Benchmark Item:
An entrepreneur from State A decided to sell hot sauce to the public, \
labeling it "Best Hot Sauce." A company incorporated in State B and \
headquartered in State C sued the entrepreneur in federal court in State C. \
The complaint sought $50,000 in damages and alleged that the entrepreneur's \
use of the name "Best Hot Sauce" infringed the company's federal trademark. \
The entrepreneur filed an answer denying the allegations, and the parties \
began discovery. Six months later, the entrepreneur moved to dismiss for \
lack of subject-matter jurisdiction. Should the court grant the \
entrepreneur's motion?
  A) No, because the company's claim arises under federal law.
  B) No, because the entrepreneur waived the right to challenge subject-matter jurisdiction by not raising the issue initially by motion or in the answer.
  C) Yes, because although the claim arises under federal law, the amount in controversy is not satisfied.
  D) Yes, because although there is diversity, the amount in controversy is not satisfied.

JSON output:
{{
  "sub_steps": [
    "user pastes the fact pattern and four answer choices, asking which is correct",
    "AI identifies that this is a federal subject-matter jurisdiction question and surfaces the two grounds (federal question vs. diversity)",
    "AI flags the salient facts: federal trademark claim, $50,000 amount, parties from different states",
    "AI checks the live legal rule — federal-question jurisdiction has no amount-in-controversy threshold; SMJ can be raised at any time",
    "user asks whether the timing of the motion matters; AI confirms it does not for SMJ and cites the relevant procedural rule",
    "AI recommends option A and explains why federal-question grounds jurisdiction directly"
  ],
  "ranked_gwa_ids": [
    "evaluating_compliance",
    "updating_using_knowledge",
    "interpreting_information",
    "providing_consultation",
    "communicating_internally"
  ],
  "rationale": "The interaction centers on applying a specific legal rule (federal-question jurisdiction) to the facts — checking compliance with the doctrinal requirements is the dominant AI move, supported by recalling the relevant rule (updating knowledge) and parsing what the fact pattern actually says (interpreting information). The user's questions and the AI's explanation form a consultative back-and-forth."
}}

---

Example:

Benchmark Item:
Self-efficacy (the belief that one has control over one's situation) as it \
related to job satisfaction was studied. When a group of teachers rated their \
ability to control their situation and their satisfaction with their job, the \
two variables had a correlation of 0.30. Which statement follows from this \
correlation?
  A) If you want teachers to be happy with their job, give them more control over their situation.
  B) If you want teachers to take more control over their situation, make them happier at their jobs.
  C) These two variables show a moderate negative correlation.
  D) Higher self-efficacy and higher job satisfaction tend to occur together, but neither causes the other.

JSON output:
{{
  "sub_steps": [
    "user shares the question and asks the AI to help reason through which option is correct",
    "AI clarifies what r = 0.30 means: a modest positive linear association, not strong, and far from causation",
    "AI walks the user through why a correlation does not establish a causal direction in either direction",
    "AI helps the user evaluate options A and B as causal claims (both unsupported by correlation alone) and option C as factually wrong (0.30 is positive, not negative)",
    "user asks 'so what is left?' — AI confirms option D restates the correlation without overclaiming causation",
    "AI explains the underlying statistical principle (correlation does not imply causation) so the user retains the reasoning, not just the answer"
  ],
  "ranked_gwa_ids": [
    "interpreting_information",
    "providing_consultation",
    "communicating_internally",
    "analyzing_data",
    "updating_using_knowledge"
  ],
  "rationale": "The interaction is fundamentally consultative: the AI walks the user through the meaning of a correlation coefficient and the limits of causal inference. Interpreting the statistical concept and communicating it back to the user dominates the AI's contribution; analytic and knowledge-recall steps support that consultation."
}}

---

Example:

Benchmark Item:
Fertilizer from an agricultural area runs off into a river. The river carries \
the nutrients from this fertilizer and deposits them into an ocean bay. After \
the nutrients enter the bay, scientists monitoring the water would most \
likely see a decrease in which of these dissolved gases?
  A) oxygen
  B) nitrogen
  C) carbon dioxide
  D) carbon monoxide

JSON output:
{{
  "sub_steps": [
    "user pastes the scenario and answer choices and asks the AI to explain the underlying mechanism and pick an answer",
    "AI walks the user through the eutrophication chain: nutrient runoff causes an algal bloom, the bloom dies, microbial decomposition consumes oxygen, and hypoxia results",
    "AI rules out the other gases for the user: nitrogen is the input not the output, carbon dioxide is produced by decomposition (would increase, not decrease), carbon monoxide is not a typical aquatic gas",
    "user asks why this matters in practice — AI explains the broader environmental impact (dead zones, fish kills) so the answer is grounded in real-world consequences",
    "AI confirms option A (oxygen) and offers to elaborate on related concepts such as algal blooms or water-quality monitoring if the user wants to learn more"
  ],
  "ranked_gwa_ids": [
    "providing_consultation",
    "communicating_internally",
    "interpreting_information",
    "updating_using_knowledge",
    "evaluating_compliance"
  ],
  "rationale": "The interaction is the AI explaining a multi-step environmental mechanism to the user — consultative teaching plus communication of the reasoning chain. Recalling the eutrophication process is the knowledge backbone; helping the user evaluate which dissolved gas is affected is interpretive."
}}

---

Return exactly this JSON (no other text):
{{
  "sub_steps": ["<interaction move>", "<interaction move>", ...],
  "ranked_gwa_ids": ["<gwa_id_1>", "<gwa_id_2>", ...],
  "rationale": "<2–3 sentences linking the ranked GWAs back to the interaction>"
}}
"""

SYSTEM_PROMPT = (
    "You are an expert AI benchmark evaluator specializing in workforce "
    "capability analysis. Your job is to assess how well benchmark items "
    "measure capabilities relevant to real-world work activities (O*NET "
    "Generalized Work Activities). "
    "You must return only valid JSON matching the schema provided."
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def render_prompt(item: dict, perspective: str) -> str:
    gwa_list = "\n".join(
        f"  - {c.id}: {c.name} — {c.definition}" for c in CONCEPTS_GWA
    )
    task_text = item["prompt"]
    if item.get("choices"):
        task_text += "\n\nChoices:\n" + "\n".join(
            f"  ({chr(ord('A') + i)}) {c}" for i, c in enumerate(item["choices"])
        )
    template = PROMPT_AUTOMATION if perspective == "automation" else PROMPT_AUGMENTATION
    return template.format(task_text=task_text, gwa_list=gwa_list)


def _extract_json(content: str) -> str:
    """Strip markdown fences and prose; return the JSON substring."""
    if content.startswith("```"):
        lines = content.splitlines()
        inner = [line for line in lines[1:] if line.strip() != "```"]
        content = "\n".join(inner).strip()
    try:
        json.loads(content)
        return content
    except (json.JSONDecodeError, ValueError):
        pass
    start = content.find("{")
    if start == -1:
        return content
    depth = 0
    for i in range(start, len(content)):
        if content[i] == "{":
            depth += 1
        elif content[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = content[start:i + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except (json.JSONDecodeError, ValueError):
                    break
    return content


def call_judge(model: str, system: str, user: str) -> dict:
    """Call an LLM and return parsed JSON."""
    import litellm
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
    }
    response = litellm.completion(**kwargs)
    content = response.choices[0].message.content or ""
    return json.loads(_extract_json(content))


def load_items(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"missing: {path}\n"
            f"Run `python -m analysis.fetch_olmes_items` first to materialize items.jsonl."
        )
    rows = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="claude", choices=list(MODELS),
                    help="Which judge to call (default: claude).")
    ap.add_argument("--perspective", default="both",
                    choices=["automation", "augmentation", "both"],
                    help="Which framing(s) to score (default: both).")
    ap.add_argument("--n", type=int, default=1,
                    help="Number of items to score (default: 1, smoke test).")
    ap.add_argument("--all", action="store_true",
                    help="Score every item in data/items.jsonl (overrides --n).")
    ap.add_argument("--items", type=Path, default=DATA / "items.jsonl")
    ap.add_argument("--out", type=Path,
                    default=DATA / "judge_loadings_refreshed.jsonl")
    args = ap.parse_args()

    api_key_env = "ANTHROPIC_API_KEY" if args.judge == "claude" else "OPENAI_API_KEY"
    if not os.environ.get(api_key_env):
        raise SystemExit(
            f"missing env var: {api_key_env}\n"
            f"Copy env.example to .env, fill in keys, and `source .env`."
        )

    items = load_items(args.items)
    if not args.all:
        items = items[: args.n]

    perspectives = ["automation", "augmentation"] if args.perspective == "both" \
        else [args.perspective]
    model = MODELS[args.judge]

    n_calls = len(items) * len(perspectives)
    print(f"Running {n_calls} judge calls "
          f"({len(items)} items × {len(perspectives)} perspectives, judge={args.judge})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_err = 0
    t0 = time.time()
    with args.out.open("w") as out_f:
        for item in items:
            for perspective in perspectives:
                user = render_prompt(item, perspective)
                t_call = time.time()
                try:
                    result = call_judge(model, SYSTEM_PROMPT, user)
                    ok = True
                    n_ok += 1
                except Exception as e:
                    print(f"  [ERR] {item['task']}/{item['doc_id']} {perspective}: {e}",
                          file=sys.stderr)
                    result = {}
                    ok = False
                    n_err += 1
                row = {
                    "task": item["task"],
                    "doc_id": item["doc_id"],
                    "judge": args.judge,
                    "model": model,
                    "perspective": perspective,
                    "sub_steps": result.get("sub_steps", []),
                    "ranked_gwa_ids": result.get("ranked_gwa_ids", []),
                    "rationale": result.get("rationale", ""),
                    "elapsed_s": round(time.time() - t_call, 3),
                    "ok": ok,
                }
                out_f.write(json.dumps(row) + "\n")
                out_f.flush()

    print(f"\n{n_ok} ok, {n_err} errors, {time.time()-t0:.1f}s")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
