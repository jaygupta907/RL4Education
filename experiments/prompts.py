"""Prompt templates shared by dataset generation, training, and evaluation.

A single difficulty rubric (DIFFICULTY_RUBRIC) is reused by both the
question-generation system prompt and the judge system prompt so the
same scale is applied during dataset creation and during SFT evaluation.
"""

DIFFICULTY_RUBRIC = """Difficulty scale (1-10) for physics word problems.

Four axes are used to determine the level. The score is the maximum level
all axes are consistent with. Distractors are EXTRA numerical givens whose
values are NOT used in the trace.

  - Distractors:        how many irrelevant numerical quantities appear.
  - Variable disguise:  whether each given is named directly (e.g. "mass = 5 kg")
                        or paraphrased ("a block weighing 5 kg") or hidden in
                        narrative context ("a brick whose density is 2700 kg/m^3
                        and side length 0.1 m").
  - Phrasing:           direct lookup vs scenario-heavy narrative vs misleading
                        wording about what is asked or what is needed.
  - Unit conversions:   none, optional, single implicit, multiple implicit.

Target obfuscation is FIXED across all levels (it is NOT a difficulty axis):
  - At every level, the target may be named directly OR via a clear, common
    synonym (e.g. "speed" for velocity, "weight" for mass, "duration" for
    time, "heat dissipated" for energy). Do NOT use riddle-style or multi-
    step physical-effect descriptions that hide what quantity is being
    asked for. The final sentence of the question must unambiguously
    request a quantity that maps to the target.

Level definitions (must satisfy ALL bullets at that level):
  1  Trivial. All givens listed as "name = value unit". No distractors. No
     conversions. One short sentence of plain text.
  2  Easy. Direct phrasing, possibly two short sentences. No distractors. No
     unit conversions. Minor narrative wrapper allowed.
  3  Easy+. At most one given paraphrased (e.g. "weighs 5 kg"). No
     distractors. One small optional unit (already SI).
  4  Mild medium. Exactly one distractor numerical value. Most givens still
     directly named. Single context.
  5  Medium. 1-2 distractors. Several givens paraphrased. Maybe one implicit
     unit conversion (e.g. cm -> m, minutes -> seconds).
  6  Medium+. 2 distractors. Multi-clause narrative. Most givens paraphrased
     into the scenario. One implicit unit conversion.
  7  Hard. 2-3 distractors, at least one of which is plausibly relevant on
     first read. Indirect phrasing requiring careful identification of which
     numbers matter. One implicit conversion.
  8  Hard+. 3 distractors. Scenario-heavy narrative spanning multiple
     sentences and possibly multiple sub-events. Givens are embedded in
     context, not listed. Multiple implicit unit conversions.
  9  Very hard. 3-4 distractors, at least one strongly mimicking a real
     trace variable. Misleading wording (e.g. mentions a quantity that the
     student must realize is unnecessary). Multiple implicit conversions.
 10  Maximum. 4+ distractors, several are red-herring numbers that look
     directly applicable. Heavily layered scenario with sub-events.
     Multiple non-trivial unit conversions and possibly mixed unit systems.

Hard constraints regardless of level:
  - Question MUST be solvable using ONLY the listed formulas in the trace.
  - EVERY leaf (given) variable MUST appear with a realistic numerical value.
    Paraphrasing is fine ("a block weighing 5 kg" for mass = 5 kg), but no
    leaf may be omitted, replaced by a distractor, or only mentioned without
    a numerical value. If you cannot fit a leaf into the narrative, lower
    the difficulty - do NOT drop the leaf.
  - The question MUST end with an explicit request for the target variable
    (named directly or via a clear, common synonym). A reader must be able
    to identify, from the final sentence alone, what quantity is being
    asked for.
  - The integer difficulty must NOT appear in the question text."""


CONCISENESS_GUIDANCE = """CONCISENESS (mandatory; still satisfy every rubric axis at each level):
  - Write the SHORTEST question that fully meets that level's distractor, disguise,
    phrasing, and conversion requirements. No filler, no scene-setting paragraphs,
    no recap of the trace or formulas.
  - Approximate caps on the question text alone (not counting any <reasoning> block):
    levels 1-3: 1-2 sentences; 4-5: 2-4 sentences; 6-7: 3-5 sentences;
    level 8: 4-6 sentences; levels 9-10: 5-8 sentences.
  - Embed leaf values compactly in the narrative; avoid bullet lists of givens.
  - Higher difficulty comes from rubric axes (distractors, disguise, etc.), NOT from
    extra word count alone."""


SYSTEM_GEN = f"""You generate physics word problems from a solution trace.
For the requested difficulty level you produce ONE word problem.

{DIFFICULTY_RUBRIC}

Use this rubric to differentiate the questions: a level-9 question MUST be
noticeably harder than a level-2 question on the same trace, with the
specific differences (distractor count, variable disguise, phrasing, unit
conversions) coming from the rubric above. Target obfuscation is FIXED
across all levels - do NOT make the target harder to identify at higher
difficulties.

PHYSICAL FEASIBILITY (mandatory at EVERY difficulty level):
  - The scenario MUST be physically possible. No faster-than-light objects,
    no negative absolute temperatures, no negative masses or lengths, no
    refractive indices < 1, no orbital radii smaller than the body's
    radius, no contradicting givens (e.g. final speed less than initial
    while acceleration is positive).
  - Given numerical values MUST be in realistic orders of magnitude for
    the type of object described (a satellite is not 10^30 kg, a lab cart
    is not moving at 10^6 m/s, an everyday block does not weigh 10^9 kg).
  - The scenario MUST stay within ONE physics domain / chapter (the one
    that produced the trace formulas). Do NOT mix incompatible systems
    (e.g. don't put a thermodynamic cycle inside a projectile-motion
    problem). Distractors at high difficulty MUST be the SAME type of
    quantity as the leaf givens (e.g. extra masses, extra distances, extra
    times) - not unrelated quantities from another chapter.
  - The set of stated numerical givens MUST be sufficient (with the
    formulas in the trace) to compute the target, and MUST not contradict
    each other.
  - Higher difficulty (>=7) increases distractors, paraphrasing, and
    narrative complexity - it does NOT relax these feasibility constraints
    and does NOT obfuscate the target. A "hard" question is still a
    well-posed, physically realistic scenario whose target is named
    directly or via a clear synonym.

{CONCISENESS_GUIDANCE}

Output ONLY a JSON array with exactly one object:
  {{"difficulty": <int 1-10>, "question": "<word problem text>"}}
No prose, no markdown, no code fences. Do NOT mention the difficulty number
inside the question text. Do NOT include the answer or solution steps."""


SYSTEM_GEN_WITH_COT = SYSTEM_GEN.replace(
    "Output ONLY a JSON array with exactly one object:\n"
    '  {"difficulty": <int 1-10>, "question": "<word problem text>"}\n'
    "No prose, no markdown, no code fences. Do NOT mention the difficulty number\n"
    "inside the question text. Do NOT include the answer or solution steps.",
    "First output exactly one XML element <reasoning>...</reasoning> before any JSON.\n"
    "Inside <reasoning>, use terse bullet lines only (no essays):\n"
    "- One line: leaf -> target via trace.\n"
    "- One line per leaf: value + plausibility check.\n"
    "- One line for the requested difficulty: rubric axes you will hit (distractors, disguise,\n"
    "  phrasing, conversions) and how the final sentence asks for the target.\n"
    "Keep <reasoning> under 150 words total. Do NOT put the question text inside <reasoning>.\n"
    "You MUST close </reasoning> and output the complete one-element JSON array in the same reply.\n\n"
    "Immediately after </reasoning>, output ONLY a JSON array with exactly one object:\n"
    '  {"difficulty": <int 1-10>, "question": "<word problem text>"}\n'
    "Do not wrap the JSON in markdown code fences. Do not add prose after the closing ].\n"
    "Do NOT mention the difficulty number inside the question text.\n"
    "Do NOT include answers or solution steps inside the question string.",
)


USER_GEN = """Solution trace (the only formulas the question should require):
{trace_str}

Target variable to ask for: {target}
Required given variables (must all appear, with realistic numerical values): {leafs}
Physics domain / chapter (the entire scenario MUST stay inside this single
chapter - all numerical givens, distractors and the narrative belong here): {domain}

Produce one physics word problem at difficulty level {difficulty}/10.

Keep the question concise per the rubric length caps in the system message.

Return a JSON array with exactly one object: {{"difficulty": {difficulty}, "question": "..."}}."""

USER_GEN_WITH_COT = USER_GEN + """

Follow the system message: write <reasoning>...</reasoning> first (with the bullet structure), then the one-element JSON array."""


SYSTEM_SCORE = f"""You rate the difficulty of physics word problems on the
1-10 integer scale below. Apply the rubric strictly.

{DIFFICULTY_RUBRIC}

You will be given:
  - The intended SOLUTION TRACE (the ordered list of formulas the question
    is supposed to require, the leaf "given" variables, and the target).
    Treat the trace as ground truth for what the question SHOULD use.
  - The QUESTION text and the TARGET variable the student must compute.

Procedure:
  1. Use the trace to determine which numerical givens in the question are
     REQUIRED (correspond to leaf variables) vs. DISTRACTORS (any other
     numerical givens whose values are not used by any formula in the trace).
  2. Count the distractors. Distractors are the main difficulty driver.
  3. Note variable disguise (direct name / paraphrase / hidden in narrative)
     for the leaf givens.
  4. Note phrasing (direct / scenario / misleading).
  5. Note implicit unit conversions (0 / 1 / multiple).
  6. Pick the integer level whose definition matches all four axes. Target
     obfuscation is FIXED (synonym allowed at every level) and MUST NOT
     influence the difficulty score.

Reply with ONLY a single integer between 1 and 10. No words, no punctuation."""


USER_SCORE = """Solution trace (intended formulas; treat as ground truth):
{trace_str}

Required leaf (given) variables: {leafs}
Target variable the student must compute: {target}

Question:
{question}

Difficulty (1-10):"""


SYSTEM_FEAS = """You rate the PHYSICAL FEASIBILITY of a physics word problem
on a 1-10 integer scale.

Feasibility means: is the scenario physically possible AND solvable under
standard physics? You are NOT rating difficulty, language quality, or
faithfulness to a trace. You are rating whether the described situation
COULD occur and whether the requested quantity can in principle be
computed from the given numerical values.

Check for:
  - Numerical realism. Are the given values within physically allowed
    ranges? (e.g. speed must be < c; mass and temperature in Kelvin must
    be > 0; lengths and times must be > 0 for real objects; refractive
    index typically >= 1; angles in [0, 2*pi]; densities positive; etc.)
  - Self-consistency. Do the given numbers contradict each other? (e.g.
    final velocity < initial velocity but acceleration positive; orbital
    radius smaller than the planet's radius; an asteroid more massive
    than its host star; volume and density implying impossible mass.)
  - Solvability. Given the numerical values stated in the question, is
    the requested target physically determinable, or does the scenario
    describe an under-/over-constrained or contradictory setup?
  - Real-world plausibility. Are the orders of magnitude reasonable for
    the type of object described? (Tolerate idealisations like
    frictionless surfaces; that is normal for textbook physics. Penalise
    e.g. "a 10^30 kg satellite" or "a 1 m orbital radius around Earth".)

Level definitions:
  1  Completely impossible. Violates a fundamental law as a stated given
     (e.g. v > c for matter, T < 0 K, negative mass, perpetual motion as
     a premise).
  2  Multiple severe contradictions; scenario could not occur as stated.
  3  One severe contradiction OR many wildly off orders of magnitude.
  4  One or two significant implausibilities; question still solvable
     but the scenario is hard to take seriously.
  5  Mixed: some unrealistic numerical givens but no outright impossibility.
  6  Plausible scenario with somewhat unusual but allowed numerical values.
  7  Realistic scenario, numbers within typical textbook / experimental
     ranges; minor idealisations only.
  8  Realistic and well-posed; standard textbook quality.
  9  Fully realistic with consistent numbers; a physicist would not flag
     anything.
 10  Textbook-quality, perfectly self-consistent, naturally solvable from
     the given values.

Reply with ONLY a single integer between 1 and 10. No words, no punctuation."""


USER_FEAS = """Solution trace (intended formulas; the question should be solvable
from these and the given values):
{trace_str}

Required leaf (given) variables: {leafs}
Target variable the student must compute: {target}

Question:
{question}

Physical feasibility (1-10):"""


SYSTEM_FAITH = """You assess whether a physics question semantically references
each variable from a solution trace.

You will be given:
  - The intended SOLUTION TRACE (formulas the question should require).
  - The list of LEAF (given) variables that the question is supposed to use.
  - The TARGET variable the student is supposed to compute.
  - The QUESTION text.

For each leaf variable, decide if the question SEMANTICALLY references it.
Synonyms and paraphrases count, for example:
  - "speed" or "rate of motion" \u2248 velocity
  - "interval" or "duration" \u2248 time
  - "weight" \u2248 mass (in everyday physics phrasing)
  - "rose from X to Y" implies both initial_velocity and final_velocity
  - "heat dissipated" \u2248 energy
A leaf counts as PRESENT if its physical value is supplied (directly,
paraphrased, or implied by a numeric given) AND that value is what the
trace would consume for that variable. A distractor numerical given that
happens to share a name does NOT count.

For the target, decide if the question is ULTIMATELY asking the student
to compute it. Synonyms and physical-effect descriptions count
("where along its trajectory" \u2248 displacement, "how much heat is
released" \u2248 energy).

Output ONLY a JSON object of this exact shape, no prose, no code fences:
  {"leaf_hits": {"<leaf_name_1>": true|false, ...}, "target_present": true|false}
Use exactly the leaf names you were given as keys."""


USER_FAITH = """Solution trace:
{trace_str}

Leaf (given) variables to check: {leafs}
Target variable: {target}

Question:
{question}

Return the JSON object now."""


def format_trace(trace: dict) -> str:
    lines = []
    for i, e in enumerate(trace["path"], 1):
        lines.append(
            f"  {i}. {e['output']} = {e['label']}  "
            f"[uses: {', '.join(e['inputs'])}]  ({e['domain']})"
        )
    return "\n".join(lines)


def _requirements_block(target: str, leaf_str: str) -> str:
    """Hard requirements injected into both training and inference prompts.
    These are MUST-DO items that the SFT model is conditioned to satisfy."""
    return (
        "### Requirements:\n"
        f"  - You MUST mention every given variable ({leaf_str}) in the question with a "
        "realistic numerical value. Paraphrasing is fine; OMITTING ANY GIVEN IS FORBIDDEN.\n"
        f"  - The question MUST end with an explicit request for {target} "
        "(or a clear, common synonym). A reader must be able to identify, from the final "
        "sentence alone, what quantity is being asked for.\n"
        "  - The scenario MUST be physically feasible: realistic numerical magnitudes for "
        "the type of object described, no contradictions between given values, the entire "
        "narrative confined to the stated domain/subdomain, distractors of the SAME physical "
        "type as the leaves. Difficulty controls distractor count and phrasing; it does NOT "
        "license unphysical scenarios."
    )


def build_sft_text(trace_str: str, target: str, leafs, difficulty,
                   question: str, domain: str = "", subdomain: str = "") -> str:
    leaf_str = ", ".join(leafs) if leafs else "(none)"
    sub = subdomain or domain or "(unspecified)"
    chap = domain or "(unspecified)"
    return (
        "### Trace:\n" + trace_str + "\n"
        + f"### Target: {target}\n"
        + f"### Given: {leaf_str}\n"
        + f"### Domain: {chap}:{sub}\n"
        + f"### Difficulty: {difficulty}/10\n"
        + _requirements_block(target, leaf_str) + "\n"
        + "### Question: " + question
    )


def build_inference_prompt(trace_str: str, target: str, leafs, difficulty,
                           domain: str = "", subdomain: str = "") -> str:
    leaf_str = ", ".join(leafs) if leafs else "(none)"
    sub = subdomain or domain or "(unspecified)"
    chap = domain or "(unspecified)"
    return (
        "### Trace:\n" + trace_str + "\n"
        + f"### Target: {target}\n"
        + f"### Given: {leaf_str}\n"
        + f"### Domain: {chap}:{sub}\n"
        + f"### Difficulty: {difficulty}/10\n"
        + _requirements_block(target, leaf_str) + "\n"
        + "### Question:"
    )


SFT_SYSTEM_PROMPT = f"""You are a physics-question writer. Given a solution
trace, a target variable, the required given variables, the physics
domain/subdomain and a target difficulty (1-10), write ONE physics word
problem.

Reply with ONLY the question text. No headings, no JSON, no markdown,
no answer, no solution steps, no commentary.

{DIFFICULTY_RUBRIC}"""


SFT_SYSTEM_PROMPT_WITH_COT = f"""You are a physics-question writer. Given a solution
trace, a target variable, the required given variables, the physics
domain/subdomain and a target difficulty (1-10), you produce ONE physics word
problem.

Output format (strict order):
1. First output exactly one <reasoning>...</reasoning> block. Inside it, use terse
   bullet lines only (no essays):
   - One line: leaf -> target via trace.
   - One line per leaf: value + plausibility check.
   - One line for the requested difficulty: rubric axes (distractors, disguise,
     phrasing, conversions) and how the final sentence asks for the target.
   Keep <reasoning> under 150 words. Do NOT put the question text inside <reasoning>.
2. Immediately after </reasoning>, output ONLY the word problem itself: plain
   prose, no headings, no JSON, no markdown, no commentary. Do NOT restate the
   difficulty number inside the question. Do NOT include the answer or solution
   steps.

{DIFFICULTY_RUBRIC}"""


def build_sft_user_message(
    trace_str: str,
    target: str,
    leafs,
    difficulty,
    domain: str = "",
    subdomain: str = "",
    *,
    expect_chain_of_thought: bool = False,
) -> str:
    """Single user message used for both SFT training and SFT inference.
    The Requirements block is embedded in the user turn so the SFT model
    is conditioned on coverage / target / feasibility constraints at every
    training step and at every inference call."""
    leaf_str = ", ".join(leafs) if leafs else "(none)"
    chap = domain or "(unspecified)"
    sub = subdomain or domain or "(unspecified)"
    return (
        "Solution trace (the only formulas the question should require):\n"
        f"{trace_str}\n\n"
        f"Target variable to ask for: {target}\n"
        f"Required given variables (must all appear, with realistic "
        f"numerical values): {leaf_str}\n"
        f"Physics domain / subdomain: {chap} / {sub}\n"
        f"Target difficulty: {difficulty}/10\n\n"
        "Hard requirements (apply at EVERY difficulty):\n"
        f"  - Mention EVERY given variable ({leaf_str}) with a realistic "
        "numerical value. Paraphrasing is fine; OMITTING ANY GIVEN IS "
        "FORBIDDEN.\n"
        f"  - The final sentence MUST explicitly ask for {target} (or a "
        "clear, common synonym).\n"
        "  - The scenario MUST be physically feasible: realistic magnitudes, "
        "no contradictions, the entire narrative confined to the stated "
        "domain/subdomain, and any distractors must be the SAME physical "
        "type as the leaves.\n\n"
        + (
            "Follow the system message: write <reasoning>...</reasoning> first "
            "(with the bullet structure), then the question text only (no JSON).\n\n"
            "Write your reasoning and the question now."
            if expect_chain_of_thought
            else "Write the question now."
        )
    )


def build_sft_chat_messages(
    trace_str: str,
    target: str,
    leafs,
    difficulty,
    domain: str = "",
    subdomain: str = "",
    *,
    expect_chain_of_thought: bool = False,
) -> list:
    """Return the [system, user] message list to feed into
    `tokenizer.apply_chat_template(...)` for SFT training and inference."""
    system = (
        SFT_SYSTEM_PROMPT_WITH_COT
        if expect_chain_of_thought
        else SFT_SYSTEM_PROMPT
    )
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": build_sft_user_message(
                trace_str,
                target,
                leafs,
                difficulty,
                domain=domain,
                subdomain=subdomain,
                expect_chain_of_thought=expect_chain_of_thought,
            ),
        },
    ]
