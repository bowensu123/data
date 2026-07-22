"""
Prompt templates for each MLLM/LLM call in the pipeline.

The paper (Section 4) specifies *what* each call must accomplish but not the
exact wording of its prompts (that detail is not public). These templates
are my own reconstruction, written to satisfy exactly the stated
requirements of each step; treat their wording as a reasonable default that
you will likely want to tune against your own MLLM's behavior.

Every function returns a plain string prompt. Callers are responsible for
attaching the actual video / frames as separate multimodal content blocks
(see llm_client.py) — text and vision are kept apart here so the same
templates work whether frames are sampled client-side or the API accepts a
raw video blob.
"""

# --------------------------------------------------------------------------- #
# Stage 2: QA Synthesis
# --------------------------------------------------------------------------- #

SCENE_SEGMENTATION_PROMPT = """You are annotating a video for training a streaming video assistant.

Watch the provided video (already sampled at {fps} FPS, duration {duration:.1f}s) and segment it
into contiguous scenes that capture the temporal progression of events.

For each scene, output:
- start_s: scene start time in seconds (float, 0 <= start_s < duration)
- end_s: scene end time in seconds (float, start_s < end_s <= duration)
- description: one to three sentences describing what happens in this scene

Return ONLY a JSON list of objects with keys "start_s", "end_s", "description".
Scenes must be contiguous and cover the full video (no gaps, no overlaps).
"""

CANDIDATE_QA_RT_PROACTIVE_PROMPT = """You are constructing streaming question-answering data from a video.

You are given the video and its scene-level descriptions:
{scenes_json}

Generate up to {max_qas} candidate question-answer pairs of two kinds:

1. "real_time": the question could naturally be asked at some timestamp q_time, and can be
   answered immediately using only information available up to and including q_time.
   In this case a_time MUST equal q_time.

2. "proactive": the question is naturally asked at q_time, but the answer only becomes
   determinable later, once enough future evidence has accumulated by a_time > q_time
   (e.g. "will the driver run the red light?", "is the pot going to boil over?").

Ground every question and answer strictly in the observable video content. Prefer questions
a real user would ask while watching live (open-ended, not just "what color is the shirt").

Return ONLY a JSON list of objects with keys:
"type" ("real_time" or "proactive"), "question", "answer", "q_time_s", "a_time_s".
"""

VERIFY_RT_QA_PROMPT = """You are verifying a candidate Real-Time QA sample for a streaming video dataset.

You are given the video content up to and including timestamp {a_time:.1f}s (the answer time),
and the candidate pair:
Question: {question}
Answer: {answer}
Timestamp (q_time = a_time): {a_time:.1f}s

Check all of the following:
1. Is the question a reasonable thing to ask while watching up to this point in the video?
2. Is the answer correctly and fully supported by the video content up to this timestamp?
3. Is the timestamp accurate (i.e., the necessary visual evidence is present by this time,
   not before it appears and not requiring later content)?

Return ONLY a JSON object: {{"pass": true|false, "reason": "<short justification>"}}.
"""

VERIFY_PROACTIVE_QA_PROMPT = """You are verifying a candidate Proactive QA sample for a streaming video dataset.

You are given the video content up to and including timestamp {a_time:.1f}s (the answer time),
and the candidate pair:
Question: {question}
Answer: {answer}
Question time: {q_time:.1f}s
Answer time: {a_time:.1f}s (a_time > q_time)

Check all of the following:
1. Can the question naturally be raised at q_time (i.e. is it not already answerable then)?
2. Is there sufficient information in the video by a_time to support the answer?
3. Is the answer correctly and fully supported by the video content up to a_time?

Return ONLY a JSON object: {{"pass": true|false, "reason": "<short justification>"}}.
"""

# --- Multi-Response QA sub-pipeline ---

CANDIDATE_MULTI_QUESTION_PROMPT = """You are constructing "Multi-Response QA" data: one question about an ongoing
event that should receive several valid answers as the event evolves.

Video clip: [{clip_start:.1f}s, {clip_end:.1f}s]
Scene description: {scene_description}

Propose up to {max_questions} candidate questions about a state or event in this clip that
plausibly changes or accumulates evidence over the clip's duration (e.g. "how many people have
entered the room so far?", "what is the cat doing now?"). For each, propose a natural q_time_s
within [{clip_start:.1f}, {clip_end:.1f}] at which a user could first ask it.

Return ONLY a JSON list of objects with keys "question", "q_time_s".
"""

CHECK_MULTI_ANSWERABLE_PROMPT = """Video clip: [{clip_start:.1f}s, {clip_end:.1f}s]
Candidate question (asked at t={q_time:.1f}s): {question}

Decide:
1. Is this a reasonable question to ask about this clip?
2. Can it be given multiple distinct, valid answers at different timestamps within
   [{q_time:.1f}, {clip_end:.1f}] as the depicted event evolves (not just one static answer)?

Return ONLY a JSON object: {{"pass": true|false, "reason": "<short justification>"}}.
"""

GENERATE_MULTI_ANSWERS_PROMPT = """Video clip: [{clip_start:.1f}s, {clip_end:.1f}s]
Question (asked at t={q_time:.1f}s): {question}

Track how the answer to this question changes over time in the clip. Produce a list of
(answer, timestamp) pairs, one for each moment where the correct answer meaningfully changes
or updates, timestamps in [{q_time:.1f}, {clip_end:.1f}].

Return ONLY a JSON list of objects with keys "text", "timestamp_s".
"""

VERIFY_MULTI_ANSWER_PROMPT = """Video clip content up to and including t={timestamp:.1f}s.
Question: {question}
Candidate answer at this timestamp: {answer}

Check both of the following:
1. Is this answer correctly inferable from the video content up to this timestamp?
2. Does it appropriately answer the question (not vague, not contradicted by the video)?

Return ONLY a JSON object: {{"pass": true|false, "reason": "<short justification>"}}.
"""

# --------------------------------------------------------------------------- #
# Stage 3: QA Refinement
# --------------------------------------------------------------------------- #

GENERATE_DIFFICULTY_SIBLINGS_PROMPT = """You are given a video (content up to and including t={t:.1f}s) and an
existing question anchored at that same timestamp:

Original question: {question}

Generate 4 NEW questions, also anchored at t={t:.1f}s, that progress across these difficulty
levels (one question per level):
1. perception          - simple perceptual recognition ("what color is X", "is X present")
2. recognition          - naming / identifying an object, action, or attribute
3. understanding         - explaining what is happening or why, at a basic level
4. reasoning / advanced_reasoning - inferring intent, predicting consequences, or multi-step
   reasoning over what has been observed so far

Return ONLY a JSON list of 4 objects with keys "difficulty" (one of "perception",
"recognition", "understanding", "reasoning", "advanced_reasoning") and "question".
"""

GENERATE_ANSWER_FOR_QUESTION_PROMPT = """Video content up to and including t={t:.1f}s.
Question: {question}

Answer the question accurately and concisely, using only information visible/inferable from
the video up to this timestamp.

Return ONLY a JSON object: {{"answer": "<answer text>"}}.
"""

REWRITE_QUESTION_PROMPT = """Rewrite the following question into a semantically equivalent form, following
this style template: "{template}"

Original question: {question}

Rules:
- Preserve the original meaning exactly.
- Keep key entities, actions, and temporal references (e.g. "just now", "since I started",
  specific objects/people) unchanged.
- Only vary surface phrasing / register to match the template's style.

Return ONLY a JSON object: {{"question": "<rewritten question>"}}.
"""

# Candidate phrasing templates (Section 4.3, "predefine a set of candidate templates").
# These are intentionally style/register templates, not content — content is preserved
# by REWRITE_QUESTION_PROMPT above.
PROACTIVE_QUESTION_TEMPLATES = [
    "Direct question: '{}'",
    "Casual/spoken register, as if asking a friend: '{}'",
    "Polite request framing, e.g. starting with 'Could you let me know...': '{}'",
    "Urgent/concerned framing, e.g. starting with 'Quick — ...': '{}'",
    "Third-person framing about what will happen: '{}'",
]

MULTI_RESPONSE_QUESTION_TEMPLATES = [
    "Ongoing-monitoring framing, e.g. 'Keep me posted on ...': '{}'",
    "Direct question: '{}'",
    "Casual/spoken register: '{}'",
    "Explicit repeat-tracking framing, e.g. 'Let me know each time ...': '{}'",
    "Polite request framing: '{}'",
]

# --------------------------------------------------------------------------- #
# Stage 5: Quality Verification (post window-truncation)
# --------------------------------------------------------------------------- #

QUALITY_VERIFY_RT_PROMPT = """You are quality-checking a truncated training sample for a streaming video
assistant. You are given exactly what the model would see:
- the frames of the retained video window [{window_start:.1f}s, {window_end:.1f}s] (attached)
- the retained QA history (text only, outside the video window): {qa_history}

Target answer to check (produced at t={target_time:.1f}s): {target_answer}
(In response to: {target_question})

Judge whether the target answer is:
1. visually grounded in the retained video window (or supported by the retained QA history),
2. factually correct given that content,
3. temporally consistent (nothing it claims requires evidence outside the retained window),
4. free of hallucination.

Return ONLY a JSON object: {{"pass": true|false, "reason": "<short justification>"}}.
"""

QUALITY_VERIFY_PROACTIVE_MULTI_PROMPT = """You are quality-checking a truncated training sample for a streaming video
assistant. You are given exactly what the model would see:
- the frames of the retained video window [{window_start:.1f}s, {window_end:.1f}s] (attached)
- the retained QA history (text only, outside the video window): {qa_history}

Target answer to check (produced at t={target_time:.1f}s): {target_answer}
(In response to: {target_question}, asked at t={query_time:.1f}s)

Judge whether:
1. the response timing is temporally appropriate given the retained window/history, and
2. the content is accurate and grounded in the retained video context and QA history.

Return ONLY a JSON object: {{"pass": true|false, "reason": "<short justification>"}}.
"""
