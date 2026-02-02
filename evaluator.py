import json
import pandas as pd
from llm_client import get_openai
from text_utils import sanitize_text


def build_eval_messages(rubric: dict, activity: str, exchange: str, metadata: dict):
    system = {
        "role": "system",
        "content": (
            "You are an expert QA evaluator for telecommunications calls. "
            "Score the agent's performance for the given activity using the rubric. "
            "Provide evidence-based observations and tactical coaching."
        ),
    }
    user = {
        "role": "user",
        "content": f"""
<RULES>
1) Use only the exchange and rubric provided.
2) Be objective; do not be overly critical.
3) No nulls.
4) Return efficacy_score as a percentage from 0 to 100 (not 0–1).
</RULES>

<METADATA>
{json.dumps(metadata)}
</METADATA>

<ACTIVITY>
{activity}
</ACTIVITY>

<RUBRIC>
{json.dumps(rubric)}
</RUBRIC>

<EXCHANGE>
{exchange}
</EXCHANGE>

<OUTPUT>
Return JSON:
{{
  "activity": "string",
  "efficacy_score": "number",
  "detailed_observations": "string",
  "tactical_feedback": "string"
}}
</OUTPUT>
""".strip(),
    }
    return [system, user]


def evaluate_blocks(blocks_df: pd.DataFrame, rubric: dict, metadata: dict) -> pd.DataFrame:
    client, model = get_openai()

    schema = {
        "name": "activity_evaluation",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "activity": {"type": "string"},
                "efficacy_score": {"type": "number", "minimum": 0, "maximum": 100},
                "detailed_observations": {"type": "string"},
                "tactical_feedback": {"type": "string"},
            },
            "required": ["activity", "efficacy_score", "detailed_observations", "tactical_feedback"],
        },
    }

    rows = []
    for _, r in blocks_df.iterrows():
        activity = str(r["activity"])
        present = bool(r.get("present", True))
        exchange = sanitize_text(r.get("exchange", ""))

        if (not present) or (not exchange.strip()):
            expected = rubric.get(activity, {}) if isinstance(rubric, dict) else {}
            exp_good = expected.get("good", "") if isinstance(expected, dict) else ""
            exp_bad = expected.get("bad", "") if isinstance(expected, dict) else ""

            obs = "Activity not present in the call transcript for this interaction."
            fb_parts = []
            if exp_good:
                fb_parts.append(f"Expected: {exp_good}")
            if exp_bad:
                fb_parts.append(f"Avoid: {exp_bad}")
            fb = " ".join(fb_parts) if fb_parts else "Ensure this step is completed when applicable."

            rows.append(
                {
                    "activity": activity,
                    "present": False,
                    "duration_sec": float(r.get("duration_sec", 0.0)),
                    "efficacy_score": 0.0,
                    "detailed_observations": obs,
                    "tactical_feedback": fb,
                }
            )
            continue

        resp = client.chat.completions.create(
            model=model,
            messages=build_eval_messages(rubric, activity, exchange, metadata),
            response_format={"type": "json_schema", "json_schema": schema},
        )
        parsed = json.loads(resp.choices[0].message.content)

        score = float(parsed.get("efficacy_score", 0.0))
        if 0 <= score <= 1:
            score *= 100.0

        rows.append(
            {
                "activity": activity,
                "present": True,
                "duration_sec": float(r.get("duration_sec", 0.0)),
                "efficacy_score": score,
                "detailed_observations": parsed.get("detailed_observations", ""),
                "tactical_feedback": parsed.get("tactical_feedback", ""),
            }
        )

    out = pd.DataFrame(rows)
    out["activity_order"] = pd.Categorical(out["activity"], categories=blocks_df["activity"].tolist(), ordered=True)
    out = out.sort_values("activity_order").drop(columns=["activity_order"]).reset_index(drop=True)

    return out