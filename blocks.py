import pandas as pd


def build_activity_blocks(transcript_df: pd.DataFrame, cls_df: pd.DataFrame, enum: list[str]) -> pd.DataFrame:
    joined = transcript_df.merge(cls_df[["phrase_rank", "activity"]], on="phrase_rank", how="left")
    joined["speaker_text"] = "[" + joined["speaker"].astype(str) + "]: " + joined["text"].astype(str)

    blocks = joined.groupby("activity", dropna=False).agg(duration_sec=("duration_sec", "sum")).reset_index()

    exchanges = (
        joined.groupby("activity", dropna=False)["speaker_text"]
        .apply(lambda x: "\n".join(x))
        .reset_index(name="exchange")
    )

    observed = blocks.merge(exchanges, on="activity", how="left")

    # Keep missing activities visible, in enum order
    full = pd.DataFrame({"activity": enum})
    out = full.merge(observed, on="activity", how="left")

    out["present"] = out["duration_sec"].notna()
    out["duration_sec"] = out["duration_sec"].fillna(0.0).astype(float)
    out["exchange"] = out["exchange"].fillna("").astype(str)

    out["activity_order"] = pd.Categorical(out["activity"], categories=enum, ordered=True)
    out = out.sort_values("activity_order").drop(columns=["activity_order"]).reset_index(drop=True)

    return out
