import io
import zipfile
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AutoEIT Score Consolidator",
    page_icon="🎙️",
    layout="wide",
)

st.title("AutoEIT Score Consolidator")
st.caption(
    "Upload a zip export from the AutoEIT platform. "
    "The app will consolidate multi-session participants by keeping each "
    "participant's **first attempt at every item**, then recalculate scores."
)

# ---------------------------------------------------------------------------
# ZIP type detection
# Mirrors the logic in combine_zips_v2.py:
#   - AUDIO_CSV: contains bio.csv + submissions.csv + .mp3 files
#   - CSV_ONLY:  contains bio.csv + submissions.csv (no .mp3 files)
#   - AUDIO_ONLY: contains .mp3 files but no bio.csv / submissions.csv
#   - UNKNOWN: none of the above
# ---------------------------------------------------------------------------
ZIP_TYPE_AUDIO_CSV = "AUDIO_CSV"
ZIP_TYPE_CSV_ONLY  = "CSV_ONLY"
ZIP_TYPE_AUDIO_ONLY = "AUDIO_ONLY"
ZIP_TYPE_UNKNOWN   = "UNKNOWN"


def detect_zip_type(zip_bytes: bytes) -> str:
    """Inspect ZIP contents and return one of the ZIP_TYPE_* constants."""
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = set(zf.namelist())
    except zipfile.BadZipFile:
        return ZIP_TYPE_UNKNOWN

    has_bio  = "bio.csv" in names
    has_subs = "submissions.csv" in names
    has_mp3  = any(n.lower().endswith(".mp3") for n in names)

    if has_bio and has_subs and has_mp3:
        return ZIP_TYPE_AUDIO_CSV
    if has_bio and has_subs:
        return ZIP_TYPE_CSV_ONLY
    if has_mp3:
        return ZIP_TYPE_AUDIO_ONLY
    return ZIP_TYPE_UNKNOWN


# ---------------------------------------------------------------------------
# Helper: extract session ID from audio_file_name
# e.g. "BVQEA-2026111078-683-1.mp3"  →  683
# ---------------------------------------------------------------------------
def extract_session_id(audio_file_name: str) -> int | None:
    """Return the numeric session ID (3rd dash-delimited segment)."""
    parts = str(audio_file_name).split("-")
    if len(parts) >= 3:
        try:
            return int(parts[2])
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Core consolidation
# ---------------------------------------------------------------------------
def consolidate(bio_df: pd.DataFrame, subs_df: pd.DataFrame):
    """
    Returns
    -------
    consolidated_bio  : pd.DataFrame  – one row per submitter_id
    consolidated_subs : pd.DataFrame  – one row per (submitter_id, item_index)
    summary           : pd.DataFrame  – per-participant summary for display
    """

    # --- Parse test_start_time so we can rank sessions chronologically ------
    bio = bio_df.copy()
    bio["test_start_time"] = pd.to_datetime(bio["test_start_time"], errors="coerce")

    # Rank each session for a participant (1 = earliest)
    bio["_session_rank"] = (
        bio.sort_values("test_start_time")
           .groupby("submitter_id")
           .cumcount() + 1
    )

    # Build a mapping: (submitter_id, session_id_from_filename) → session_rank
    # We need to know which numeric session ID in the filename corresponds to
    # which ranked session.  We do this by assigning a session_file_id to each
    # bio row using the submissions data.
    subs = subs_df.copy()
    subs["_session_id"] = subs["audio_file_name"].apply(extract_session_id)

    # For each (submitter_id, session_id) pair that appears in submissions,
    # find the bio row with the matching rank.
    # Strategy: for each submitter, collect the distinct session IDs that
    # appear in submissions (in ascending numeric order) and map them to the
    # bio session ranks (also ascending by start_time).
    session_rank_map: dict[tuple, int] = {}  # (submitter_id, session_id) → rank

    for subid, grp in subs.groupby("submitter_id"):
        # Unique session IDs in submissions, sorted ascending (lower ID = earlier)
        session_ids_sorted = sorted(grp["_session_id"].dropna().unique())
        # Bio rows for this participant, sorted by start_time
        bio_rows = bio[bio["submitter_id"] == subid].sort_values("test_start_time")
        for rank_idx, sess_id in enumerate(session_ids_sorted, start=1):
            session_rank_map[(subid, int(sess_id))] = rank_idx

    subs["_session_rank"] = subs.apply(
        lambda r: session_rank_map.get((r["submitter_id"], r["_session_id"])),
        axis=1,
    )

    # --- For each (submitter_id, item_index) keep lowest session rank --------
    subs_sorted = subs.sort_values(["submitter_id", "item_index", "_session_rank"])
    consolidated_subs = (
        subs_sorted
        .dropna(subset=["_session_rank"])
        .groupby(["submitter_id", "item_index"], sort=False)
        .first()
        .reset_index()
    )
    # Drop helper columns
    consolidated_subs = consolidated_subs.drop(
        columns=[c for c in consolidated_subs.columns if c.startswith("_")]
    )
    # Restore original column order
    consolidated_subs = consolidated_subs[subs_df.columns]

    # --- Recalculate mean_mer and mean_accuracy per participant --------------
    agg = (
        consolidated_subs
        .groupby("submitter_id")
        .agg(
            recalc_mean_mer=("mer", "mean"),
            recalc_mean_accuracy=("accuracy", "mean"),
            item_count=("item_index", "count"),
        )
        .reset_index()
    )

    # --- Build consolidated bio (one row per participant) --------------------
    # Use the first session row (lowest _session_rank = 1) for all metadata
    first_bio = (
        bio.sort_values("_session_rank")
           .groupby("submitter_id", sort=False)
           .first()
           .reset_index()
    )

    # Count total sessions per participant
    session_counts = bio.groupby("submitter_id").size().reset_index(name="_n_sessions")

    first_bio = first_bio.merge(session_counts, on="submitter_id", how="left")
    first_bio = first_bio.merge(agg, on="submitter_id", how="left")

    # Overwrite scores
    first_bio["mean_mer"] = first_bio["recalc_mean_mer"]
    first_bio["mean_accuracy"] = first_bio["recalc_mean_accuracy"]

    # Clear time fields for multi-session participants; add notes column
    multi_mask = first_bio["_n_sessions"] > 1

    first_bio["notes"] = ""
    first_bio.loc[multi_mask, "notes"] = (
        "consolidated from " + first_bio.loc[multi_mask, "_n_sessions"].astype(str) + " attempts"
    )
    first_bio.loc[multi_mask, "test_start_time"] = pd.NaT
    first_bio.loc[multi_mask, "test_end_time"] = pd.NaT
    first_bio.loc[multi_mask, "test_duration"] = ""

    # Restore original column order + append notes
    drop_cols = [c for c in first_bio.columns if c.startswith("_")] + [
        "recalc_mean_mer", "recalc_mean_accuracy", "item_count"
    ]
    first_bio = first_bio.drop(columns=drop_cols)
    original_cols = [c for c in bio_df.columns if c in first_bio.columns]
    extra_cols = [c for c in first_bio.columns if c not in bio_df.columns]
    consolidated_bio = first_bio[original_cols + extra_cols]

    # --- Summary table for display ------------------------------------------
    summary_rows = []
    for _, row in (
        bio.groupby("submitter_id")
        .agg(
            submitter_name=("submitter_name", "first"),
            n_sessions=("_session_rank", "max"),
        )
        .reset_index()
        .iterrows()
    ):
        subid = row["submitter_id"]
        n_sess = int(row["n_sessions"])
        item_count_row = agg.loc[agg["submitter_id"] == subid, "item_count"]
        n_items = int(item_count_row.values[0]) if not item_count_row.empty else 0
        summary_rows.append(
            {
                "submitter_id": subid,
                "name": row["submitter_name"],
                "sessions_found": n_sess,
                "consolidated_items": n_items,
                "multi_session": n_sess > 1,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["multi_session", "submitter_id"], ascending=[False, True]
    )

    return consolidated_bio, consolidated_subs, summary_df


# ---------------------------------------------------------------------------
# Build output CSV zip in memory
# ---------------------------------------------------------------------------
def build_output_zip(bio_df: pd.DataFrame, subs_df: pd.DataFrame) -> bytes:
    """Write bio.csv and submissions.csv into an in-memory zip and return the bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("bio.csv", bio_df.fillna("").to_csv(index=False, encoding="utf-8-sig"))
        zf.writestr("submissions.csv", subs_df.fillna("").to_csv(index=False, encoding="utf-8-sig"))
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Build consolidated audio zip in memory
# ---------------------------------------------------------------------------
def build_output_audio_zip(
    audio_zip_bytes: bytes,
    consolidated_subs_df: pd.DataFrame,
) -> tuple[bytes, list[str]]:
    """
    Copy only the audio files referenced in consolidated_subs_df into a new zip.

    Returns
    -------
    zip_bytes : bytes       – the consolidated audio zip
    missing   : list[str]  – filenames that were expected but absent from the input zip
    """
    keep = set(consolidated_subs_df["audio_file_name"].dropna())

    buf = io.BytesIO()
    missing: list[str] = []

    with zipfile.ZipFile(io.BytesIO(audio_zip_bytes)) as src_zf:
        available = set(src_zf.namelist())
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as dst_zf:
            for fname in keep:
                if fname in available:
                    dst_zf.writestr(fname, src_zf.read(fname))
                else:
                    missing.append(fname)

    buf.seek(0)
    return buf.read(), missing


# ---------------------------------------------------------------------------
# Derive output zip filename from uploaded filename
# Mirrors the input suffix:
#   foo_AUDIO_CSV.zip         → foo_AUDIO_CSV_consolidated.zip
#   foo_AUDIO_CSV_consolidated.zip  → foo_AUDIO_CSV_consolidated.zip  (idempotent)
#   foo_CSV.zip               → foo_CSV_consolidated.zip
#   foo_Audio.zip             → foo_Audio_consolidated.zip
#   foo.zip                   → foo_consolidated.zip
# ---------------------------------------------------------------------------
def output_zip_name(uploaded_name: str) -> str:
    """Derive the consolidated output filename from the uploaded filename."""
    stem = uploaded_name
    if stem.lower().endswith(".zip"):
        stem = stem[:-4]
    # Already consolidated — return as-is (idempotent)
    if stem.lower().endswith("_consolidated"):
        return f"{stem}.zip"
    return f"{stem}_consolidated.zip"


def audio_output_zip_name(uploaded_name: str) -> str:
    """
    Derive the audio output filename for an AUDIO_CSV zip.
    e.g. A1_Red_AOPEB_AUDIO_CSV.zip → A1_Red_AOPEB_Audio_consolidated.zip
    """
    stem = uploaded_name
    if stem.lower().endswith(".zip"):
        stem = stem[:-4]
    if stem.lower().endswith("_consolidated"):
        stem = stem[:-13]  # strip _consolidated before re-adding
    # Replace _AUDIO_CSV suffix (case-insensitive) with _Audio
    import re
    stem = re.sub(r"_AUDIO_CSV$", "_Audio", stem, flags=re.IGNORECASE)
    return f"{stem}_consolidated.zip"


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload your AutoEIT export zip",
    type="zip",
    help=(
        "Drag and drop or click to upload. Accepts:\n"
        "- **AUDIO_CSV zip** (bio.csv + submissions.csv + MP3s — current platform format)\n"
        "- **CSV-only zip** (bio.csv + submissions.csv)\n"
        "- **Audio-only zip** (MP3s only — for use with a separately uploaded CSV zip)\n\n"
        "The zip type is detected automatically from its contents."
    ),
)

# Second uploader — only shown when user uploads an audio-only zip, or a
# CSV-only zip without audio.  Hidden when an AUDIO_CSV zip is uploaded.
uploaded_secondary = None

if uploaded_file is not None:
    primary_bytes = uploaded_file.read()
    zip_type = detect_zip_type(primary_bytes)

    if zip_type == ZIP_TYPE_AUDIO_CSV:
        st.info("Detected: combined Audio + CSV zip (AUDIO_CSV format).", icon="ℹ️")

    elif zip_type == ZIP_TYPE_CSV_ONLY:
        st.info("Detected: CSV-only zip.", icon="ℹ️")
        uploaded_secondary = st.file_uploader(
            "Upload matching audio zip (optional)",
            type="zip",
            help="If you have a separate audio zip for this export, upload it here "
                 "to also receive a consolidated audio zip.",
            key="secondary_audio",
        )

    elif zip_type == ZIP_TYPE_AUDIO_ONLY:
        st.info("Detected: audio-only zip.", icon="ℹ️")
        uploaded_secondary = st.file_uploader(
            "Upload matching CSV zip (required)",
            type="zip",
            help="The uploaded file contains only audio. Upload the matching CSV zip "
                 "(with bio.csv and submissions.csv) to process scores.",
            key="secondary_csv",
        )

    elif zip_type == ZIP_TYPE_UNKNOWN:
        st.error(
            "Could not determine the zip type. The file must contain bio.csv and "
            "submissions.csv (for CSV processing) and/or .mp3 files (for audio)."
        )
        st.stop()

# ---------------------------------------------------------------------------
# Resolve which bytes are the CSV source and which are the audio source
# ---------------------------------------------------------------------------
csv_zip_bytes   = None
csv_zip_name    = None
audio_zip_bytes = None
audio_zip_name  = None

if uploaded_file is not None:
    if zip_type == ZIP_TYPE_AUDIO_CSV:
        csv_zip_bytes = primary_bytes
        csv_zip_name  = uploaded_file.name
        audio_zip_bytes = primary_bytes       # same zip — MP3s extracted from it
        audio_zip_name  = uploaded_file.name

    elif zip_type == ZIP_TYPE_CSV_ONLY:
        csv_zip_bytes = primary_bytes
        csv_zip_name  = uploaded_file.name
        if uploaded_secondary is not None:
            audio_zip_bytes = uploaded_secondary.read()
            audio_zip_name  = uploaded_secondary.name

    elif zip_type == ZIP_TYPE_AUDIO_ONLY:
        audio_zip_bytes = primary_bytes
        audio_zip_name  = uploaded_file.name
        if uploaded_secondary is not None:
            csv_zip_bytes = uploaded_secondary.read()
            csv_zip_name  = uploaded_secondary.name

# ---------------------------------------------------------------------------
# Process when we have CSV data
# ---------------------------------------------------------------------------
if csv_zip_bytes is not None:
    # --- Read CSVs from the zip ---------------------------------------------
    bio_raw  = None
    subs_raw = None
    try:
        with zipfile.ZipFile(io.BytesIO(csv_zip_bytes)) as zf:
            names = zf.namelist()
            missing_files = [f for f in ("bio.csv", "submissions.csv") if f not in names]
            if missing_files:
                st.error(
                    f"The zip is missing the following required file(s): "
                    f"{', '.join(missing_files)}"
                )
                st.stop()
            bio_raw  = pd.read_csv(zf.open("bio.csv"),          dtype={"submitter_id": str})
            subs_raw = pd.read_csv(zf.open("submissions.csv"),  dtype={"submitter_id": str})
    except zipfile.BadZipFile:
        st.error("The uploaded file does not appear to be a valid zip archive.")
        st.stop()

    if bio_raw is None or subs_raw is None:
        st.stop()

    # --- Run consolidation --------------------------------------------------
    with st.spinner("Consolidating…"):
        consolidated_bio, consolidated_subs, summary_df = consolidate(bio_raw, subs_raw)

    # --- Summary stats ------------------------------------------------------
    total = len(summary_df)
    multi = int(summary_df["multi_session"].sum())
    single = total - multi

    st.success("Consolidation complete.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total participants", total)
    col2.metric("Single-session", single)
    col3.metric("Multi-session (consolidated)", multi)

    # --- Per-participant table -----------------------------------------------
    st.subheader("Participant summary")

    display_df = summary_df.copy()
    display_df["status"] = display_df["multi_session"].map(
        {True: "consolidated", False: "single session"}
    )
    display_df = display_df.drop(columns=["multi_session"])
    display_df = display_df.rename(
        columns={
            "submitter_id": "Submitter ID",
            "name": "Name",
            "sessions_found": "Sessions found",
            "consolidated_items": "Items in output",
            "status": "Status",
        }
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_order=["Submitter ID", "Name", "Sessions found", "Items in output", "Status"],
    )

    # --- Downloads -----------------------------------------------------------
    st.subheader("Download")

    # CSV zip (always available when CSV data was processed)
    out_csv_bytes = build_output_zip(consolidated_bio, consolidated_subs)
    out_csv_name  = output_zip_name(csv_zip_name)
    st.download_button(
        label=f"Download {out_csv_name}",
        data=out_csv_bytes,
        file_name=out_csv_name,
        mime="application/zip",
    )

    # Audio zip
    if audio_zip_bytes is not None:
        try:
            if not zipfile.is_zipfile(io.BytesIO(audio_zip_bytes)):
                st.error("The audio source does not appear to be a valid zip archive.")
            else:
                with st.spinner("Consolidating audio files…"):
                    out_audio_bytes, missing_audio = build_output_audio_zip(
                        audio_zip_bytes, consolidated_subs
                    )

                if missing_audio:
                    st.warning(
                        f"{len(missing_audio)} audio file(s) listed in submissions.csv were not "
                        f"found in the audio zip and have been omitted from the output:\n"
                        + "\n".join(f"- {f}" for f in sorted(missing_audio))
                    )

                # Name the audio output zip
                if zip_type == ZIP_TYPE_AUDIO_CSV:
                    out_audio_name = audio_output_zip_name(audio_zip_name)
                else:
                    out_audio_name = output_zip_name(audio_zip_name)

                st.download_button(
                    label=f"Download {out_audio_name}",
                    data=out_audio_bytes,
                    file_name=out_audio_name,
                    mime="application/zip",
                )
        except zipfile.BadZipFile:
            st.error("The audio source does not appear to be a valid zip archive.")
