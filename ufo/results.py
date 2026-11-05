""" Summarize UFO experimental results """
import csv
from collections import defaultdict
from pathlib import Path

GROUP_FIELDS = (
    "protocol",
    "family",
    "modality",
    "treatment",
    "weight_decay",
    "max_steps",
    "lambda",
    "schedule_start",
    "schedule_floor",
    "schedule_updates"
)

def load_final_rows(directory: str | Path) -> list[dict[str, str]]:
    final_rows = []
    for path in sorted(Path(directory).glob("*.csv")):
        if path.name == "summary.csv":
            continue
        with path.open(newline="") as file:
            rows = list(csv.DictReader(file))
        if rows:
            final_rows.append(rows[-1])
    return final_rows

def kaplan_meier_median(observations: list[tuple[int, bool]]) -> int | None:
    survival = 1.0
    event_times = sorted({time for time, event in observations if event})
    for time in event_times:
        at_risk = sum(observed_time >= time for observed_time, _ in observations)
        events = sum(observed_time == time and event for observed_time, event in observations)
        survival *= 1.0 - events / at_risk
        if survival <= 0.5:
            return time
    return None

def summarize_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in GROUP_FIELDS)
        grouped[key].append(row)
    summaries = []
    for key, group in sorted(grouped.items()):
        observations = []
        for row in group:
            event = bool(int(row["event"]))
            time = int(float(row["grok_step"] if event else row["stop_step"]))
            observations.append((time, event))
        grokked = sum(event for _, event in observations)
        summary = dict(zip(GROUP_FIELDS, key))
        summary.update(
            {
                "runs": len(group),
                "grokked": grokked,
                "censored": len(group) - grokked,
                "grokking_rate": grokked / len(group),
                "km_median": kaplan_meier_median(observations),
            }
        )
        summaries.append(summary)
    return summaries

def write_summary(directory: str | Path = "results") -> Path:
    directory = Path(directory)
    summaries = summarize_rows(load_final_rows(directory))
    output = directory / "summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[*GROUP_FIELDS, "runs", "grokked", "censored", "grokking_rate", "km_median"])
        writer.writeheader()
        writer.writerows(summaries)
    return output
