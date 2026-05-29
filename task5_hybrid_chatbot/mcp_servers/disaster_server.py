"""MCP server for natural disaster analytics over CSV files using pandas."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from mcp.server.fastmcp import FastMCP

DEFAULT_DISASTER_DIR = os.getenv(
    "DISASTERS_CSV_DIR",
    "/Users/Uladzimir_Tulinau/Library/CloudStorage/OneDrive-EPAM/SAS AI course/final_task/DISASTERS",
)

mcp = FastMCP("disaster-data-server")


NUMERIC_COLUMNS = [
    "Year",
    "Start Year",
    "End Year",
    "Total Deaths",
    "No Affected",
    "No Injured",
    "No Homeless",
    "Total Affected",
    "Total Damages ('000 US$)",
]

SAFE_TEXT = re.compile(r"^[A-Za-z0-9 .,'()/_-]{1,120}$")


def _validate_text_input(name: str, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) > 120 or not SAFE_TEXT.match(text):
        raise ValueError(f"Invalid value for '{name}'")
    return text


def _validate_year(name: str, value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    year = int(value)
    if year < 1900 or year > 2100:
        raise ValueError(f"Invalid year range for '{name}'")
    return year


def _load_disaster_frames(disaster_dir: str = DEFAULT_DISASTER_DIR) -> pd.DataFrame:
    """Load and combine all disaster CSV files from a directory."""
    directory = Path(disaster_dir)
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Disaster directory not found: {disaster_dir}")

    csv_paths = sorted(directory.glob("*.csv"))
    if not csv_paths:
        raise ValueError(f"No CSV files found in: {disaster_dir}")

    frames = []
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path, low_memory=False)
        frame["_source_file"] = csv_path.name
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)

    for col in NUMERIC_COLUMNS:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    return combined


def _filter_frame(
    frame: pd.DataFrame,
    country: Optional[str] = None,
    disaster_type: Optional[str] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> pd.DataFrame:
    """Apply optional filters used by multiple tools."""
    country = _validate_text_input("country", country)
    disaster_type = _validate_text_input("disaster_type", disaster_type)
    start_year = _validate_year("start_year", start_year)
    end_year = _validate_year("end_year", end_year)

    filtered = frame

    if country and "Country" in filtered.columns:
        filtered = filtered[filtered["Country"].str.contains(country, case=False, na=False)]

    if disaster_type and "Disaster Type" in filtered.columns:
        filtered = filtered[
            filtered["Disaster Type"].str.contains(disaster_type, case=False, na=False)
        ]

    if start_year is not None and "Year" in filtered.columns:
        filtered = filtered[filtered["Year"] >= int(start_year)]

    if end_year is not None and "Year" in filtered.columns:
        filtered = filtered[filtered["Year"] <= int(end_year)]

    return filtered


def _safe_int(value: float | int | None) -> int:
    if value is None or pd.isna(value):
        return 0
    return int(value)


@mcp.tool()
def list_disaster_types(limit: int = 20) -> str:
    """List disaster types and event counts from the dataset."""
    frame = _load_disaster_frames()
    if "Disaster Type" not in frame.columns:
        return "Column 'Disaster Type' is missing in the disaster dataset."

    if limit < 1 or limit > 100:
        return "Parameter 'limit' must be between 1 and 100."

    counts = (
        frame["Disaster Type"]
        .fillna("Unknown")
        .value_counts()
        .head(max(1, min(limit, 100)))
    )

    lines = ["Disaster types by number of events:"]
    for dtype, count in counts.items():
        lines.append(f"- {dtype}: {int(count)}")
    return "\n".join(lines)


@mcp.tool()
def get_disaster_summary(
    country: Optional[str] = None,
    disaster_type: Optional[str] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    limit: int = 10,
) -> str:
    """Return summary rows and aggregate metrics for filtered disasters."""
    if limit < 1 or limit > 50:
        return "Parameter 'limit' must be between 1 and 50."

    frame = _load_disaster_frames()
    filtered = _filter_frame(
        frame,
        country=country,
        disaster_type=disaster_type,
        start_year=start_year,
        end_year=end_year,
    )

    if filtered.empty:
        return "No disaster events found for the specified filters."

    total_events = len(filtered)
    total_deaths = _safe_int(filtered.get("Total Deaths", pd.Series(dtype=float)).sum())
    total_affected = _safe_int(
        filtered.get("Total Affected", pd.Series(dtype=float)).sum()
    )

    lines = [
        "Disaster summary:",
        f"- events: {total_events}",
        f"- total_deaths: {total_deaths}",
        f"- total_affected: {total_affected}",
        "Top matching events:",
    ]

    cols = [
        "Year",
        "Country",
        "Disaster Type",
        "Disaster Subtype",
        "Event Name",
        "Total Deaths",
        "Total Affected",
    ]
    available_cols = [c for c in cols if c in filtered.columns]

    preview = filtered.sort_values(by=["Year"], ascending=False).head(
        max(1, min(limit, 50))
    )

    for _, row in preview.iterrows():
        parts = []
        for col in available_cols:
            value = row[col]
            if pd.isna(value):
                value = "n/a"
            parts.append(f"{col}={value}")
        lines.append("- " + "; ".join(parts))

    return "\n".join(lines)


@mcp.tool()
def top_disasters_by_metric(
    metric: str = "Total Deaths",
    country: Optional[str] = None,
    disaster_type: Optional[str] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    top_n: int = 5,
) -> str:
    """Return top disaster events ranked by a numeric metric."""
    metric = _validate_text_input("metric", metric)
    if metric is None:
        return "Parameter 'metric' is required."

    if top_n < 1 or top_n > 50:
        return "Parameter 'top_n' must be between 1 and 50."

    frame = _load_disaster_frames()
    filtered = _filter_frame(
        frame,
        country=country,
        disaster_type=disaster_type,
        start_year=start_year,
        end_year=end_year,
    )

    if filtered.empty:
        return "No disaster events found for the specified filters."

    if metric not in filtered.columns:
        return f"Metric '{metric}' is not available in the dataset."

    ranked = filtered.dropna(subset=[metric]).sort_values(by=[metric], ascending=False)
    if ranked.empty:
        return f"No rows found with non-null values for metric '{metric}'."

    lines = [f"Top disasters by {metric}:"]
    preview = ranked.head(max(1, min(top_n, 50)))

    for _, row in preview.iterrows():
        year = row.get("Year", "n/a")
        country_name = row.get("Country", "n/a")
        dtype = row.get("Disaster Type", "n/a")
        event = row.get("Event Name", "n/a")
        metric_value = row.get(metric, "n/a")
        if pd.notna(metric_value):
            metric_value = int(metric_value)
        lines.append(
            f"- Year={year}; Country={country_name}; Type={dtype}; "
            f"Event={event}; {metric}={metric_value}"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
