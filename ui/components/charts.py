"""Altair chart builders for the dashboard."""
from __future__ import annotations

import altair as alt
import pandas as pd


def history_forecast_chart(history: list[dict], forecast: list[dict]) -> alt.LayerChart:
    """History line + forecast line with a shaded prediction interval band."""
    hist_df = pd.DataFrame(history)
    if not hist_df.empty:
        hist_df["date"] = pd.to_datetime(hist_df["date"])
        hist_df["series"] = "history"
        hist_df = hist_df.rename(columns={"units": "value"})

    fc_df = pd.DataFrame(forecast)
    fc_df["date"] = pd.to_datetime(fc_df["date"])
    fc_df = fc_df.rename(columns={"predicted_units": "value"})
    fc_df["series"] = "forecast"

    layers = []
    band = (
        alt.Chart(fc_df)
        .mark_area(opacity=0.2, color="#4C78A8")
        .encode(x="date:T", y="lower:Q", y2="upper:Q")
    )
    layers.append(band)
    if not hist_df.empty:
        layers.append(
            alt.Chart(hist_df[["date", "value", "series"]])
            .mark_line(color="#8899A6")
            .encode(x="date:T", y="value:Q")
        )
    layers.append(
        alt.Chart(fc_df).mark_line(color="#4C78A8", strokeDash=[4, 3]).encode(x="date:T", y="value:Q")
    )
    return alt.layer(*layers).properties(height=280).interactive()


def bar(frame: pd.DataFrame, column: str) -> alt.Chart:
    return (
        alt.Chart(frame.reset_index())
        .mark_bar()
        .encode(x=alt.X(f"{frame.index.name}:N", sort="-y"), y=f"{column}:Q")
        .properties(height=220)
    )
