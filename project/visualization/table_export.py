"""Table writer that emits a CSV plus a LaTeX booktabs sibling.

Public API:
    save_table(df, csv_path, *, caption=None, label=None,
               highlight=None, index=False, formatters=None)

The CSV is written exactly as before (no column-name or schema changes).
Alongside it, a ``.tex`` file is written that can be dropped into Overleaf
via ``\\input{...}``. Tables use ``\\toprule`` / ``\\midrule`` /
``\\bottomrule`` from the ``booktabs`` LaTeX package.

Formatting rules:
    * Floats are rendered to 3 significant figures.
    * |x| < 1e-3 (and != 0) switches to scientific notation.
    * Booleans render as Yes / No (no LaTeX ``True``/``False`` clutter).
    * Integers and strings pass through unchanged.
    * The ``highlight`` argument may bold the row whose value in a given
      column is max/min (commonly ``("Votes", "max")`` or ``("mean", "max")``).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Number formatting                                                            #
# --------------------------------------------------------------------------- #

def _format_float(x: float, sig: int = 3) -> str:
    if pd.isna(x):
        return "--"
    if x == 0:
        return "0"
    ax = abs(x)
    if ax < 1e-3:
        return f"{x:.{sig - 1}e}"
    if ax >= 1e4:
        return f"{x:.{sig - 1}e}"
    # round to `sig` significant figures
    exp = math.floor(math.log10(ax))
    digits = sig - 1 - exp
    if digits < 0:
        digits = 0
    return f"{x:.{digits}f}"


def _format_cell(value: Any, sig: int) -> str:
    if value is None:
        return "--"
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if bool(value) else "No"
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    if isinstance(value, (float, np.floating)):
        if pd.isna(value):
            return "--"
        return _format_float(float(value), sig=sig)
    if isinstance(value, (np.ndarray, list, tuple)):
        return ", ".join(_format_cell(v, sig) for v in value)
    s = str(value)
    return _escape_latex(s)


def _escape_latex(s: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&":  r"\&",
        "%":  r"\%",
        "$":  r"\$",
        "#":  r"\#",
        "_":  r"\_",
        "{":  r"\{",
        "}":  r"\}",
        "~":  r"\textasciitilde{}",
        "^":  r"\textasciicircum{}",
    }
    out = []
    for ch in s:
        out.append(repl.get(ch, ch))
    return "".join(out)


# --------------------------------------------------------------------------- #
# DataFrame -> LaTeX                                                           #
# --------------------------------------------------------------------------- #

def _column_alignment(df: pd.DataFrame) -> str:
    aligns = []
    for col in df.columns:
        dt = df[col].dtype
        if pd.api.types.is_numeric_dtype(dt) and not pd.api.types.is_bool_dtype(dt):
            aligns.append("r")
        else:
            aligns.append("l")
    return "".join(aligns)


def _pretty_header(name: str) -> str:
    return _escape_latex(str(name).replace("_", " "))


def _resolve_highlight(
    df: pd.DataFrame,
    highlight: tuple[str, str] | Mapping[str, str] | None,
) -> set[int]:
    """Return the set of row positions to render in bold."""
    if highlight is None or df.empty:
        return set()

    rules: dict[str, str]
    if isinstance(highlight, tuple) and len(highlight) == 2:
        rules = {highlight[0]: highlight[1]}
    elif isinstance(highlight, Mapping):
        rules = dict(highlight)
    else:
        return set()

    bold_rows: set[int] = set()
    for col, mode in rules.items():
        if col not in df.columns:
            continue
        s = df[col]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        if s.dropna().empty:
            continue
        if mode == "max":
            idx = s.idxmax()
        elif mode == "min":
            idx = s.idxmin()
        else:
            continue
        # Map a (possibly non-default) DataFrame index back to row position.
        try:
            pos = df.index.get_loc(idx)
        except KeyError:
            continue
        if isinstance(pos, slice):
            bold_rows.update(range(pos.start or 0, pos.stop or 0))
        elif isinstance(pos, np.ndarray):
            bold_rows.update(int(i) for i in np.where(pos)[0])
        else:
            bold_rows.add(int(pos))
    return bold_rows


def dataframe_to_latex(
    df: pd.DataFrame,
    *,
    caption: str | None = None,
    label: str | None = None,
    highlight: tuple[str, str] | Mapping[str, str] | None = None,
    sig: int = 3,
    index: bool = False,
) -> str:
    if df is None or df.empty:
        body = "% (empty table)"
        align = "l"
    else:
        work = df.reset_index() if index else df.reset_index(drop=True)
        align = _column_alignment(work)
        bold_rows = _resolve_highlight(work, highlight)

        header = " & ".join(_pretty_header(c) for c in work.columns) + r" \\"

        rows: list[str] = []
        for i, (_, row) in enumerate(work.iterrows()):
            cells = [_format_cell(row[c], sig=sig) for c in work.columns]
            if i in bold_rows:
                cells = [f"\\textbf{{{c}}}" for c in cells]
            rows.append(" & ".join(cells) + r" \\")
        body = header + "\n\\midrule\n" + "\n".join(rows)

    lines = [r"\begin{table}[ht]", r"\centering"]
    if caption:
        lines.append(r"\caption{" + _escape_latex(caption) + "}")
    if label:
        lines.append(r"\label{" + label + "}")
    lines.append(r"\begin{tabular}{" + align + "}")
    lines.append(r"\toprule")
    lines.append(body)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Public entry point                                                           #
# --------------------------------------------------------------------------- #

def save_table(
    df: pd.DataFrame,
    csv_path: str | Path,
    *,
    caption: str | None = None,
    label: str | None = None,
    highlight: tuple[str, str] | Mapping[str, str] | None = None,
    index: bool = False,
    sig: int = 3,
) -> tuple[Path, Path]:
    """Write the CSV (unchanged schema) plus a booktabs .tex sibling.

    Returns ``(csv_path, tex_path)``.

    The ``label`` argument is filled in from the CSV stem if omitted,
    yielding ``\\label{tab:<stem>}``.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=index)

    if label is None:
        label = f"tab:{csv_path.stem}"
    if caption is None:
        caption = csv_path.stem.replace("_", " ")

    tex = dataframe_to_latex(
        df,
        caption=caption,
        label=label,
        highlight=highlight,
        sig=sig,
        index=index,
    )
    tex_path = csv_path.with_suffix(".tex")
    tex_path.write_text(tex)
    return csv_path, tex_path
