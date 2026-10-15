"""The paper's number formats: times, bytes, ratios, ranges, token counts, parameter counts and macro names.

Every value that the pipeline writes into a table body or a macro goes through one of these functions, so a quantity
has one printed form wherever it appears.  Sizes use decimal units (``MB`` is ``10^6`` bytes), as the paper does;
thousands are separated by ``{,}``.  A missing value prints as ``--`` (:data:`DASH`).
"""
from __future__ import annotations

import math

DASH = "--"
"""A cell or number that the run does not provide."""
DIGITS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven",
          "8": "Eight", "9": "Nine"}


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def to_float(v):
    """A CSV field as a float (None if empty or not a number)."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def thousands(text: str) -> str:
    """``12,345.6`` -> ``12{,}345.6`` (LaTeX: no space after the comma)."""
    return text.replace(",", "{,}")


def num(v, decimals: int = 0) -> str:
    """A number with ``decimals`` decimals and thousands separators: ``2{,}073.9``."""
    if _missing(v):
        return DASH
    return thousands(f"{v:,.{decimals}f}")


def sig(v, digits: int = 3) -> str:
    """``v`` with ``digits`` significant digits, trailing zeros dropped: 0.15, 1.65, 93.2, 18.5, 1{,}722."""
    if _missing(v):
        return DASH
    if v == 0:
        return "0"
    exp = math.floor(math.log10(abs(v)))
    decimals = max(0, digits - 1 - exp)
    text = f"{round(v, decimals):,.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return thousands(text)


def t(sec) -> str:
    """Seconds as the paper's tables print them: 76.1\\,ms, 712\\,ms, 1.8\\,s, 27.4\\,s, 2.9\\,min.  The unit is chosen
    after rounding to the printed precision, so that a value just below a unit's boundary takes the larger unit
    (0.9996 s prints as 1.0\\,s, not 1000\\,ms)."""
    if _missing(sec):
        return DASH
    if sec < 1 and round(sec * 1e3) < 1000:
        return f"{sec * 1e3:.0f}\\,ms" if sec >= 0.0995 else f"{sec * 1e3:.1f}\\,ms"
    if round(sec, 1) < 100:
        return f"{sec:.1f}\\,s"
    return f"{sec / 60:.1f}\\,min"


def secs(sec, digits: int = 3) -> str:
    """Seconds with ``digits`` significant digits and the unit: 0.15\\,s, 18.5\\,s, 93.2\\,s."""
    return DASH if _missing(sec) else f"{sig(sec, digits)}\\,s"


def b(n) -> str:
    """Bytes as the paper's tables print them: 63.3\\,kB, 14.8\\,MB, 325\\,MB, 6.6\\,GB.  The unit is chosen after
    rounding to the printed precision (999.7\\,MB prints as 1.0\\,GB)."""
    if _missing(n):
        return DASH
    units = (("B", 1.0), ("kB", 1e3), ("MB", 1e6), ("GB", 1e9))
    i = max(j for j, (_, k) in enumerate(units) if n >= k or j == 0)
    if i < len(units) - 1 and round(n / units[i][1]) >= 1000:      # 999.7 MB: the next unit
        i += 1
    unit, k = units[i]
    v = n / k
    if unit == "B":
        return f"{v:.0f}\\,B"
    return (f"{v:.1f}" if round(v, 1) < 100 else f"{v:.0f}") + f"\\,{unit}"


def mb(n, decimals: int = 1) -> str:
    """Bytes in MB without the unit: ``2{,}073.9``."""
    return DASH if _missing(n) else num(n / 1e6, decimals)


def gb(n, decimals: int = 1) -> str:
    """Bytes in GB without the unit."""
    return DASH if _missing(n) else num(n / 1e9, decimals)


def x(v, decimals: int = 1) -> str:
    """A ratio: ``4.7$\\times$``."""
    return DASH if _missing(v) else f"{v:.{decimals}f}$\\times$"


def fixed(v, decimals: int = 1) -> str:
    """A plain number with ``decimals`` decimals (no separators)."""
    return DASH if _missing(v) else f"{v:.{decimals}f}"


def pct(v, decimals: int = 0) -> str:
    """A fraction as a percentage number without the sign (the text writes ``\\%``): 0.172 -> ``17``."""
    return DASH if _missing(v) else f"{100 * v:.{decimals}f}"


def signed(v, decimals: int = 1) -> str:
    """A signed number in math mode: ``$-3.1$``, ``$+0.5$``; a nonzero value that rounds to zero keeps one more
    decimal (``$-0.04$``)."""
    if _missing(v):
        return DASH
    d = decimals
    while d < decimals + 2 and v != 0 and round(abs(v), d) == 0:
        d += 1
    return f"${v:+.{d}f}$"


def sci(v, digits: int = 2) -> str:
    """Scientific notation in math mode: ``$4.3\\cdot10^{-4}$`` (0 prints as ``0``)."""
    if _missing(v):
        return DASH
    if v == 0:
        return "0"
    exp = math.floor(math.log10(abs(v)))
    mant = round(v / 10 ** exp, digits - 1)
    if abs(mant) >= 10:
        mant, exp = mant / 10, exp + 1
    return f"${mant:.{digits - 1}f}\\cdot10^{{{exp}}}$"


def about(v, digits: int = 2) -> str:
    """``v`` rounded to ``digits`` significant figures, for "about N" in the text: 31{,}910 -> ``32{,}000``."""
    if _missing(v):
        return DASH
    if v == 0:
        return "0"
    exp = math.floor(math.log10(abs(v)))
    r = round(v, digits - 1 - exp)
    return num(r, max(0, digits - 1 - exp))


def tokens(n) -> str:
    """A token count: ``2{,}048``."""
    return DASH if _missing(n) else thousands(f"{int(n):,}")


def params(n) -> str:
    """A parameter count: 124M, 4.0B, 13.0B, 61.7K."""
    if _missing(n):
        return DASH
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if n >= 1e8:
        return f"{n / 1e6:.0f}M"
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}K"


def rng(lo, hi, fmt=sig) -> str:
    """``lo--hi`` in the format ``fmt`` (one value when both print the same)."""
    if _missing(lo) or _missing(hi):
        return DASH
    a, c = fmt(lo), fmt(hi)
    return a if a == c else f"{a}--{c}"


def factor(theirs, ours):
    """``(text, theirs / ours)`` as the comparison tables print the factor between a published cost and ours:
    ``4{,}422$\\times\\downarrow$`` when ours is lower, ``6.4$\\times\\uparrow$`` when higher, ``same`` within 1%;
    ``(DASH, None)`` when either is missing."""
    if _missing(theirs) or _missing(ours) or not theirs or not ours:
        return DASH, None
    q = theirs / ours
    if 0.99 < q < 1.01:
        return "same", q
    v = q if q >= 1 else 1 / q
    s = thousands(f"{v:,.0f}") if v >= 100 else (f"{v:.0f}" if v >= 10 else f"{v:.1f}")
    if v < 1.05:                                # '1.0' would read as a tie
        s = f"{v:.2f}"
    return s + (r"$\times\downarrow$" if q >= 1 else r"$\times\uparrow$"), q


def macro_name(*parts) -> str:
    """A LaTeX macro name from words and numbers: each part's alphanumeric words with their first letter in upper
    case (the rest kept), digits spelled out (macro names have letters only).  ``macro_name("Crossover", "gpt2",
    512, "Kpre")`` -> ``CrossoverGptTwoFiveOneTwoKpre``."""
    words = []
    for p in parts:
        for w in "".join(c if c.isalnum() else " " for c in str(p)).split():
            words.append(w[0].upper() + w[1:])
    return "".join(DIGITS.get(c, c) for c in "".join(words))
