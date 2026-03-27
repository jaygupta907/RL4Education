"""
PDF Equation Extractor
======================
Extracts math equations from PDF files and resolves variable symbols
to their English names, producing a structured JSON output.

    force = mass * acceleration
    Youngs_modulus = stress / strain
    displacement = initial_velocity*time + 0.5*acceleration*time^2

Works on both digital (LaTeX-generated) and scanned PDFs.
Handles multi-column layouts, typeset fractions (sigma over epsilon),
and cross-page variable definitions.

Usage:
    python eq_extractor.py input.pdf [output.json]

Requirements:
    pip install pymupdf pytesseract pillow
    # System: tesseract-ocr, poppler-utils
    # Ubuntu:  sudo apt install tesseract-ocr poppler-utils
    # Windows: https://github.com/UB-Mannheim/tesseract/wiki
    #          https://github.com/oschwartz10612/poppler-windows/releases

Output JSON schema:
    {
      "source_pdf": "...",
      "total_pages": N,
      "total_equations": N,
      "variable_registry": { "F": [{"symbol": "F", "name": "force", ...}] },
      "equations": [
        {
          "eq_id": "eq_0001_p3",
          "page": 3,
          "raw": "F = m * a",
          "canonical": "force = mass * acceleration",
          "variables": {"F": {"name":"force",...}, ...},
          "method": "text"   // or "ocr"
        }
      ]
    }
"""
# RUN USING python filename.py <pdf_file> <results_json>
import re
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import defaultdict

import fitz            # pymupdf
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from PIL import Image
import io

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Unicode → ASCII normalization
# ─────────────────────────────────────────────────────────────────────────────
GREEK = {
    'α':'alpha','β':'beta','γ':'gamma','δ':'delta','ε':'epsilon',
    'ζ':'zeta','η':'eta','θ':'theta','ι':'iota','κ':'kappa',
    'λ':'lambda','μ':'mu','ν':'nu','ξ':'xi','π':'pi','ρ':'rho',
    'σ':'sigma','τ':'tau','υ':'upsilon','φ':'phi','χ':'chi',
    'ψ':'psi','ω':'omega','Δ':'Delta','∆':'Delta','Σ':'Sigma',
    'Ω':'Omega','Γ':'Gamma','Λ':'Lambda','Π':'Pi','Θ':'Theta',
    '∝':'proportional_to','≈':'approx','≠':'neq',
    '≤':'leq','≥':'geq','×':'*','÷':'/','·':'*','∞':'inf',
    '∂':'d','∇':'nabla','√':'sqrt','∫':'integral','±':'+-',
}

# Font names that indicate math/symbol content
MATH_FONT_KEYWORDS = {
    'symbol','cmsy','cmmi','cmex','msbm','stix','lmmath',
    'mtextra','mtmi','mtms','mtsy','euclid','mathtime',
}

def is_math_font(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in MATH_FONT_KEYWORDS)

def norm(text: str) -> str:
    """Replace Unicode math chars with ASCII equivalents."""
    for ch, rep in GREEK.items():
        text = text.replace(ch, rep)
    return text.strip()

EQ_NUM_RE = re.compile(r'^\(\d+(?:\.\d+)?\)$')


# ─────────────────────────────────────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Span:
    """A single text span with full metadata from the PDF."""
    text: str; norm_text: str
    x0: float; y0: float; x1: float; y1: float
    font: str; size: float; flags: int; page: int
    math_font: bool; is_eq_num: bool

    @property
    def cx(self): return (self.x0 + self.x1) / 2
    @property
    def w(self): return self.x1 - self.x0

    def x_overlaps(self, other, tol=0.25):
        ov = min(self.x1, other.x1) - max(self.x0, other.x0)
        shorter = min(self.w, other.w)
        return shorter > 0 and ov / shorter >= tol


@dataclass
class Row:
    """A horizontal row of spans, grouped by y-position."""
    spans: list

    @property
    def text(self): return ' '.join(s.norm_text for s in self.spans)
    @property
    def y0(self): return self.spans[0].y0 if self.spans else 0
    @property
    def x0(self): return min(s.x0 for s in self.spans) if self.spans else 0
    @property
    def x1(self): return max(s.x1 for s in self.spans) if self.spans else 0

    def x_overlaps_row(self, other, tol=0.2):
        ov = min(self.x1, other.x1) - max(self.x0, other.x0)
        shorter = min(self.x1 - self.x0, other.x1 - other.x0)
        return shorter > 0 and ov / shorter >= tol


@dataclass
class MathBlock:
    """A candidate equation before canonicalization."""
    raw: str; page: int; y0: float
    method: str = 'text'
    chapter: int = 0


@dataclass
class VarDef:
    """A resolved variable definition: symbol → English name."""
    symbol: str; name: str
    unit: Optional[str] = None
    page: int = -1
    confidence: float = 1.0
    chapter: int = 0   # chapter index for cross-topic scoping


@dataclass
class Equation:
    """Final output equation with canonical English form."""
    eq_id: str; page: int; raw: str; canonical: str
    variables: dict = field(default_factory=dict)
    method: str = 'text'


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 1 — Span extraction
# ─────────────────────────────────────────────────────────────────────────────
def get_spans(page: fitz.Page, page_num: int) -> list:
    spans = []
    for block in page.get_text('dict')['blocks']:
        if block.get('type') != 0:
            continue
        for line in block['lines']:
            for s in line['spans']:
                t = s['text'].strip()
                if not t:
                    continue
                n = norm(t)
                if not n:
                    continue
                b = s['bbox']
                spans.append(Span(
                    text=t, norm_text=n,
                    x0=b[0], y0=b[1], x1=b[2], y1=b[3],
                    font=s.get('font', ''), size=s.get('size', 10),
                    flags=s.get('flags', 0), page=page_num,
                    math_font=is_math_font(s.get('font', '')),
                    is_eq_num=bool(EQ_NUM_RE.match(t.strip())),
                ))
    return sorted(spans, key=lambda s: (s.y0, s.x0))


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 2 — Column detection
# ─────────────────────────────────────────────────────────────────────────────
def detect_columns(spans: list, page_width: float) -> list:
    """
    Find major text columns using span-density peaks.
    Uses 40px buckets; columns must have >=3 spans to be detected.
    Merges nearby peaks (within 120px) to avoid fragmenting on
    isolated fraction characters (Delta, =, L etc.).
    Returns list of (x0, x1) tuples.
    """
    if not spans:
        return [(0, page_width)]
    B = 40
    hist = defaultdict(int)
    for s in spans:
        hist[int(s.x0 / B) * B] += 1
    peaks = sorted(x for x, c in hist.items() if c >= 3)
    if not peaks:
        return [(0, page_width)]
    # Merge peaks within 120px
    merged = [peaks[0]]
    for x in peaks[1:]:
        if x - merged[-1] > 120:
            merged.append(x)
    if len(merged) < 2:
        return [(0, page_width)]
    cols = []
    for i, cs in enumerate(merged):
        cx0 = max(0, cs - 10)
        cx1 = merged[i + 1] - 15 if i + 1 < len(merged) else page_width
        cols.append((cx0, cx1))
    return cols


def col_spans(spans: list, col: tuple) -> list:
    return [s for s in spans if s.x0 >= col[0] - 5 and s.x1 <= col[1] + 10]


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 3 — Row grouping
# ─────────────────────────────────────────────────────────────────────────────
def group_rows(spans: list, y_tol: float = 3.0) -> list:
    """
    Group spans into horizontal rows.
    y_tol=3 is intentionally tight so fraction numerators/denominators
    (which sit 10-15px above/below the equation line) stay in separate rows.
    """
    if not spans:
        return []
    srt = sorted(spans, key=lambda s: (s.y0, s.x0))
    rows = [Row([srt[0]])]
    ref = srt[0].y0
    for s in srt[1:]:
        if abs(s.y0 - ref) <= y_tol:
            rows[-1].spans.append(s)
        else:
            rows.append(Row([s]))
            ref = s.y0
    for r in rows:
        r.spans.sort(key=lambda s: s.x0)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 4 — MathBlock reconstruction
# ─────────────────────────────────────────────────────────────────────────────
def _sat_qualifies(other: Row, row_rhs_start: float, row_x1: float) -> bool:
    """
    Check if 'other' row qualifies as a fraction satellite of the equation row.
    Satellites must be:
      - In the RHS x-zone of the equation
      - Math-like (not a full prose expression)
      - Not a prose connector word
      - Not punctuated (no commas)
    """
    ot = other.text.strip()
    if not ot:
        return False
    words = ot.split()

    # Must be in the RHS x-zone
    rhs_zone = (other.x0 >= row_rhs_start - 15 and other.x0 <= row_x1 + 60)
    if not rhs_zone:
        return False

    # Reject if contains commas (prose phrase)
    if ',' in ot:
        return False

    # Reject prose connectors and transition words
    PROSE_CONNECTORS = {
        'thus','hence','therefore','however','moreover','furthermore',
        'also','note','since','because','where','when','if','as',
        'so','but','and','or','the','a','an','this','that',
    }
    first_tok = words[0].lower().rstrip('.,;:')
    if first_tok in PROSE_CONNECTORS:
        return False

    # Reject noun phrases with apostrophes (e.g. "Young's modulus")
    if "'" in ot or '\u2019' in ot:
        return False

    has_math = any(s.math_font for s in other.spans)
    has_math_name = any(w in ot for w in [
        'sigma','epsilon','theta','omega','alpha','beta','lambda',
        'mu','phi','tau','pi','rho','Delta','Sigma','Gamma',
        'sin','cos','tan','sqrt','log','exp',
    ])
    is_single_token = len(words) == 1
    is_short_expr = len(words) <= 3 and any(c in ot for c in '/*+-^()')

    # Multi-word non-math phrase → not a fraction component
    if len(words) > 1 and not has_math and not has_math_name:
        return False

    # Full expression (e.g. "stress proportional_to strain") → not a component
    if len(words) >= 3:
        alpha_toks = [w for w in words if w.isalpha() and len(w) > 1]
        op_toks = [w for w in words if w in (
            'proportional_to','approx','neq','leq','geq'
        ) or (not w.isalnum() and len(w) == 1)]
        if len(alpha_toks) >= 2 and op_toks:
            return False

    return has_math or has_math_name or is_single_token or is_short_expr


def reconstruct_blocks(rows: list, body_size: float,
                       frac_band: float = 32.0) -> list:
    """
    Core reconstruction logic.

    For every row containing '=':
      1. Find the = sign position (rhs_start)
      2. Scan ±frac_band vertically for satellite rows in the RHS zone
      3. For each RHS span, if a satellite sits directly above/below at
         the same x-position → they form a vertical fraction (numer/denom)
      4. If the row ends with '=' (empty RHS), look for numer/denom
         satellites paired with the = sign itself
      5. Build reconstructed equation string and return as MathBlocks

    Also handles isolated fraction bars: lone '=' spans at math font size
    with a numerator above and denominator below.
    """
    used = set()
    blocks = []

    # Pass A: Isolated fraction bar rows
    for ri, row in enumerate(rows):
        t = row.text.strip()
        if t not in ('=', '–', '—') or len(row.spans) > 2:
            continue
        if not any(s.size >= body_size * 1.05 or s.math_font for s in row.spans):
            continue
        above = [(rj, r) for rj, r in enumerate(rows)
                 if rj != ri and rj not in used
                 and 0 < row.y0 - r.y0 <= frac_band
                 and row.x_overlaps_row(r)]
        below = [(rj, r) for rj, r in enumerate(rows)
                 if rj != ri and rj not in used
                 and 0 < r.y0 - row.y0 <= frac_band
                 and row.x_overlaps_row(r)]
        if above and below:
            above.sort(key=lambda x: row.y0 - x[1].y0)
            below.sort(key=lambda x: x[1].y0 - row.y0)
            nj, nr = above[0]
            dj, dr = below[0]
            nt = nr.text.strip()
            dt = dr.text.strip()
            if len(nt.split()) <= 5 and len(dt.split()) <= 5:
                used.update([ri, nj, dj])
                blocks.append(MathBlock(
                    raw=f'({nt}) / ({dt})',
                    page=row.spans[0].page, y0=row.y0))

    # Pass B: Equation rows
    for ri, row in enumerate(rows):
        if ri in used:
            continue
        if '=' not in row.text:
            continue

        # Find the = sign position for RHS x-zone
        eq_span = next((s for s in row.spans if '=' in s.norm_text), None)
        row_rhs_start = eq_span.x0 if eq_span else row.x0

        # Gather satellite rows (above and below, within frac_band)
        sats_above, sats_below = [], []
        for rj, other in enumerate(rows):
            if rj == ri or rj in used:
                continue
            if '=' in other.text:
                continue  # don't absorb other equation rows
            dy_up = row.y0 - other.y0
            dy_dn = other.y0 - row.y0
            qualifies = _sat_qualifies(other, row_rhs_start, row.x1)
            if 0 < dy_up <= frac_band and qualifies:
                sats_above.append((rj, other, dy_up))
            if 0 < dy_dn <= frac_band and qualifies:
                sats_below.append((rj, other, dy_dn))

        sats_above.sort(key=lambda x: x[2])
        sats_below.sort(key=lambda x: x[2])

        # Build per-span fraction substitutions
        frac_subs = {}       # span index → "numer/denom"
        consumed_sats = set()

        for si, span in enumerate(row.spans):
            if span.is_eq_num:
                continue
            best_above, best_below = None, None

            for rj, sat_row, dy in sats_above:
                if rj in consumed_sats:
                    continue
                for ss in sat_row.spans:
                    if span.x_overlaps(ss):
                        if best_above is None or dy < best_above[2]:
                            best_above = (rj, ss, dy)

            for rj, sat_row, dy in sats_below:
                if rj in consumed_sats:
                    continue
                for ss in sat_row.spans:
                    if span.x_overlaps(ss):
                        if best_below is None or dy < best_below[2]:
                            best_below = (rj, ss, dy)

            if best_above and best_below:
                numer = best_above[1].norm_text.strip()
                denom = best_below[1].norm_text.strip()
                if len(numer.split()) <= 4 and len(denom.split()) <= 4:
                    frac_subs[si] = f'{numer}/{denom}'
                    consumed_sats.update([best_above[0], best_below[0]])
            elif best_above and not best_below:
                # span is denominator, above is numerator
                numer = best_above[1].norm_text.strip()
                denom = span.norm_text.strip()
                if len(numer.split()) <= 4:
                    frac_subs[si] = f'{numer}/{denom}'
                    consumed_sats.add(best_above[0])
            elif best_below and not best_above:
                # span is numerator, below is denominator
                numer = span.norm_text.strip()
                denom = best_below[1].norm_text.strip()
                if len(denom.split()) <= 4:
                    frac_subs[si] = f'{numer}/{denom}'
                    consumed_sats.add(best_below[0])

        # Reconstruct equation text
        parts = []
        for si, span in enumerate(row.spans):
            if span.is_eq_num:
                continue
            parts.append(frac_subs[si] if si in frac_subs else span.norm_text)

        eq_text = ' '.join(parts).strip()

        # Handle empty RHS (row ends with '='): look for numer/denom from sats
        if eq_text.rstrip().endswith('=') and (sats_above or sats_below):
            rhs_numer, rhs_denom = None, None
            for rj, sat_row, dy in sats_above:
                for ss in sat_row.spans:
                    if eq_span and (eq_span.x_overlaps(ss) or ss.x0 > eq_span.x0):
                        rhs_numer = (rj, sat_row.text.strip())
                        break
                if rhs_numer:
                    break
            if not rhs_numer and sats_above:
                rj, sr, _ = sats_above[0]
                rhs_numer = (rj, sr.text.strip())
            for rj, sat_row, dy in sats_below:
                for ss in sat_row.spans:
                    if eq_span and (eq_span.x_overlaps(ss) or ss.x0 > eq_span.x0):
                        rhs_denom = (rj, sat_row.text.strip())
                        break
                if rhs_denom:
                    break
            if not rhs_denom and sats_below:
                rj, sr, _ = sats_below[0]
                rhs_denom = (rj, sr.text.strip())

            if rhs_numer and rhs_denom:
                nj, nt = rhs_numer
                dj, dt = rhs_denom
                if len(nt.split()) <= 4 and len(dt.split()) <= 4:
                    eq_text += f' {nt}/{dt}'
                    consumed_sats.update([nj, dj])
            elif rhs_numer:
                nj, nt = rhs_numer
                eq_text += f' {nt}'
                consumed_sats.add(nj)

        # Clean up equation number labels like (8.1)
        eq_text = re.sub(r'\(\d+(?:\.\d+)?\)\s*$', '', eq_text).strip()
        eq_text = re.sub(r'\s+', ' ', eq_text).strip()

        if eq_text and '=' in eq_text:
            used.add(ri)
            used.update(consumed_sats)
            blocks.append(MathBlock(
                raw=eq_text, page=row.spans[0].page, y0=row.y0))

    return blocks


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 5 — Equation validation (post-reconstruction gate)
# ─────────────────────────────────────────────────────────────────────────────
PROSE_LHS_WORDS = {
    'the','a','an','in','at','if','this','that','when','where',
    'since','thus','hence','note','let','we','for','is','are',
    'as','so','but','and','or','not','it','its','they',
    'magnitude','restoring','applied','internal','total','change',
    'ratio','value','amount','number','rate','force','area','body',
    'line','curve','graph','region','point','figure','table',
    'section','chapter','equation','above','below','following',
}

def is_valid_equation(text: str) -> bool:
    """
    Return True only if text looks like a real mathematical equation.
    Applied AFTER reconstruction to avoid killing valid equations early.
    """
    text = text.strip()
    if not text or '=' not in text or len(text) < 3:
        return False

    lhs, _, rhs = text.partition('=')
    lhs = lhs.strip()
    rhs = rhs.strip()

    # Strip inline label prefixes like "Newton's Law: F = m*a"
    if ':' in lhs:
        candidate = lhs.rsplit(':', 1)[-1].strip()
        if candidate and len(candidate.split()) <= 2 and re.match(r'^[A-Za-z_]', candidate):
            lhs = candidate

    if not lhs or not rhs:
        return False

    # LHS: max 4 tokens, first must not be a prose word
    lhs_tokens = lhs.split()
    if len(lhs_tokens) > 4:
        return False
    first = lhs_tokens[0].lower().rstrip('.,;:()')
    if first in PROSE_LHS_WORDS:
        return False

    # LHS must not contain prose punctuation
    if any(c in lhs for c in ',;()'):
        return False

    # LHS with '/' — both sides must be distinct short symbols
    if '/' in lhs:
        parts = lhs.split('/')
        if len(parts) == 2 and parts[0].strip().lower() == parts[1].strip().lower():
            return False
        if any(len(p.strip().split()) > 2 for p in parts):
            return False

    # RHS must have math structure OR be short
    rhs_words = rhs.split()
    has_operator = any(c in rhs for c in '*/+-^()')
    has_math_word = any(w in rhs for w in [
        'sqrt','Delta','sigma','epsilon','theta','omega','alpha',
        'beta','lambda','mu','phi','tau','pi','rho','sin','cos',
        'tan','log','exp','ln','proportional_to',
    ])
    is_number = bool(re.match(r'^[\d.\-+eE]+$', rhs.strip()))
    is_short = len(rhs_words) <= 4

    if not (has_operator or has_math_word or is_number or is_short):
        return False

    # Reject prose RHS: many words, no operators
    if len(rhs_words) > 6 and not has_operator and not has_math_word:
        return False

    STOPS = {'the','a','an','is','are','be','to','of','in','at','by',
             'for','and','or','not','with','from','which','that','this',
             'known','called','defined','given','as'}
    if len(rhs_words) > 3:
        stop_ratio = sum(1 for w in rhs_words
                        if w.lower().rstrip('.,;:') in STOPS) / len(rhs_words)
        if stop_ratio > 0.5:
            return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 6 — Variable definition extraction
# ─────────────────────────────────────────────────────────────────────────────
NAME_STOPS = {
    'and','or','of','in','at','is','are','be','to','for','with',
    'which','that','this','by','from','when','where','if','then',
    'between','about','through','as','on','under',
}

def clean_name(raw: str) -> str:
    """Extract a clean snake_case name from a captured phrase."""
    raw = raw.strip(" .,;:()'\"")
    words = raw.lower().split()
    out = []
    for w in words:
        if w in NAME_STOPS:
            break
        if len(w) == 1 and not w.isalpha():
            break
        out.append(w)
    out = out[:4]
    if not out:
        return ''
    # Reject single stop-words captured as names
    if len(out) == 1 and out[0] in NAME_STOPS:
        return ''
    return '_'.join(out)


SYM_BLACKLIST = {
    # Prefix fragments
    'non','pre','sub','re','un','co','de','di','pro','per',
    'bi','tri','ex','al','el','le','mo','am','an','on',
    # Common English words
    'is','be','it','at','by','to','if','we','so','no','up',
    'let','see','use','add','get','put','run','new','old',
    'note','case','part','type','show','give','take','make',
    'load','area','same','this','that','they','true','only',
    'both','each','some','more','less','very','most','many',
    'one','two','all','any','its','our','has','had','was','can',
    'why','how','who','the',
    # Physics text words that look like symbols
    'time','path','work','heat','body','tube','pipe','book',
    'wave','beam','coil','disc','band','mean','form','semi',
    'cone','half','full','zero','gain','loss','law','set','ask',
    # Common abbreviations that appear in academic text
    'ncf','irs','sg','retd','pgt','npe','ncert','ncfse',
}

def valid_symbol(sym: str) -> bool:
    """Return True if sym could be a real physics/math variable symbol."""
    if not sym or len(sym) > 4 or ' ' in sym:
        return False
    if not sym[0].isalpha():
        return False
    if sym.lower() in SYM_BLACKLIST:
        return False
    return True


# Chapter/section heading detector — resets symbol scope across topics
CHAPTER_HEAD_RE = re.compile(
    r"^(?:chapter|section|part)\s*\d+|^[A-Z][A-Z\s]{3,40}$",
    re.IGNORECASE
)

DEF_PATTERNS = [
    # 1. "noun-phrase (symbol)" — most reliable
    (re.compile(
        r'([a-zA-Z][a-zA-Z\s]{1,25}?)\s*\(\s*([A-Za-z_][A-Za-z_0-9]{0,3})\s*\)',
        re.IGNORECASE), 'noun_sym'),
    # 2. "symbol (noun-phrase)"
    (re.compile(
        r'\b([A-Za-z_][A-Za-z_0-9]{0,3})\s*\(\s*([a-zA-Z][a-zA-Z\s\']{2,30})\s*\)',
        re.IGNORECASE), 'sym_noun'),
    # 3. "where X is/denotes [the] name ..."
    (re.compile(
        r'\bwhere\s+([A-Za-z_][A-Za-z_0-9]{0,3})\s+'
        r'(?:is|are|denotes?|represents?)\s+(?:the\s+|an?\s+)?'
        r'([a-z][a-z\-]{1,20}(?:\s+[a-z][a-z\-]{1,20}){0,3})'
        r'(?:\s+(?:and|or|of|in|at|which|that|,|;|\.).*)?$',
        re.IGNORECASE | re.MULTILINE), 'where'),
    # 4. "let X be/denote [the] name ..."
    (re.compile(
        r'\blet\s+([A-Za-z_][A-Za-z_0-9]{0,3})\s+'
        r'(?:be|denote|represent)\s+(?:the\s+|an?\s+)?'
        r'([a-z][a-z\-]{1,20}(?:\s+[a-z][a-z\-]{1,20}){0,3})'
        r'(?:\s+(?:and|or|of|in|at|which|that|,|;|\.).*)?$',
        re.IGNORECASE | re.MULTILINE), 'let'),
    # 5. Notation table: "X — name" or "X: name" (whole line)
    (re.compile(
        r'^\s*([A-Za-z_][A-Za-z_0-9]{0,3})\s*(?:—|–|:)\s*'
        r'([a-z][a-z\s\-]{1,25})\s*$', re.IGNORECASE), 'table'),
]


def extract_defs(text: str, page_num: int) -> list:
    """Extract variable definitions from a block of text."""
    defs = []
    seen = set()
    chapter = 0

    # Join lowercase continuation lines to fix truncated where-clauses
    # e.g. "where g is the\ngravitational acceleration" → joined
    text = re.sub(r'(?<!\n)\n(?=[a-z,])', ' ', text)

    for line in text.replace('.', '.\n').split('\n'):
        line = line.strip()
        if not line or len(line) < 5:
            continue

        # Chapter/section boundary: reset symbol scope and advance chapter index
        if CHAPTER_HEAD_RE.match(line.strip()):
            seen = set()
            chapter += 1
            continue

        for pat, kind in DEF_PATTERNS:
            for m in pat.finditer(line):
                g1, g2 = m.group(1).strip(), m.group(2).strip()

                # noun_sym: g1=noun, g2=symbol
                if kind == 'noun_sym':
                    sym, name_raw = g2, g1
                else:
                    sym, name_raw = g1, g2

                name = clean_name(name_raw)
                if not name or not valid_symbol(sym):
                    continue
                if sym in seen:
                    continue

                # Reject single lowercase figure labels (a, b, c, d, e)
                if kind in ('noun_sym', 'table') and len(sym) == 1 \
                        and sym.islower() and sym in 'abcde':
                    continue

                # Single uppercase from noun_sym — only accept known physics symbols
                if kind == 'noun_sym' and len(sym) == 1 and sym.isupper():
                    if sym not in 'FmavpqVCIRLBEkKYTstnNUHG':
                        continue

                conf = 1.0 if kind in ('noun_sym', 'where', 'let', 'table') else 0.7
                defs.append(VarDef(
                    symbol=sym, name=name,
                    page=page_num, confidence=conf, chapter=chapter,
                ))
                seen.add(sym)

    return defs


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 7 — Cross-page variable registry
# ─────────────────────────────────────────────────────────────────────────────
def build_registry(all_defs: list) -> dict:
    reg = defaultdict(list)
    for vd in all_defs:
        existing = reg[vd.symbol]
        if not any(e.name == vd.name and e.page == vd.page for e in existing):
            existing.append(vd)
    return dict(reg)


def resolve(sym: str, eq_page: int, registry: dict,
            window: int = 5, eq_chapter: int = 0):
    """
    Find best definition for symbol given the equation's page and chapter.
    Prefers same-chapter definitions (2x bonus) to handle cross-topic reuse
    of the same symbol (e.g. 'm' = mass in mechanics, magnification in optics).
    """
    cands = registry.get(sym, [])
    if not cands:
        return None

    def score(vd):
        chapter_match = 2.0 if (eq_chapter > 0 and vd.chapter == eq_chapter) else 1.0
        d = abs(vd.page - eq_page)
        if d == 0:
            base = vd.confidence
        elif d <= window:
            base = vd.confidence * (1 - d / (window + 1) * 0.6)
        else:
            base = vd.confidence * 0.2
        return base * chapter_match

    best = max(cands, key=score)
    best.confidence = round(score(best), 2)
    return best


# Known physics/math symbol fallbacks (used when document doesn't define a symbol)
PHYSICS_FALLBACKS = {
    'F':'force', 'm':'mass', 'a':'acceleration', 'v':'velocity',
    'u':'initial_velocity', 's':'displacement', 't':'time',
    'g':'gravitational_acceleration', 'W':'weight', 'P':'power',
    'E':'energy', 'k':'spring_constant', 'p':'momentum',
    'L':'length', 'A':'area', 'V':'voltage', 'T':'period',
    'f':'frequency', 'r':'radius', 'h':'height', 'n':'moles',
    'N':'particles', 'Q':'heat', 'U':'internal_energy',
    'R':'resistance', 'C':'capacitance', 'B':'magnetic_field',
    'q':'charge', 'I':'current', 'Y':'young_modulus',
    'K':'kinetic_energy', 'G':'gravitational_constant',
    'lambda':'wavelength', 'omega':'angular_velocity',
    'alpha':'angular_acceleration', 'theta':'angle',
    'sigma':'stress', 'epsilon':'strain', 'tau':'torque',
    'rho':'density', 'mu':'friction_coefficient',
    'phi':'flux', 'nu':'frequency', 'pi':'pi',
    'Delta':'change_in', 'nabla':'gradient',
    'sin':'sin', 'cos':'cos', 'tan':'tan',
    'sqrt':'sqrt', 'ln':'ln', 'log':'log',
}


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 8 — Canonicalization
# ─────────────────────────────────────────────────────────────────────────────
# Math function names that should never be replaced by variable names
MATH_FUNCTIONS = {
    'sin','cos','tan','sqrt','log','ln','exp','max','min',
    'sum','integral','proportional_to','approx','Delta',
    'nabla','inf','d',
}

# Tokenizer: match word tokens not inside apostrophe contractions
TOK_RE = re.compile(r"(?<![A-Za-z'\-])([A-Za-z_][A-Za-z_0-9]*)(?![A-Za-z'\-])")


def canonicalize(raw: str, eq_page: int, registry: dict,
                 eq_chapter: int = 0) -> tuple:
    """
    Replace symbol tokens in equation string with English names.
    Returns (canonical_string, variables_dict).
    """
    variables = {}

    def replace(m):
        sym = m.group(1)
        if sym in MATH_FUNCTIONS:
            return sym
        if sym in variables:
            return variables[sym]['name']
        vd = resolve(sym, eq_page, registry, eq_chapter=eq_chapter)
        if vd:
            variables[sym] = asdict(vd)
            return vd.name
        if sym in PHYSICS_FALLBACKS:
            name = PHYSICS_FALLBACKS[sym]
            variables[sym] = asdict(VarDef(
                symbol=sym, name=name, page=-1, confidence=0.4))
            return name
        return sym  # leave unknown symbols as-is

    canonical = TOK_RE.sub(replace, raw)
    return canonical, variables


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 9 — OCR fallback (for scanned/garbage pages)
# ─────────────────────────────────────────────────────────────────────────────
def is_garbage(text: str) -> bool:
    """Heuristic: is extracted text garbled (scanned page)?"""
    if not text or len(text) < 20:
        return True
    weird = sum(1 for c in text if ord(c) > 127 and c not in GREEK)
    return weird / len(text) > 0.3


def ocr_page(page: fitz.Page, dpi: int = 200) -> str:
    """Render page to image and run Tesseract OCR."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img = Image.open(io.BytesIO(pix.tobytes('png')))
    return pytesseract.image_to_string(img, config='--oem 3 --psm 6')


def body_size(spans: list) -> float:
    """Find the most common (body text) font size on a page."""
    if not spans:
        return 10.0
    sizes = [round(s.size) for s in spans]
    return max(set(sizes), key=sizes.count)


# ─────────────────────────────────────────────────────────────────────────────
#  Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run(pdf_path: str, output_path: str = None) -> dict:
    """
    Run the full extraction pipeline.

    Args:
        pdf_path:    Path to the input PDF file.
        output_path: Path for the JSON output (default: same dir as PDF).

    Returns:
        dict with keys: source_pdf, total_pages, total_equations,
                        variable_registry, equations
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f'PDF not found: {pdf_path}')
    if output_path is None:
        output_path = pdf_path.with_suffix('.equations.json')

    doc = fitz.open(str(pdf_path))
    log.info(f'PDF: {pdf_path.name}  ({len(doc)} pages)')

    all_blocks: list = []
    all_defs: list = []

    for i, page in enumerate(doc):
        pn = i + 1
        raw_text = page.get_text('text')

        if is_garbage(raw_text):
            # Scanned page: use OCR
            log.info(f'  p{pn}: OCR fallback (scanned page)')
            text = norm(ocr_page(page))
            method = 'ocr'
            for line in text.split('\n'):
                line = line.strip()
                clean = re.sub(r'\(\d+(?:\.\d+)?\)\s*$', '', line).strip()
                if is_valid_equation(clean):
                    all_blocks.append(MathBlock(
                        raw=clean, page=pn, y0=0, method='ocr'))
            all_defs.extend(extract_defs(text, pn))
        else:
            # Digital page: span-level extraction
            method = 'text'
            spans = get_spans(page, pn)
            if not spans:
                continue
            bsize = body_size(spans)
            cols = detect_columns(spans, page.rect.width)

            for col in cols:
                cs = col_spans(spans, col)
                if not cs:
                    continue
                rows = group_rows(cs)
                blocks = reconstruct_blocks(rows, bsize)
                for b in blocks:
                    b.method = method
                    if is_valid_equation(b.raw):
                        all_blocks.append(b)

            all_defs.extend(extract_defs(norm(raw_text), pn))

        eq_ct = sum(1 for b in all_blocks if b.page == pn)
        df_ct = sum(1 for d in all_defs if d.page == pn)
        if eq_ct or df_ct:
            log.info(f'  p{pn} [{method}]: {eq_ct} equations  {df_ct} definitions')

    # Build page→chapter map from definitions for cross-topic scoping
    page_to_chapter = {}
    for vd in all_defs:
        if vd.page not in page_to_chapter:
            page_to_chapter[vd.page] = vd.chapter
    for b in all_blocks:
        b.chapter = page_to_chapter.get(b.page, 0)

    # Deduplicate equations (same raw text)
    seen_raws = set()
    unique = []
    for b in all_blocks:
        k = re.sub(r'\s+', ' ', b.raw).strip()
        if k not in seen_raws:
            seen_raws.add(k)
            unique.append(b)

    registry = build_registry(all_defs)
    log.info(f'Variable registry: {len(registry)} unique symbols')

    equations = []
    for idx, b in enumerate(unique):
        # Strip inline label prefixes: "Newton's Law: F = m*a" → "F = m*a"
        raw = b.raw
        lhs_part, eq_sep, rhs_part = raw.partition('=')
        if eq_sep and ':' in lhs_part:
            candidate = lhs_part.rsplit(':', 1)[-1].strip()
            if candidate and len(candidate.split()) <= 2 \
                    and re.match(r'^[A-Za-z_]', candidate):
                raw = candidate + ' = ' + rhs_part.strip()
                b.raw = raw

        canon, variables = canonicalize(
            b.raw, b.page, registry, eq_chapter=b.chapter)

        equations.append(asdict(Equation(
            eq_id=f'eq_{idx + 1:04d}_p{b.page}',
            page=b.page, raw=b.raw, canonical=canon,
            variables=variables, method=b.method,
        )))

    result = {
        'source_pdf': str(pdf_path),
        'total_pages': len(doc),
        'total_equations': len(equations),
        'variable_registry': {
            sym: [asdict(v) for v in vds]
            for sym, vds in registry.items()
        },
        'equations': equations,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    log.info(f'Done — {len(equations)} equations written to {output_path}')
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    pdf = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        result = run(pdf, out)
        print(f'\nExtracted {result["total_equations"]} equations '
              f'from {result["total_pages"]} pages.')
        if result['equations']:
            print('\nSample output:')
            for eq in result['equations'][:5]:
                print(f'  [{eq["page"]}] {eq["canonical"]}')
            if len(result['equations']) > 5:
                print(f'  ... ({len(result["equations"]) - 5} more)')
    except FileNotFoundError as e:
        print(f'Error: {e}')
        sys.exit(1)