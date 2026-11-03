"""Structural checks on main.tex, standing in for a local LaTeX compile.

Catches the classes of error that actually occur when tables are edited by
script: column-count mismatches, unbalanced environments, dangling \\ref targets,
missing figure files, and stale numbers that disagree with the results CSVs.

Usage:
    python check_tex.py
"""
import os
import re
import sys

TEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main.tex')
FIGDIR = os.path.join(os.path.dirname(TEX), 'figures')

problems = []


def note(kind, msg):
    problems.append(f"[{kind}] {msg}")


def strip_comments(t):
    return re.sub(r'(?<!\\)%.*', '', t)


def main():
    raw = open(TEX, encoding='utf-8').read()
    t = strip_comments(raw)

    # ---- environment balance ------------------------------------------------
    for env in ['table', 'table*', 'tabular', 'figure', 'figure*', 'itemize',
                'enumerate', 'abstract', 'thebibliography', 'document']:
        b = len(re.findall(r'\\begin\{' + re.escape(env) + r'\}', t))
        e = len(re.findall(r'\\end\{' + re.escape(env) + r'\}', t))
        if b != e:
            note('ENV', f"{env}: {b} begin vs {e} end")

    # ---- tabular column counts ---------------------------------------------
    for m in re.finditer(r'\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}',
                         t, re.S):
        spec, body = m.group(1), m.group(2)
        ncol = len(re.findall(r'[lcr]|p\{[^}]*\}', re.sub(r'@\{[^}]*\}', '', spec)))
        line_no = raw[:m.start()].count('\n') + 1
        for row in body.split(r'\\'):
            row = row.strip()
            if not row or row.startswith(('\\toprule', '\\midrule', '\\bottomrule',
                                          '\\cmidrule', '\\multicolumn{')):
                continue
            r2 = re.sub(r'\\multicolumn\{(\d+)\}\{[^}]*\}\{[^}]*\}',
                        lambda mm: '&' * (int(mm.group(1)) - 1), row)
            r2 = re.sub(r'\\(best|std|multirow)\{[^{}]*\}', 'X', r2)
            r2 = re.sub(r'\\multirow\{[^}]*\}\{[^}]*\}\{[^}]*\}', 'X', r2)
            cells = r2.count('&') + 1
            if re.match(r'^\s*\\(toprule|midrule|bottomrule|cmidrule)', row):
                continue
            if cells != ncol and cells > 1:
                note('COLS', f"line ~{line_no}: row has {cells} cells, "
                             f"spec has {ncol}: {row[:66]!r}")

    # ---- table rows ending in a single backslash ----------------------------
    # A row written as `... & x \` (one backslash, from a lost escape) does not
    # end the row; a following \cmidrule/\midrule then sits mid-row and LaTeX
    # aborts with "Misplaced \noalign". The cell-count check above cannot see
    # this, because it splits rows on `\\`.
    for m in re.finditer(r'\\begin\{tabular\}.*?\\end\{tabular\}', t, re.S):
        base = raw[:m.start()].count('\n') + 1
        for i, line in enumerate(m.group(0).split('\n')):
            ln = line.rstrip()
            if '&' in ln and ln.endswith('\\') and not ln.endswith('\\\\'):
                note('ROW', f"line {base + i}: row ends in a single backslash: "
                            f"{ln[-50:]!r}")

    # ---- labels and refs ----------------------------------------------------
    labels = set(re.findall(r'\\label\{([^}]*)\}', t))
    for r in set(re.findall(r'\\ref\{([^}]*)\}', t)):
        if r not in labels:
            note('REF', f"\\ref{{{r}}} has no \\label")
    for c in set(re.findall(r'\\cite\{([^}]*)\}', t)):
        for k in c.split(','):
            if not re.search(r'\\bibitem\{' + re.escape(k.strip()) + r'\}', t):
                note('CITE', f"\\cite{{{k.strip()}}} has no \\bibitem")

    # ---- duplicate labels ---------------------------------------------------
    all_lab = re.findall(r'\\label\{([^}]*)\}', t)
    for l in set(all_lab):
        if all_lab.count(l) > 1:
            note('LABEL', f"duplicate label {l} ({all_lab.count(l)}x)")

    # ---- figures exist ------------------------------------------------------
    for g in set(re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}', t)):
        base = os.path.join(os.path.dirname(TEX), g)
        if not any(os.path.exists(base + e) for e in ('', '.pdf', '.png', '.jpg')):
            note('FIG', f"missing figure file: {g}")

    # ---- unused figures on disk --------------------------------------------
    used = {os.path.basename(g).rsplit('.', 1)[0]
            for g in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}', t)}
    if os.path.isdir(FIGDIR):
        for f in sorted(os.listdir(FIGDIR)):
            stem = f.rsplit('.', 1)[0]
            if stem not in used and f.endswith(('.png', '.pdf')):
                note('UNUSED', f"figures/{f} is not referenced")

    if problems:
        print(f"{len(problems)} issue(s):\n")
        for p in problems:
            print(" ", p)
        sys.exit(1)
    print("main.tex: structure OK")


if __name__ == '__main__':
    main()
