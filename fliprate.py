import json
import argparse
import string
import re

_EQUIV = {
    "indoor": "indoors",
    "indoors": "indoors",
    "outdoor": "outdoors",
    "outdoors": "outdoors",
    "out": "outdoors",
    "indo": "indoors"
}

def read_texts_with_lines(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append((lineno, _normalize_text(obj.get("text"))))
    return rows

def _normalize_text(t):
    if t is None:
        return None
    s = str(t).strip().lower()
    # keep only the first alphabetic word, drop everything after it
    m = re.search(r"[a-z]+", s)
    s = m.group(0) if m else s
    return _EQUIV.get(s, s)

def read_texts(path):
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            texts.append(_normalize_text(obj.get("text")))
    return texts

def fliprate(baseline_path, test_path):
    a0 = read_texts(baseline_path)
    a1 = read_texts(test_path)
    n = min(len(a0), len(a1))
    if n == 0:
        return 0.0
    flips = sum(1 for i in range(n) if a0[i] != a1[i])
    return flips / n

def print_mismatches(baseline_path, test_path):
    a0 = read_texts_with_lines(baseline_path)
    a1 = read_texts_with_lines(test_path)
    n = min(len(a0), len(a1))
    mismatches = 0
    for i in range(n):
        l0, t0 = a0[i]
        l1, t1 = a1[i]
        if t0 != t1:
            mismatches += 1
            print(f"[MISMATCH] A:{l0} text={t0!r}")
            print(f"           B:{l1} text={t1!r}")
    return mismatches

def main():
    p = argparse.ArgumentParser()
    p.add_argument("baseline")
    p.add_argument("test")
    p.add_argument("--mismatches", action="store_true",
                   help="Print lines where 'text' differs.")
    args = p.parse_args()

    if args.mismatches:
        n = print_mismatches(args.baseline, args.test)
        print(f"Total mismatches: {n}")
    else:
        print(fliprate(args.baseline, args.test))

if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) != 3:
#         print("Usage: python fliprate.py baseline.jsonl test.jsonl")
#         raise SystemExit(1)
#     print(fliprate(sys.argv[1], sys.argv[2]))