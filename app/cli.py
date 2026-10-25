"""Task 4 -- terminal version of the Amharic next-word suggester.

    python app/cli.py [--order 3]

Type words and press Enter to add them; enter 1-5 to accept a suggestion;
end a word with '…' style partial input by typing it followed by '+'
(e.g. 'አብ+') to see completions.  Commands: :2 :3 :4 change the context
length, :c clears the sentence, :q quits.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amharic_lm import pipeline  # noqa: E402
from amharic_lm.predictor import NextWordPredictor  # noqa: E402


def show(pred: NextWordPredictor, text: str):
    res = pred.suggest(text)
    print("\n" + "─" * 60)
    print(f"  text   : {text.strip() or '(empty)'}")
    print(f"  context: {' '.join(res['context']) or '-'}   ({pred.order}-gram"
          + (f", completing '{res['prefix']}'" if res["prefix"] else "") + ")")
    for i, s in enumerate(res["suggestions"], 1):
        print(f"   {i}. {s['word']:<16} P={s['p']:.4f}   seen as {s['matched']:<8} count={s['count']:,}")
    if not res["suggestions"]:
        print("   (no suggestions)")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=3, choices=[2, 3, 4])
    args = ap.parse_args()
    print("loading model ...")
    pred = NextWordPredictor(pipeline.full_model(mmap=True), args.order)
    text = ""
    while True:
        res = show(pred, text)
        try:
            cmd = input("» ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if cmd in (":q", "q"):
            break
        if cmd == ":c":
            text = ""
        elif cmd in (":2", ":3", ":4"):
            pred.order = int(cmd[1])
        elif cmd.isdigit() and 1 <= int(cmd) <= len(res["suggestions"]):
            text = pred.apply(text, res["suggestions"][int(cmd) - 1]["word"])
        elif cmd.endswith("+"):
            text = text.rstrip() + (" " if text.strip() else "") + cmd[:-1]
        elif cmd:
            text = text.rstrip() + (" " if text.strip() else "") + cmd + " "


if __name__ == "__main__":
    main()
