#!/usr/bin/env python3
"""draft_vocab.py OUT.bin K TOKENIZER MODEL.gguf GGUF_PY TEXT_FILE... : a reduced vocabulary for the DSpark drafter's head
(--draft-vocab): the tokens of the given texts by count, every special token (<|...|>), then the lowest ids (in a BPE
vocabulary the earliest merges) until there are K; written as int32 count + ids (sorted). data/draft_vocab_32768.bin was made
from ~1.3 M tokens of English docs, C++ / Python code and Slovenian text: 15,213 distinct tokens, which covered 98-99 % of
held-out text, filled up to 32,768. A token outside the set can never be drafted; the target still checks every draft, so
the answers do not change, only how many drafts are taken."""
import collections
import json
import struct
import subprocess
import sys

OUT, K, TOKENIZER, MODEL, GGUF_PY = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
sys.path.insert(0, GGUF_PY)
import gguf  # noqa: E402

NVU = 200058   # Inkling 2's real vocabulary (the rest of the 201,024 rows is padding)


def tok(text):
    ids = []
    for i in range(0, len(text), 200_000):
        p = subprocess.run([TOKENIZER, "-m", MODEL, "--ids", "--log-disable", "--stdin"], input=text[i:i + 200_000],
                           capture_output=True, text=True, env={"CUDA_VISIBLE_DEVICES": ""})
        line = [x for x in p.stdout.splitlines() if x.startswith("[")]
        ids += json.loads(line[-1]) if line else []
    return ids


cnt = collections.Counter()
for f in sys.argv[6:]:
    cnt.update(tok(open(f, encoding="utf-8", errors="replace").read()))
r = gguf.GGUFReader(MODEL)
fld = r.fields["tokenizer.ggml.tokens"]
keep = {i for i, j in enumerate(fld.data) if bytes(fld.parts[j]).decode("utf-8", "replace").startswith("<|") and i < NVU}
for i, _ in cnt.most_common():
    if len(keep) >= K:
        break
    if i < NVU:
        keep.add(i)
i = 0
while len(keep) < K:
    keep.add(i)
    i += 1
ids = sorted(keep)[:K]
open(OUT, "wb").write(struct.pack("i", len(ids)) + struct.pack(f"{len(ids)}i", *ids))
print(f"{OUT}: {len(ids)} tokens, {len(cnt)} distinct in the texts ({sum(cnt.values())} tokens)")
