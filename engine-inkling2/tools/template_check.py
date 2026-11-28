#!/usr/bin/env python3
"""template_check.py MODEL.gguf TOKENIZER GGUF_PY: serve/ink2_server.py's render() against Inkling 2's own chat template (the GGUF's Jinja, rendered
the way transformers does it: trim_blocks, lstrip_blocks, the {% generation %} tag, tojson without ASCII escapes) on
conversations with tools, tool calls, tool results, reasoning and Slovenian text. Every case must give the same text to the
character. Needs jinja2 (pip install jinja2 in the venv)."""
import json
import sys

from pathlib import Path
MODEL, TOK, GPY = sys.argv[1:4]
sys.argv = [sys.argv[0], "--port", "1", "--sock", "-", "--model", MODEL, "--tokenizer", TOK, "--gguf-py", GPY]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "serve"))
import ink2_server as srv  # noqa: E402
from jinja2 import nodes  # noqa: E402
from jinja2.ext import Extension, loopcontrols  # noqa: E402
from jinja2.sandbox import ImmutableSandboxedEnvironment  # noqa: E402


class Generation(Extension):
    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return nodes.CallBlock(self.call_method("_out"), [], [], body).set_lineno(lineno)

    def _out(self, caller):
        return caller()


def raise_exception(msg):
    raise RuntimeError(msg)


env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[Generation, loopcontrols])
env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(
    x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
env.globals["raise_exception"] = raise_exception
TMPL = env.from_string(srv.TEMPLATE)

BOLHA = {"type": "function", "function": {"name": "bolha_search", "description": "Ads on bolha.com: title, price, location, date, link.",
                                          "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "A short term, e.g. \"rtx 3090\"."}},
                                                         "required": ["query"]}}}
WEB = {"type": "function", "function": {"name": "web_fetch", "description": "Read a public web page.",
                                        "parameters": {"type": "object", "properties": {"url": {"type": "string"}, "part": {"type": "integer"}},
                                                       "required": ["url"]}}}
call = lambda i, name, args: {"id": i, "type": "function", "function": {"name": name, "arguments": args}}   # noqa: E731

CASES = [
    ("navaden", None, "none", [{"role": "user", "content": "Kdo je napisal Krst pri Savici?"}]),
    ("sistem + razmislek", None, "low", [
        {"role": "system", "content": "Odgovarjaš kratko, v slovenščini."},
        {"role": "user", "content": "Koliko je 17 × 23?"},
        {"role": "assistant", "content": "391.", "reasoning_content": "17·23 = 391"},
        {"role": "user", "content": "In 18 × 23?"}]),
    ("orodja, klic, rezultat", [BOLHA, WEB], "high", [
        {"role": "system", "content": "Si Klepet. Za oglase uporabi bolha_search."},
        {"role": "user", "content": "Poišči rabljen RTX 3090 na Bolhi, čim cenejši."},
        {"role": "assistant", "content": "", "reasoning_content": "Uporabim orodje.", "tool_calls": [call("c1", "bolha_search", "{\"query\": \"rtx 3090\"}")]},
        {"role": "tool", "tool_call_id": "c1", "content": "[{\"title\": \"RTX 3090 Founders\", \"price\": \"650 €\", \"location\": \"Celje\"}]"},
        {"role": "user", "content": "Še kaj v Ljubljani?"}]),
    ("dva klica, gnezdeni argumenti", [WEB, BOLHA], 0.7, [
        {"role": "user", "content": "Preberi stran in poišči oglase."},
        {"role": "assistant", "content": None, "tool_calls": [call("a", "web_fetch", {"url": "https://example.com/č", "part": 2}),
                                                             call("b", "bolha_search", "{\"query\": \"kolo\", \"Z\": [1, {\"b\": 2, \"a\": \"ž\"}]}")]},
        {"role": "tool", "name": "web_fetch", "content": "Besedilo strani …"},
        {"role": "tool", "tool_call_id": "b", "content": "[]"}]),
    ("vsebina po delih", None, "medium", [
        {"role": "user", "content": [{"type": "text", "text": "Prvi del."}, {"type": "text", "text": "Drugi del."}]}]),
]

bad = 0
for name, tools, eff, msgs in CASES:
    jmsgs = json.loads(json.dumps(msgs))
    for m in jmsgs:   # the template wants arguments as objects (it says "canonicalize upstream")
        for tc in m.get("tool_calls") or []:
            a = tc["function"]["arguments"]
            tc["function"]["arguments"] = json.loads(a) if isinstance(a, str) else a
    ref = TMPL.render(messages=jmsgs, tools=tools, reasoning_effort=eff, add_generation_prompt=False)
    ours = srv.render(msgs, eff, tools)
    # effort 0 with no tools: the server adds the answer's start on purpose (an answer without reasoning, 23 Nov)
    if eff in (0, 0.0, "none") and not tools and ours.endswith("<|message_model|><|content_text|>"):
        ours = ours[: -len("<|message_model|><|content_text|>")]
    if ref == ours:
        print(f"{name}: same ({len(ref)} characters)")
    else:
        bad += 1
        k = next((i for i, (x, y) in enumerate(zip(ref, ours)) if x != y), min(len(ref), len(ours)))
        print(f"{name}: DIFFERENT from character {k}\n  template: {ref[max(0, k - 80):k + 80]!r}\n  ours:     {ours[max(0, k - 80):k + 80]!r}")
print("DONE", "all the same" if not bad else f"{bad} different")
