#!/usr/bin/env python3
"""Benchmark an already running local LLM server (Basalt or Strata) over its OpenAI API, then check for new ECC errors.

It starts no engine: start the server first (Basalt: bin/basalt serve --config basalt_config.json; Strata: its own
launcher), then run

    python engine_bench.py [--url http://127.0.0.1:8080] [--tests community,modeldock] [--runs 2] [--out result.json]

Tests
  community  Basalt's community-1 workload over HTTP: the conv, struct and prose prompts of data/bench (thinking off,
             temperature 0.6 / 0 / 0.6, at most 1200 / 8192 / 1200 tokens) and prefill at 16K and 64K depth (the bench
             document once and four times, behind a random prefix so no prompt cache matches). decode_tps is the
             generated tokens of the three prompts over their summed decode time, as basalt-bench --community counts
             it. Prefill is prompt tokens over the time to the first token, so it includes the HTTP round trip.
  modeldock  ModelDock's local benchmark, protocol modeldock-bench/v3 (runner 0.1.5), run locally: a fingerprint and a
             cache warm-up (untimed), --runs probe runs of about 12-15K prompt tokens and 512 output tokens with five
             needles and a table to recall (decode: the median of the valid runs), and a long read of 32K-64K tokens
             (prefill). The prompts are the same text the reference runner builds for the same nonce; speeds come from
             the engine's own `timings` when the server reports them, else from the client clock. The upload, the
             published ranking and the /tokenize token check are left out.

A test whose prompt and output do not fit the server's context is skipped and named. After the tests, on Windows, every
corrected memory error (WHEA-Logger 47) logged since the start is counted and decoded to its memory channel and DIMM
(AMD memory controller registers). Nothing is downloaded or uploaded: only the given URL is contacted.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import random
import re
import secrets
import statistics
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.path.join(HERE, "bench_prompts.json")

# community-1 decode prompts: (name, temperature, max tokens)
DECODE = [("conv", 0.6, 1200), ("struct", 0.0, 8192), ("prose", 0.6, 1200)]
PREFILL_COPIES = {"16K": 1, "64K": 4}   # the bench document repeated to each depth


# ---------------------------------------------------------------- HTTP
def get_json(url: str, timeout: float = 10.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def num(v):
    """A finite non-negative number, or None (strings of digits too, as the reference runner reads them)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) and f >= 0 else None


class Server:
    def __init__(self, base: str, model: str | None, seed: int, timeout: float):
        self.base = base.rstrip("/")
        if self.base.endswith("/v1"):
            self.base = self.base[:-3]
        self.timeout = timeout
        self.seed = seed
        self.health = {}
        try:
            self.health = get_json(self.base + "/health")
        except Exception:
            pass
        models = get_json(self.base + "/v1/models")
        data = models.get("data") or []
        if not data and not model:
            raise SystemExit(f"{self.base}/v1/models lists no model")
        self.model = model or data[0]["id"]
        meta = next((m.get("meta") or {} for m in data if m.get("id") == self.model), {})
        self.ctx = int(meta.get("n_ctx") or self.health.get("max_context") or 0)
        self.service = self.health.get("service") or "unknown"

    def stream(self, body: dict, timeout: float | None = None) -> dict:
        """One streamed chat completion: its text, the server's usage and timings, and the client clock."""
        body = dict(body, model=self.model, stream=True, stream_options={"include_usage": True})
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json"})
        t0 = time.perf_counter()
        t_first = None
        content, reasoning = [], []
        chunks = frames = 0
        usage, timings = {}, {}
        finish = None
        done = malformed = False
        stream_error = None
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        done = True
                        break
                    try:
                        ev = json.loads(data)
                    except ValueError:
                        malformed = True
                        continue
                    if not isinstance(ev, dict):
                        malformed = True
                        continue
                    frames += 1
                    if ev.get("error") and stream_error is None:
                        stream_error = str(ev["error"])[:200]
                    if isinstance(ev.get("usage"), dict):
                        usage.update(ev["usage"])
                    if isinstance(ev.get("timings"), dict):
                        timings.update(ev["timings"])
                    for ch in ev.get("choices") or []:
                        d = ch.get("delta") or {}
                        c = d.get("content") if isinstance(d.get("content"), str) else ""
                        r = d.get("reasoning_content") if isinstance(d.get("reasoning_content"), str) else (
                            d.get("reasoning") if isinstance(d.get("reasoning"), str) else "")
                        if c or r:
                            if t_first is None:
                                t_first = time.perf_counter()
                            chunks += 1
                        content.append(c)
                        reasoning.append(r)
                        finish = ch.get("finish_reason") or finish
        except urllib.error.HTTPError as e:
            return {"error": f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}", "status": e.code}
        except Exception as e:   # connection reset, timeout
            return {"error": f"{type(e).__name__}: {e}", "status": None}
        t_end = time.perf_counter()
        return {"content": "".join(content), "reasoning": "".join(reasoning), "usage": usage, "timings": timings,
                "chunks": chunks, "ttft_s": (t_first or t_end) - t0, "delivery_s": (t_end - t_first) if t_first else 0.0,
                "elapsed_s": t_end - t0, "finish": finish,
                "complete": frames > 0 and finish is not None and done and not malformed and stream_error is None,
                "stream_error": stream_error}

    def chat(self, prompt: str, max_tokens: int, temperature: float) -> dict:
        """The community requests: thinking off, seeded, times and token counts."""
        r = self.stream({"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                         "temperature": temperature, "seed": self.seed,
                         "chat_template_kwargs": {"enable_thinking": False}})
        if "error" in r:
            return r
        prompt_tokens = r["usage"].get("prompt_tokens")
        out_tokens = r["usage"].get("completion_tokens")
        if out_tokens is None:
            out_tokens = r["chunks"]   # no usage from the server: stream chunks, about one token each
        return {"prompt_tokens": prompt_tokens, "out_tokens": out_tokens, "ttft_s": r["ttft_s"],
                "decode_s": r["delivery_s"], "finish": r["finish"],
                "pp_tps": (prompt_tokens / r["ttft_s"]) if prompt_tokens and r["ttft_s"] > 0 else None,
                "tg_tps": ((out_tokens - 1) / r["delivery_s"]) if out_tokens and out_tokens > 1 and r["delivery_s"] > 0
                else None}


def fits(srv: Server, approx_prompt: int, max_tokens: int) -> bool:
    return srv.ctx <= 0 or approx_prompt + max_tokens <= srv.ctx


# ---------------------------------------------------------------- community
def community(srv: Server, prompts: dict) -> dict:
    print("== community-1 workload")
    res = {"decode": {}, "prefill": {}}
    for name, temp, mt in DECODE:
        r = srv.chat(prompts["decode"][name], mt, temp)
        res["decode"][name] = r
        if "error" in r:
            print(f"  {name:7s} ERROR {r['error']}")
        else:
            print(f"  {name:7s} {r['out_tokens']:5d} tokens  {r['tg_tps'] or 0:7.1f} tok/s  (TTFT {r['ttft_s']:.2f} s)")
    ok = [r for r in res["decode"].values() if "error" not in r and r["decode_s"] > 0]
    if len(ok) == len(DECODE):
        tok = sum(r["out_tokens"] - 1 for r in ok)
        sec = sum(r["decode_s"] for r in ok)
        res["decode_tps"] = tok / sec
        print(f"  decode_tps {res['decode_tps']:.1f} tok/s  ({tok} tokens in {sec:.1f} s, token-weighted)")
    p = prompts["prefill"]
    for depth, copies in PREFILL_COPIES.items():
        approx = p["body_tokens"] * copies + 40
        if not fits(srv, approx, 1):
            print(f"  prefill {depth}: skipped (about {approx} tokens; the server's context is {srv.ctx})")
            res["prefill"][depth] = {"skipped": f"context {srv.ctx}"}
            continue
        nonce = " ".join(f"{random.randrange(16 ** 6):06x}" for _ in range(4))   # no prompt cache can match
        r = srv.chat(p["head"] + nonce + "\n" + p["body"] * copies + p["tail"], 1, 0.0)
        res["prefill"][depth] = r
        if "error" in r:
            print(f"  prefill {depth}: ERROR {r['error']}")
        else:
            print(f"  prefill {depth}: {r['prompt_tokens']} tokens  {r['pp_tps'] or 0:7.0f} tok/s  "
                  f"(TTFT {r['ttft_s']:.2f} s)")
    return res


# ---------------------------------------------------------------- ModelDock protocol v3
# Ported from modeldock-bench.mjs 0.1.5 (sha256 6d2e414f...229b): the same constants, the same prompt for the same
# nonce, the same checks and the same headline rules.
M32 = 0xFFFFFFFF
MAX_OUTPUT_TOKENS, MIN_OUTPUT_TOKENS = 512, 128
DEFAULT_PROMPT_CHARS = 56_000
NEEDLE_POSITIONS = (0.01, 0.25, 0.5, 0.75, 0.99)
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
TABLE_ROWS = 6
FINGERPRINT_PROMPT = ('Continue this sentence with exactly one more plain English sentence and nothing else: "A careful '
                      'local benchmark begins by writing down every setting, because"')
FINGERPRINT_MAX_TOKENS = 32
CACHE_WARMUP_PROMPT_CHARS, CACHE_WARMUP_MAX_TOKENS = 18_000, 256
UNSTABLE_SPREAD = 0.15
LONG_READ_MAX_TOKENS, LONG_READ_MIN_TOKENS, LONG_READ_OUTPUT_TOKENS = 65_536, 32_768, 32
PREFIX_CACHE_TOKENS = 64
FILLER = (
    "A repeatable local test starts by writing down the exact software versions, the hardware in use and every setting that could change the outcome.",
    "Each run should use the same inputs, so that a difference in the result can be traced to the system rather than to the workload.",
    "Observations are recorded as they happen, with timestamps, instead of being reconstructed from memory after the test is over.",
    "When an error appears, it is kept in the log together with the conditions that produced it, because a failure is also a measurement.",
    "Uncertainty is reported alongside every number, and a single unusually good run is never presented as the typical result.",
    "Before comparing two machines, the tester confirms that both ran the same model file, the same quantization and the same context size.",
    "Background tasks are closed, power settings are fixed and the device is allowed to reach a steady temperature before timing begins.",
    "A warm-up request loads the model and fills caches that would otherwise distort the first timed measurement.",
    "Results are shared with enough detail that another person could repeat the test and expect a similar outcome.",
    "If a measurement cannot be explained, it is marked for review rather than silently removed from the record.",
    "The final report separates what was measured directly from what was estimated, and names the source of every estimate.",
    "Small differences between runs are expected; the goal is a fair description of typical behaviour, not a single record.",
)
THINK_BLOCK = re.compile(r"<think>[\s\S]*?(</think>|$)", re.I)


def imul(a: int, b: int) -> int:
    return ((a & M32) * (b & M32)) & M32


def fnv1a32(text: str) -> int:
    h = 0x811C9DC5
    for ch in text:
        h = imul(h ^ ord(ch), 0x01000193)
    return h


def mulberry32(seed: int):
    state = seed & M32

    def rnd() -> float:
        nonlocal state
        state = (state + 0x6D2B79F5) & M32
        t = state
        t = imul(t ^ (t >> 15), t | 1)
        t ^= (t + imul(t ^ (t >> 7), t | 61)) & M32
        return ((t ^ (t >> 14)) & M32) / 4294967296
    return rnd


def new_nonce() -> str:
    return "".join(CODE_ALPHABET[b % len(CODE_ALPHABET)] for b in secrets.token_bytes(12))


def expected_answers(nonce: str) -> dict:
    rnd = mulberry32(fnv1a32(f"answers:{nonce}"))
    segments = ["".join(CODE_ALPHABET[math.floor(rnd() * len(CODE_ALPHABET))] for _ in range(4)) for _ in NEEDLE_POSITIONS]
    quantities = [10 + math.floor(rnd() * 90) for _ in range(TABLE_ROWS)]
    return {"segments": segments, "code": "-".join(segments), "quantities": quantities, "sum": sum(quantities)}


def build_prompt(nonce: str, chars: int = DEFAULT_PROMPT_CHARS) -> str:
    answers = expected_answers(nonce)
    rnd = mulberry32(fnv1a32(f"filler:{nonce}"))
    sentences: list[str] = []
    length = 0
    while length < chars:
        s = FILLER[math.floor(rnd() * len(FILLER))]
        sentences.append(s)
        length += len(s) + 1
    n = len(sentences)
    inserts = [(min(n - 1, math.floor(pos * n)),
                f"Reference note {i + 1} of 5: access code segment {i + 1} is {answers['segments'][i]}.")
               for i, pos in enumerate(NEEDLE_POSITIONS)]
    table = " ".join(f"Item {chr(65 + i)} has quantity {q}." for i, q in enumerate(answers["quantities"]))
    inserts.append((math.floor(0.6 * n), f"Inventory table. {table} End of inventory table."))
    inserts.sort(key=lambda x: -x[0])   # stable, as the reference runner's sort
    for at, text in inserts:
        sentences.insert(at, text)
    body = " ".join(sentences)
    return "\n".join([
        f"RUN {nonce}",
        "You are reading a long ModelDock benchmark document. Read all of it: the instructions at the end refer to "
        "details spread throughout.",
        "",
        body,
        "",
        "Instructions:",
        f'1. On the first line, output only one JSON object and nothing else: {{"run":"{nonce}","code":"<segment 1>-'
        f'<segment 2>-<segment 3>-<segment 4>-<segment 5>","sum":<the sum of the six inventory quantities>}}',
        "2. Then write a detailed essay of at least 700 words about organizing a careful, repeatable local software "
        "test, in plain English paragraphs. Do not use lists, headings or code, and do not stop early.",
    ])


def probe_prompt_chars(ctx: int) -> int:
    if not ctx:
        return DEFAULT_PROMPT_CHARS
    return max(8_192 * 4, min(DEFAULT_PROMPT_CHARS, math.floor((ctx - MAX_OUTPUT_TOKENS - 512) * 3.6)))


def long_read_plan(ctx: int, chars_per_token: float, prefill_tps, budget_s: float) -> dict:
    room = ctx - LONG_READ_OUTPUT_TOKENS - 256 if ctx else LONG_READ_MAX_TOKENS
    tokens = min(LONG_READ_MAX_TOKENS, room)
    if prefill_tps and tokens / prefill_tps > budget_s:
        tokens = max(LONG_READ_MIN_TOKENS, min(tokens, math.floor(prefill_tps * budget_s)))
    if tokens < LONG_READ_MIN_TOKENS:
        return {"feasible": False, "reason": "context-too-small",
                "needed": LONG_READ_MIN_TOKENS + LONG_READ_OUTPUT_TOKENS + 256}
    return {"feasible": True, "tokens": tokens, "chars": math.floor(tokens * chars_per_token)}


def build_long_read_prompt(nonce: str, chars: int) -> str:
    lines = build_prompt(nonce, chars).split("\n")
    return "\n".join([lines[0], "You are reading a long ModelDock benchmark document.", "", lines[3], "",
                      'Instruction: output only the five access code segments in order, joined by "-", and nothing else.'])


def garble_metrics(text: str) -> dict:
    length = len(text) or 1
    replacement = text.count("�") / length
    control = len(re.findall(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", text)) / length
    plain = len(re.findall(r"[A-Za-z0-9\s.,;:'\"!?()\-　-〿一-鿿＀-￯]", text)) / length
    words = re.findall(r"[a-z']+|[一-鿿]", text.lower())
    shingles = [" ".join(words[i:i + 6]) for i in range(len(words) - 5)]
    repeat = 1 - len(set(shingles)) / len(shingles) if shingles else 0.0
    unique = len(set(words)) / len(words) if words else 0.0
    return {"garbled": replacement > 0.01 or control > 0.01 or plain < 0.8,
            "repetitive": repeat > 0.6 or (len(words) > 40 and unique < 0.12), "words": len(words)}


def first_json_line(text: str):
    lines = [ln.strip() for ln in re.sub(r"```(?:json)?", "", text, flags=re.I).splitlines() if ln.strip()]
    if not lines:
        return None
    try:
        v = json.loads(lines[0])
    except ValueError:
        return None
    return v if isinstance(v, dict) else None


def evaluate_answer(nonce: str, content: str, reasoning: str) -> dict:
    exp = expected_answers(nonce)
    hidden = bool(THINK_BLOCK.search(content)) or bool(reasoning.strip())
    visible = THINK_BLOCK.sub("", content).strip()
    js = first_json_line(visible)
    format_ok = bool(js and isinstance(js.get("run"), str) and isinstance(js.get("code"), str) and "sum" in js)
    code = str(js["code"]).upper() if format_ok else visible.upper()
    parts = re.split(r"[-\s]+", code) if format_ok else []
    recalled = [(parts[i] == s if i < len(parts) else False) if format_ok else (s in code)
                for i, s in enumerate(exp["segments"])]
    if format_ok:
        try:
            sum_ok = float(js["sum"]) == exp["sum"]
        except (TypeError, ValueError):
            sum_ok = False
    else:
        sum_ok = re.search(rf"\b{exp['sum']}\b", visible) is not None
    run_echo = js["run"] == nonce if format_ok else nonce in visible
    essay = "\n".join(visible.splitlines()[1:]) if format_ok else visible
    return {"format_ok": format_ok, "run_echo": run_echo, "recalled": recalled, "sum_ok": sum_ok,
            "truncated": (not recalled[0]) and any(recalled[1:]), "hidden_thinking": hidden,
            "garble": garble_metrics(essay), "expected_code": exp["code"], "expected_sum": exp["sum"]}


def cache_state(r: dict):
    u, t = r.get("usage") or {}, r.get("timings") or {}
    vals = [num(v) for v in (u.get("cached_tokens"), u.get("cache_read_input_tokens"),
                             (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                             t.get("cache_n"), t.get("cached_n"), t.get("cache_tokens"))]
    vals = [v for v in vals if v is not None]
    if not vals:
        return "unknown", None
    c = max(vals)
    return ("unused" if c == 0 else "prefix-only" if c <= PREFIX_CACHE_TOKENS else "used"), c


def measurement(r: dict) -> dict:
    """Engine prefill and decode from `timings` (count and time from the same object only), else the client clock."""
    t, u = r.get("timings") or {}, r.get("usage") or {}

    def pair(ck, mk):
        c, m = num(t.get(ck)), num(t.get(mk))
        return (c, m) if c and m else None
    pre, dec = pair("prompt_n", "prompt_ms"), pair("predicted_n", "predicted_ms")
    state, cached = cache_state(r)
    prompt_tokens = num(u.get("prompt_tokens"))
    if prompt_tokens is None and pre:
        prompt_tokens = pre[0] + (cached or 0)
    out = num(u.get("completion_tokens"))
    if out is None:
        out = dec[0] if dec else None
    client = (out - 1) / r["delivery_s"] if out and out > 1 and r["delivery_s"] > 0 else None
    prefill = pre[0] / (pre[1] / 1000) if pre else (prompt_tokens / r["ttft_s"] if prompt_tokens and r["ttft_s"] > 0
                                                     else None)
    return {"prompt_tokens": prompt_tokens, "out_tokens": out, "prefill_tps": prefill,
            "prefill_basis": "engine" if pre else "client",
            "engine_decode_tps": dec[0] / (dec[1] / 1000) if dec else None, "client_decode_tps": client,
            "ttft_s": r["ttft_s"], "cache_state": state, "cached_tokens": cached}


def verdict(r: dict, m: dict, a: dict) -> dict:
    void, flags = [], []
    if not r["complete"]:
        void.append("incomplete-stream")
    if m["cache_state"] == "used":
        void.append("prompt-cache-used")
    if a["truncated"]:
        void.append("prompt-truncated")
    if a["garble"]["garbled"]:
        void.append("garbled-output")
    if not (m["out_tokens"] or 0) >= MIN_OUTPUT_TOKENS:
        void.append("output-too-short")
    if a["garble"]["repetitive"]:
        flags.append("repetitive-output")
    if not a["format_ok"]:
        flags.append("format-or-template-issue")
    if not a["run_echo"]:
        flags.append("run-echo-missing")
    if not all(a["recalled"]) and not a["truncated"]:
        flags.append("long-context-recall-incomplete")
    if not a["sum_ok"]:
        flags.append("table-sum-wrong")
    if a["hidden_thinking"]:
        flags.append("includes-thinking-tokens")
    e, c = m["engine_decode_tps"], m["client_decode_tps"]
    if e and c and abs(e - c) / e > 0.25:
        flags.append("engine-client-decode-mismatch")
    return {"status": "void" if void else "flagged" if flags else "valid", "void": void, "flags": flags}


def modeldock(srv: Server, runs: int, budget_s: float) -> dict:
    print(f"== ModelDock protocol v3 (local, {runs} probe run{'s' if runs > 1 else ''})")
    deadline = time.perf_counter() + budget_s
    remaining = lambda: max(1.0, deadline - time.perf_counter())   # noqa: E731
    variant = {"reasoning_effort": "none", "cache_prompt": False}

    def ask(prompt: str, max_tokens: int):
        return srv.stream({"messages": [{"role": "user", "content": prompt}], "temperature": 0,
                           "max_tokens": max_tokens, **variant}, timeout=remaining())

    if srv.ctx and srv.ctx < 16_384:
        print(f"  skipped: the probe run needs a context of at least 16,384 tokens; the server's is {srv.ctx}")
        return {"skipped": f"context {srv.ctx}"}
    warm = ask(FINGERPRINT_PROMPT, FINGERPRINT_MAX_TOKENS)
    if "error" in warm and warm.get("status") in (400, 422):   # a rejected optional field: once without them
        variant = {}
        warm = ask(FINGERPRINT_PROMPT, FINGERPRINT_MAX_TOKENS)
    if "error" in warm:
        print(f"  warm-up failed: {warm['error']}")
        return {"error": warm["error"]}
    fingerprint = THINK_BLOCK.sub("", warm["content"]).strip()
    print(f"  fingerprint ({'standard' if variant else 'minimal'} request): {fingerprint[:120]!r}")
    cw = ask(build_prompt(new_nonce(), CACHE_WARMUP_PROMPT_CHARS), CACHE_WARMUP_MAX_TOKENS)
    if "error" in cw:
        print(f"  cache warm-up failed: {cw['error']}")
        return {"error": cw["error"]}
    print("  cache warm-up done (untimed)")

    probes = []
    for i in range(runs):
        nonce = new_nonce()
        prompt = build_prompt(nonce, probe_prompt_chars(srv.ctx))
        r = ask(prompt, MAX_OUTPUT_TOKENS)
        if "error" in r:
            print(f"  probe run {i + 1} failed: {r['error']}")
            return {"error": r["error"], "probes": probes}
        m = measurement(r)
        a = evaluate_answer(nonce, r["content"], r["reasoning"])
        v = verdict(r, m, a)
        probes.append({"nonce": nonce, "prompt_chars": len(prompt), **m, "answer": a, "verdict": v,
                       "output": r["content"][:20_000]})
        dec = m["engine_decode_tps"] or m["client_decode_tps"]
        print(f"  probe run {i + 1}: {int(m['prompt_tokens'] or 0)} prompt tok, {int(m['out_tokens'] or 0)} out, "
              f"TTFT {m['ttft_s']:.2f} s, prefill {m['prefill_tps'] or 0:.0f} tok/s, decode {dec or 0:.1f} tok/s "
              f"({'engine' if m['engine_decode_tps'] else 'client'}; client {m['client_decode_tps'] or 0:.1f}), "
              f"recall {sum(a['recalled'])}/5, sum {'ok' if a['sum_ok'] else 'wrong'}, {v['status']}"
              + (f" [{', '.join(v['void'] + v['flags'])}]" if v["void"] or v["flags"] else ""))

    first = probes[0]
    cpt = first["prompt_chars"] / first["prompt_tokens"] if first["prompt_tokens"] else 4.5
    plan = long_read_plan(srv.ctx, cpt, first["prefill_tps"], min(120.0, remaining() - 30))
    long_read = {"skipped": True, "reason": plan.get("reason"), "needed": plan.get("needed")}
    for _ in range(2 if plan["feasible"] else 0):
        nonce = new_nonce()
        prompt = build_long_read_prompt(nonce, plan["chars"])
        r = ask(prompt, LONG_READ_OUTPUT_TOKENS)
        if "error" in r:
            if r.get("status") == 400 and plan["tokens"] > LONG_READ_MIN_TOKENS:   # unknown context: the 32K floor
                plan = long_read_plan(LONG_READ_MIN_TOKENS + LONG_READ_OUTPUT_TOKENS + 256, cpt, None, 120)
                continue
            long_read = {"skipped": True, "reason": r["error"]}
            break
        m = measurement(r)
        segs = expected_answers(nonce)["segments"]
        text = THINK_BLOCK.sub("", r["content"]).upper()
        first_ok, last_ok = segs[0] in text, segs[-1] in text
        void = [x for x, bad in (("incomplete-stream", not r["complete"]), ("prompt-truncated", not first_ok),
                                 ("prompt-cache-used", m["cache_state"] == "used")) if bad]
        long_read = {"skipped": False, "planned_tokens": plan["tokens"], **m, "first_needle": first_ok,
                     "last_needle": last_ok, "status": "void" if void else "valid", "void": void}
        print(f"  long read: {int(m['prompt_tokens'] or 0)} prompt tok, prefill {m['prefill_tps'] or 0:.0f} tok/s "
              f"({m['prefill_basis']}), TTFT {m['ttft_s']:.2f} s, needles first/last "
              f"{'yes' if first_ok else 'no'}/{'yes' if last_ok else 'no'}, {long_read['status']}")
        break
    if long_read["skipped"]:
        print(f"  long read not run: {long_read['reason']}"
              + (f" (needs a context of at least {long_read['needed']} tokens)" if long_read.get("needed") else ""))

    usable = [p for p in probes if p["verdict"]["status"] != "void"]
    decodes = [p["engine_decode_tps"] or p["client_decode_tps"] for p in usable]
    decodes = [d for d in decodes if d]
    decode = statistics.median(decodes) if decodes else None
    unstable = len(decodes) > 1 and (max(decodes) - min(decodes)) / max(decodes) > UNSTABLE_SPREAD
    long_ok = not long_read["skipped"] and long_read["status"] == "valid"
    prefills = [p["prefill_tps"] for p in usable if p["prefill_tps"]]
    prefill = long_read["prefill_tps"] if long_ok else (statistics.median(prefills) if prefills else None)
    status = "void" if not usable else "flagged" if unstable or any(p["verdict"]["status"] == "flagged" for p in usable) \
        else "valid"
    head = {"status": status, "decode_tps": decode, "prefill_tps": prefill,
            "prefill_basis": "long-read" if long_ok else "probe-run",
            "prefill_prompt_tokens": long_read.get("prompt_tokens") if long_ok else first["prompt_tokens"],
            "ttft_s": first["ttft_s"], "unstable": unstable}
    print(f"  result   {status}{' (decode differs by more than 15% between runs)' if unstable else ''}")
    print(f"  decode   {decode or 0:.1f} tok/s  (median of {len(decodes)} valid probe run{'s' if len(decodes) != 1 else ''})")
    print(f"  prefill  {prefill or 0:.0f} tok/s  ({head['prefill_basis']}, "
          f"{int(head['prefill_prompt_tokens'] or 0)} tokens)")
    print(f"  TTFT     {head['ttft_s']:.2f} s  (first probe run)")
    return {"protocol": "modeldock-bench/v3 (local port of runner 0.1.5)", "fingerprint": fingerprint,
            "request_variant": "standard" if variant else "minimal", "probes": probes, "long_read": long_read,
            "result": head}


# ---------------------------------------------------------------- ECC (Windows, AMD memory controller)
XPF_MCA = "8a1e1d01-42f9-4557-9c33-565e5cc3f7e8"   # WHEA_XPF_MCA_SECTION


def ecc_since(start: dt.datetime) -> dict:
    """WHEA-Logger 47 events since `start`, each decoded to channel / chip select where the record carries AMD UMC
    registers (IPID hardware id 0x96): channel = IPID[31:0] >> 20, chip select = SYND & 7 (CS0/1 DIMM 0, CS2/3 DIMM 1)."""
    if os.name != "nt":
        return {"skipped": "the ECC check reads the Windows event log"}
    ps = ("$s=[datetime]::Parse('%s'); $e=Get-WinEvent -FilterHashtable @{LogName='System';"
          "ProviderName='Microsoft-Windows-WHEA-Logger';StartTime=$s} -ErrorAction SilentlyContinue | "
          "Where-Object Id -eq 47; $p=(Get-WinEvent -FilterHashtable @{LogName='System';"
          "ProviderName='Microsoft-Windows-Kernel-WHEA';StartTime=$s} -ErrorAction SilentlyContinue | Measure-Object).Count;"
          "$r=@($e | ForEach-Object { $d=([xml]$_.ToXml()).Event.EventData.Data; [pscustomobject]@{t=$_.TimeCreated."
          "ToString('s'); raw=($d | Where-Object Name -eq 'RawData').'#text'} }); "
          "[pscustomobject]@{pages=$p; events=$r} | ConvertTo-Json -Depth 4 -Compress") % \
        start.astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], capture_output=True,
                             text=True, timeout=180).stdout.strip()
        data = json.loads(out) if out else {"pages": 0, "events": []}
    except Exception as e:
        return {"error": f"event log not read: {e}"}
    events = data.get("events") or []
    if isinstance(events, dict):
        events = [events]
    per = {}
    undecoded = 0
    for ev in events:
        loc = None
        raw = bytes.fromhex(ev.get("raw") or "")
        if raw[:4] == b"CPER":
            n = struct.unpack_from("<H", raw, 10)[0]
            for i in range(n):
                d = 128 + i * 72
                off, length = struct.unpack_from("<II", raw, d)
                if str(uuid.UUID(bytes_le=raw[d + 16:d + 32])) != XPF_MCA or length < 88:
                    continue
                s = raw[off:off + length]
                ipid, synd = struct.unpack_from("<QQ", s, 72)
                if (ipid >> 32) & 0xFFF == 0x96:
                    ch, cs = (ipid & 0xFFFFFFFF) >> 20, synd & 7
                    loc = f"channel {'AB'[ch] if ch < 2 else ch} DIMM {cs // 2} rank {cs % 2}"
        if loc is None:
            undecoded += 1
        else:
            per[loc] = per.get(loc, 0) + 1
    return {"errors": len(events), "pages_retired": int(data.get("pages") or 0), "per_dimm": per,
            "undecoded": undecoded, "times": [ev.get("t") for ev in events]}


# ---------------------------------------------------------------- main
def gpu_info() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,power.limit,memory.used,memory.total",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:
        return "nvidia-smi not available"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080", help="the running server (default %(default)s)")
    ap.add_argument("--model", help="model id (default: the first one /v1/models lists)")
    ap.add_argument("--tests", default="community,modeldock", help="community, modeldock or both (default %(default)s)")
    ap.add_argument("--runs", type=int, choices=(1, 2), default=2, help="ModelDock probe runs (default %(default)s)")
    ap.add_argument("--budget-seconds", type=float, default=600.0,
                    help="ModelDock's time budget; the long read is shortened to fit it (default %(default)s)")
    ap.add_argument("--seed", type=int, default=7, help="sampling seed of the community requests (default %(default)s)")
    ap.add_argument("--timeout", type=float, default=900.0, help="seconds per community request (default %(default)s)")
    ap.add_argument("--no-ecc", action="store_true", help="skip the ECC check")
    ap.add_argument("--out", help="also write every measurement to this JSON file")
    a = ap.parse_args()

    tests = [t.strip() for t in a.tests.split(",") if t.strip()]
    if not set(tests) <= {"community", "modeldock"}:
        ap.error("--tests takes community and/or modeldock")
    start = dt.datetime.now().astimezone()
    try:
        srv = Server(a.url, a.model, a.seed, a.timeout)
    except Exception as e:
        print(f"no server at {a.url}: {e}\nstart the engine's server first; this script does not start one")
        return 2
    print(f"start    {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"server   {srv.base}  service={srv.service}  model={srv.model}  context={srv.ctx or 'unknown'}")
    print(f"gpu      {gpu_info()}")
    result = {"start": start.isoformat(), "url": srv.base, "service": srv.service, "model": srv.model,
              "context": srv.ctx, "gpu": gpu_info()}
    if "community" in tests:
        with open(PROMPTS, encoding="utf-8") as f:
            result["community"] = community(srv, json.load(f))
    if "modeldock" in tests:
        result["modeldock"] = modeldock(srv, a.runs, a.budget_seconds)
    if not a.no_ecc:
        ecc = ecc_since(start)
        result["ecc"] = ecc
        print(f"== ECC since {start.strftime('%H:%M:%S')}")
        if "errors" in ecc:
            print(f"  corrected memory errors: {ecc['errors']}   pages retired: {ecc['pages_retired']}")
            for loc, n in sorted(ecc["per_dimm"].items()):
                print(f"    {loc}: {n}")
            if ecc["undecoded"]:
                print(f"    not decoded to a DIMM: {ecc['undecoded']}")
        else:
            print("  " + (ecc.get("skipped") or ecc.get("error", "")))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)
        print(f"written  {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
