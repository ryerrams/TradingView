# Log Narrator — Findings & Handoff

Built and tested 2026-09-09. Artifact: `/Users/ryerrams/Documents/Trading/log_narrator.py` (526 lines).
Pipeline: `tail → parse → diff vs last spoken state → (only if material) local LLM → macOS say`.

Everything below was measured, not assumed. Paths and numbers are real.

---

## 1. The dominant finding: the log is ~99% repetition

Measured on the real MBot logs (`/Users/ryerrams/MBot/logs/*.log`):

| File | Analysis lines | Distinct payloads |
|---|---|---|
| `atmbotv2_institutional_MNQU6_paper_2026-08-17.log` | 361 | **1** |
| `atmbotv2_institutional_MNQU6_paper_2026-08-16.log` | 257 | **3** |

The bot emits a heartbeat + analysis line every ~60s regardless of whether anything happened.
**A "read each line and narrate it" design speaks the same sentence 361 times per session.**

The entire engineering problem is change detection, not prompting. After diffing:

| File | Log lines | Utterances |
|---|---|---|
| 2026-08-17 | 802 | **26** |
| 2026-08-16 | 620 | **11** |

≈ one utterance every 15 minutes. This is also what makes a local model viable — the LLM is
invoked a few dozen times a day, not 800 times.

**Transferable:** any bot/service log with a periodic heartbeat has this shape. Budget the
effort into diffing state, not into the model.

---

## 2. macOS audio: named voices are the robotic ones

`say -v Samantha` sounds synthetic. The machine has **184 voices installed but only 4 compact
English ones** (Samantha, Daniel, Karen, Moira). There are **no Premium/Enhanced/Siri entries in
`say -v '?'` and there never will be — the good voices are not addressable by name.**

**Fix: omit `-v` entirely.** Bare `say -r 150 "text"` uses the macOS **System Voice** from
*System Settings → Accessibility → Spoken Content → System Voice*, which is set to a Siri voice.
A/B confirmed by the user: the no-`-v` version was clearly better.

This was already solved in `~/TwitterSpeak/trade_processor.py` (`use_system_voice: true`, and the
comment at ~line 72 explains exactly this). Other settings that matter as much as the voice:

- `rate 150`, not the 175/185 default — slower reads as calmer, less synthetic
- Serialize speech through `queue.Queue` + a worker thread so utterances never overlap
- `gap_seconds` pause between utterances
- Idle gate: `ioreg -c IOHIDSystem` → `HIDIdleTime` (nanoseconds) to stay quiet when away

**Transferable:** applies to any macOS TTS work. Reuse `TwitterSpeak/trade_processor.py` directly
rather than re-deriving.

---

## 3. The local LLM is not yet safe for this job

The LLM's only role is turning a structured event into a spoken sentence. On `qwen2.5vl:7b` it is
fluent and **factually wrong in ways that matter**. Same events, template path vs LLM path:

| Log line | Template (correct) | LLM (wrong) |
|---|---|---|
| `OPTION C EXIT TRIGGERED … Locked PnL **+64.4pt**` | verbatim, accurate | "MNQU6 shorted, **down by sixty points**" |
| `OPTION C EXIT TRIGGERED … Locked PnL **+59.7pt**` | verbatim, accurate | "**It's time to short** MNQU6, trend is moving in our favor" |
| `Winning trade realized. Ladder reset to Level 0` | verbatim, accurate | "We exited at 30268, bracket exit" (narrated the *previous* event) |

Two failure classes, both serious for a tool that narrates money out loud:
1. **Sign inversion** — a +64 point profit spoken as a 60 point loss
2. **Fabricated advice** — an exit turned into an entry recommendation

The deterministic templates were accurate on every event; they just read as jargon.
**Fluent-and-wrong is worse than clunky-and-right here.**

Where the LLM genuinely earns its place: the free-form `bot_event` lines
(`OPTION C EXIT TRIGGERED (1m WMA30 Cross) @ 30240.75 | Peak +69.7pt | Locked PnL +69.2pt`).
Templates can only strip emoji and read the jargon aloud. That is ~2/3 of all events, so the
value is real — but only if accuracy is fixed.

---

## 4. OPEN BUG: qwen3 returns nothing on `/api/generate`

`qwen3:32b` returned `None` on every call. Root cause found, not yet fixed:

```
num_predict=60   → 6.6s  done_reason=length  response=''
num_predict=200  → 21.1s done_reason=length  response=''
```

The model **ignores the `/no_think` prefix on `/api/generate`** and spends the entire token budget
on a conversational preamble ("It looks like you're referencing a data point…"), so the budget is
exhausted before any answer. Not a model limitation — a request-shape bug.

**Next step:** call `/api/chat` with a system message instead of raw `/api/generate`, pass
`think: false`, and raise `num_predict`. Then re-run the three failing cases from §3 to see whether
model size fixes the sign-inversion. If it does → 32B becomes default (latency is irrelevant at one
event per 15 min). If it doesn't → default to `--no-llm` and invest in better deterministic phrasing.

Note: the `/no_think` recipe in the `hermes-trading-agent` memory was learned via the Hermes CLI,
which uses the chat endpoint. It does not transfer to raw `/api/generate`.

---

## 5. Bugs found and fixed during testing

1. **State clobbering** — `stale` was parsed from every line type, but only the 🔍 analysis line
   carries `STALE DATA`. The flag flipped true/false on alternating lines and fired ~770 spurious
   "feed stale / feed live" pairs per session (769 utterances from 802 lines). Fix: fields only
   merge when the line type actually carries them; `None` never overwrites prior state.
2. **Duplicate events** — a position change lands on both the 📈 and 📊 line, so every entry was
   announced twice. Fix: dedupe on `(event, log_timestamp)` plus recent-text dedupe.
3. **Few-shot leakage** — prompt examples used MNQ-shaped prices, and the model reproduced one
   verbatim for an unrelated event (said *"just went short one at thirty thousand three ten"* for
   an ETH scalp exit). **Saying the wrong thing aloud is worse than saying nothing.** Fix: examples
   use a fake instrument and a non-colliding price range, plus an explicit no-copy instruction.
   Verified zero leaks across both logs.
4. **Run-on truncation** — passing the whole state dict made the model enumerate fields until it hit
   the token cap mid-word. Fix: per-event relevance filter + keep only the first sentence.

---

## 6. Reusable for other projects

- **Change-detection-before-LLM** is the general pattern for narrating any high-frequency log.
- **`~/TwitterSpeak/trade_processor.py`** is the reference macOS TTS implementation — system voice,
  serialized queue, active-hours gate, idle gate. Copy it; don't re-derive.
- **Templates as the safety floor**: for anything where a wrong number has consequences, keep a
  deterministic path and treat the LLM as an enhancement that must be proven, not assumed.
- The parser in `log_narrator.py` is token-based (`grab()` + targeted regex per field) rather than
  one full-line regex, so a format drift degrades gracefully instead of silently emitting nothing.

## 7. State of the code

Working and tested: tailing with day-rollover and truncation handling, `--replay` for testing against
finished logs, `--dry-run`, `--no-llm`, change detection, dedupe, system-voice audio, idle gating.
Untested: live tailing against a running bot (only replay was exercised).

Default model is `qwen2.5vl:7b`. Given §3, consider `--no-llm` until §4 is resolved.
