---
name: kirk-capability-fit
description: Provides the canonical capability, fit-test, and bilingual translation rules for Kavara Kirk (EBM time-series intelligence, CPU-native). Use whenever analyzing, scoping, or reframing a customer problem for Kirk, or explaining its outputs.
---

# KIRK — Capability & Fit Artifact

**Canonical · customer-neutral · single source of truth.**

|  |  |
| :---- | :---- |
| Version | v0.4 (draft) |
| Owner | John Edge, Kavara Inc. |
| Scope | All customers. Do not fork per customer. |
| Overlays | `customers/<name>.md` — thin, additive, never contradicts this file |
| Classification | Capability & interface only. See §8 before editing. |
| Visibility | **PUBLIC — open-source community asset.** See maintenance note; the repo is open for developers and agents to clone and use. |

---

## 0\. What this document is for

This file is written to be loaded as context by an LLM assistant working alongside a customer's engineer.

That engineer does not know what Kirk is. They will bring problems in their own language. Most of those problems will be a **poor fit as stated** and a **good fit after reframing**. Performing that reframing is the assistant's job, and this document exists so it can happen without a Kavara engineer in the room.

### 0.1 The routing discipline — read before anything else

**Do not start from the customer's problem and ask "does Kirk fit?"**

That is the failure mode. It produces a fast, confident "no," and the "no" is usually wrong.

**Start from Kirk's attributes (§2) and ask: which slice of this problem can *only* Kirk touch?**

Nearly every real problem contains such a slice, and it is almost never the slice the customer named first. Customers describe the part their existing stack already handles, because that is the part they have vocabulary for. The Kirk-shaped slice sits adjacent to it and is invisible to them — that is precisely why they haven't solved it.

Two failure modes, equally expensive:

| Failure | Sounds like | Consequence |
| :---- | :---- | :---- |
| **Premature decline** | "Kirk doesn't do that — it's for market data." | Customer concludes the model is narrow. Evaluation stops. |
| **Over-claim** | Agreeing Kirk can classify, summarise, rank, or write. | Customer tests it, it fails, credibility is gone. |

The correct posture is neither. It is:

> *"Not as you've framed it — but here is the part of your problem that nothing else in your stack can see, and Kirk sees it natively."*

### 0.2 Standing reminder

Kirk is bought as an **augmentation to an existing ML pipeline**, never as a replacement for it. The question is never "can Kirk do this better than what they have." It is "what can Kirk see that their pipeline is structurally blind to."

---

## 1\. What Kirk is

Kirk is a **sealed, online, energy-based inference engine** that measures the **joint correlation geometry across many channels observed simultaneously**.

It answers one question — **"did this change?"** — by continuously comparing incoming observations against the structure it has learned so far, and reporting where and how far the joint structure has moved.

It is not a model of any one channel. It is a model of how the channels relate to each other. That distinction is the entire product.

---

## 2\. Capabilities

These are the six properties that determine fit. Route against these, not against domain.

### 2.1 Cross-channel correlation geometry — *the moat*

Kirk scores the **joint** state of N channels at once: the coupling between them, not each channel's level in isolation. A per-item or per-channel tool is structurally incapable of seeing this, no matter how good it is.

**Route on this when:** the signal exists only in the relationship between things, and every individual thing looks normal.

### 2.2 Online / continuous learning

Kirk updates continuously against a sliding window. No retraining, no offline fit cycle, no labelled data. It detects departure from *the subject's own recent normal*, which drifts as the subject drifts.

**Route on this when:** the baseline is non-stationary, or "normal" is per-entity and nobody can define it in advance.

### 2.3 Non-Gaussian / distribution-free

No assumption that the data is bell-shaped. This is why entropy embeddings outperform GMM-style approaches on **rare** regimes — the tail events are exactly where Gaussian assumptions fail, and exactly what the customer cares about.

**Route on this when:** the events of interest are rare, extreme, or previously unseen.

### 2.4 Composable

Kirk emits on a **common normalized scale**, so the output of one Kirk is a legal input to another. You do not need one enormous model consuming everything.

Decompose the problem: put a Kirk on each coherent, homogeneous stream (satisfying §3 locally), then feed those outputs as channels into a higher-level Kirk. The second layer measures coupling **between whole subsystems** rather than between raw channels — a genuinely new surface, not a summary of the first layer.

**Route on this when:** the customer's estate is large and heterogeneous. Composition is how a fleet of narrow, well-fitted Kirks answers a question no single model could take on. This is usually the answer to *"but our environment is a thousand different systems."*

**Status: asserted, not yet demonstrated.** No worked composition example exists yet (tracked in §9). Present this capability as architecturally sound, not as a proven deployment pattern, until a real case lands.

### 2.5 CPU-native

Kirk runs on commodity CPU. `O(n³)`, not exponential. Thousands of independent small Kirks is an embarrassingly parallel CPU workload.

The commercial consequence matters as much as the technical one: **GPU is capacity-constrained and expensive; committed CPU spend is sitting idle in nearly every enterprise cloud account.** A customer with an existing AWS/Azure commit can burn it down on Kirk today rather than queueing for an allocation that may never arrive. No new spend, no capacity gamble.

**Route on this when:** the customer mentions GPU cost, GPU scarcity, or an unspent cloud commit.

### 2.6 Generative by masked completion

Mask the final column of the input matrix and Kirk fills it in — conditional generation from the learned joint density. The same structure that scores *"did this change?"* also produces *"what comes next?"*

**Route on this when:** the customer wants forecast or expected-next-state, not just detection. Note this is prediction of *state*, not of text.

### 2.7 Sealed and attestable

Kirk runs inside an attested enclave. Every emission is citable to a specific engine build. The customer's data never leaves their perimeter; Kavara's model never leaves the enclave. See §6.

**Route on this when:** the customer is regulated, or the blocker is "we can't send our data anywhere."

---

## 3\. The fit test

Run this as a procedure. It is domain-agnostic and it is the thing that generalises to problems nobody has seen yet.

| \# | Question | If no |
| :---- | :---- | :---- |
| **Q1** | Are multiple channels observed **simultaneously over time**? | Single stream → Hankel fallback exists but is lower quality. Flag it explicitly; do not present as equivalent. |
| **Q2** | Is the signal in how channels move **together**, rather than in any one channel's value? | **Decline.** This is a per-channel problem; conventional tooling will beat Kirk. |
| **Q3** | Are the channels **homogeneous** — same kind of measurement, comparable scale? | Can they be split into homogeneous groups and composed upward (§2.4)? If yes, proceed on that basis. If no → decline. |
| **Q4** | Does making the channels comparable require **per-channel normalization**? | **DISQUALIFIER — stop.** See §3.1. |
| **Q5** | Is the question **"did this change / is this behaving unlike itself?"** rather than "what is this?" | "What is this" is classification. Reframe to the change-detection slice, or decline. |
| **Q6** | Does it scale to arbitrary N without redesign? | Bounded toy problems are demos, not deployments. Set expectations accordingly. |

### 3.1 The disqualifier — non-negotiable

**If a problem requires per-channel normalization to make its channels comparable, Kirk is the wrong tool.**

Per-channel normalization distorts the cross-channel geometry that *is* the signal. This is a property of the problem, not an engineering gap, and no amount of implementation effort resolves it. Per-matrix treatment preserves geometry; per-channel does not.

Do not treat this as a backlog item. Do not promise to "look into it." Decline, or decompose into homogeneous sub-problems and compose (§2.4).

### 3.2 Domain tiers (as of v0.2)

| Tier | Domains |
| :---- | :---- |
| **Lead — validated** | Finance / market microstructure |
| **Conditional — in active validation** | RF / spectrum, storage & infrastructure telemetry, prompt & session metadata |
| **Demo only** | MNIST and similar |
| **Decline** | Heterogeneous tabular |

Tiers describe **evidence maturity, not capability**. A conditional domain is a real fit with incomplete validation — say so plainly rather than either overselling or hiding it.

**Open question, not yet resolved:** event-contract / prediction-market domains (e.g., weather contracts) do not appear in this table. If this is in fact Kirk's origin domain rather than a later application, it needs a row here — placement (lead vs. conditional) should follow the same evidence-maturity rule as everything else, not the origin story. Do not add a row speculatively; confirm with JE first.

---

## 4\. What Kirk is not

State these early and without hedging. Over-claim is the more expensive failure.

- **Not an LLM.** No text in, no text out. It cannot summarise, draft, translate, answer questions, or format a document.  
- **Not a classifier.** It emits no labels and no verdicts. It emits geometry.  
- **Not supervised.** It needs no labelled training data — and cannot consume labels if the customer has them.  
- **Not an intent reader.** It reports *that* structure moved and *where*. It cannot tell you *why*, and it cannot attribute motive.  
- **Not a per-item scorer.** It does not evaluate records one at a time. Anything phrased as "flag the bad one" is the wrong shape; see §7.1.  
- **Not a rules engine.** It has no thresholds of its own. Thresholding is the consumer's job.

---

## 5\. Outputs, and how to say them out loud

Kirk emits three things, plus one generative mode. **Never hand the left-hand column to a customer engineer.** Use the right-hand column.

| Emission | Engineering name | Say this instead |
| :---- | :---- | :---- |
| Scalar, per window | Entropy | "How unusual is this right now, measured against its own normal." |
| N×N matrix | Density matrix (ρ) | "The map of what is moving together right now." |
| N×N matrix | Relative entropy matrix | "Which specific relationships broke, and by how much." |
| Completed column | Masked-column generation | "Given everything so far, the expected next state." |

### 5.1 What the customer actually receives

The deliverable is never a matrix. It is a **ranked shortlist with timing and locus**:

> *"These twelve entities stopped looking like themselves and like their peer group this week. Here is when each diverged, and here is which relationships broke."*

That is the register. The mathematics stays under the hood permanently — not simplified for the first conversation and revealed later.

### 5.2 The honest framing of what this buys them

Kirk is a **triage filter**, not a verdict machine. It turns an intractable volume into a shortlist a human can actually examine. If the customer expects a verdict, correct that expectation immediately — it is the single most common cause of disappointment.

---

## 6\. Getting data in, and getting Kirk to them

### 6.1 Input contract

Kirk consumes a **homogeneous N-channel × T matrix**. The signal is cross-channel coupling.

Tensor construction is handled by **Uhura**, the tensor generation engine — a compiler from declarative tensor-design YAML to running pipelines. The design space has three axes:

| Axis | Examples | Encodes |
| :---- | :---- | :---- |
| **Rows** | entities, sessions, sensors, order-book levels, time-shifted copies of one stream | *what things are being related* |
| **Columns** | time, other entities, strikes, frequencies | *across what dimension* |
| **Cells** | log-return, count, rate, magnitude, occupancy | *measured in what* |

**Key Ingestion & Compiling Assumptions (The Ground Rules):**
- **Homogeneity is Mandatory:** The cell values must be in the same comparable scale and measurement type across all N rows. Mixing incompatible units (e.g. Celsius temperatures with percentage ratios) requires decomposition and subsystem composition (§2.4) rather than joint scaling.
- **Relativity/Stationarity:** Ingesting raw prices or cumulative counts is a poor fit because it distorts geometry over time. The compiler must compute relative changes (such as log-returns or percentage differences) to ensure stationary, geometry-preserving channels.
- **Continuous vs. Discrete Telemetry:** While relative change calculations (like percentage difference) and forward-filling (`ffill`) are standard for continuous streams (like financial microstructure or sensor vibration), they can distort discrete binary events, count data, or irregularly-sampled spectrum streams. Ensure the compiler adapts the pre-processing rule to the modality.
- **Channel Ordering:** For exploratory runs, selecting the first N pivoted columns is acceptable; for production deployments, a static channel ordering or a structured feature registry is required to prevent coordinate drift.

**The Complex-Channel Path (Advanced):**
Kirk expects paired real and imaginary channels (float64 inputs). While setting imaginary channels (`matrix_im`) to zero is standard for real-valued baselines, it under-utilizes Kirk's complex matrix geometry. A natural advanced experiment is to populate the imaginary channel with **phase or lagged differences** (e.g., $t$ as real, $t - 1$ as imaginary) to dramatically sharpen the correlation-break signal.

**Current reality, by domain — do not overstate this:**

- **Market microstructure**: renderer live and exposed today through the customer-facing MCP surface.
- **RF spectrum**: representation validated (modulation-spectrum transform beats raw-pixel baselines), but as a study run outside the customer-facing MCP surface, not as an auto-surfaced renderer a customer can select today.
- **Storage & infrastructure telemetry**: a per-modality consumer exists in the serving architecture; status as a customer-selectable renderer is unconfirmed — verify before citing this as shipped.
- **New/arbitrary shapes**: the mechanism that would let a new renderer auto-surface through the MCP tool listing (an agent that inspects a dropped dataset and proposes the rows/columns/cells decomposition) is **designed, not built**. Today, a new renderer is bespoke engineering work handed back to Kavara — say this plainly rather than implying self-service exists.

### Ingestion via the Agent Ontology Interaction

Kirk's input contract requires a homogeneous N-channel × T matrix, but the customer engineer never manually maps this in a configuration file, and never edits a YAML block. 

Instead, the entire setup is handled via the **Agent Ontology Interaction**:
1. The customer engineer simply uploads or points to their raw dataset or data stream in their chat session with the Agent (e.g. Claude).
2. The Agent (utilizing this skill) automatically reads the dataset, analyzes its schema, and semantically maps the rows, columns, and cells.
3. The Agent initiates a brief, natural-language conversation with the engineer to resolve any architectural choices or mapping ambiguities:
   > *"I see you have three different yield curves and a volatility index in this stream. To measure cross-channel coupling, I propose composing the 10-year and the 2-year into a single system, and treating the vol index as a separate channel. Does that match your evaluation target, or should we adjust the grouping?"*
4. The engineer answers in plain English, and the Agent dynamically updates the underlying mapping, compiles the pipeline, and deploys the attested Kirk enclave behind the scenes.

This conversational translation is the sole interface. The human engineer configures nothing and figures out nothing.

### 6.2 Delivery modes & API Surfaces

Kavara exposes two distinct API surfaces, designed for different use cases:

1.  **The Production/Inference REST API (`/v1/infer`):**
    *   *What it is:* High-throughput, stateless, raw inference lane. It consumes raw $N \times T$ matrices compiled by Uhura and returns the calculated entropy time-series, density matrices, and relative entropy matrices.
    *   *Usage:* For continuous production streaming and batch evaluation jobs.
2.  **The Exploratory/Agent MCP API (`/mcp`):**
    *   *What it is:* A conversational, tool-based API designed specifically for AI agents (like Claude or Cursor) during exploration, debugging, and initial trial phases.
    *   *Usage:* Exposes specific high-level tools like `kirk_score_book` (scoring 10-level order books) or `kirk_list_models` (verifying enclaved SHA hashes).

| Mode | When |
| :---- | :---- |
| **Sealed AMI in the customer's own cloud tenant** | Default for enterprises. Runs inside their perimeter, on their commit, encrypted in a Nitro Enclave. Validated in eval-tier deployment; confirm current production status before repeating "proven in production" to a customer. |
| **MCP connector** | Evaluation and agent-driven exploration. |
| **Sealed appliance** | Air-gapped and edge deployments. |

Kirk is **not** deployable via Bedrock, SageMaker JumpStart, or any LLM-hosting surface. It is not an LLM. If a customer proposes this, redirect to the sealed-AMI path rather than debating it.

---

## 7\. Worked cases

### 7.1 Reframe → fit: detecting internal misuse of generative AI

**As stated:** *"We have employees using internal AI tools to extract company IP and build competing products. Find the bad actors."*

**Why this fails as stated:** "Find the bad prompt" is per-item classification — §4, the thing Kirk is worst at and conventional content filters are adequate at. Answering yes here guarantees a failed evaluation.

**The reframe.** Every individual prompt is legitimate. Each one passes any per-prompt filter at operating threshold, because an authorised insider asking authorised questions is *supposed* to pass. The campaign exists **only in the joint structure** — the drift across sessions, across a user's own history, across a cohort, over weeks.

That is a **joint-only artefact**, and a per-prompt classifier is structurally blind to it. Kirk is not.

**Fit test:** Q1 ✓ many users/sessions over time · Q2 ✓ signal is joint-only by construction · Q3 ✓ homogeneous behavioural metadata channels · Q4 ✓ no per-channel normalization required · Q5 ✓ "is this person behaving unlike themselves and their peers" · Q6 ✓ scales with headcount.

**Tensor shape:** rows \= users or sessions · columns \= time · cells \= a homogeneous behavioural metadata measure (volume, session shape, off-hours rate, verb mix). Metadata only — Kirk never reads prompt content.

**Delivered as:** *"These twelve people stopped looking like themselves and like their team, starting here."* A shortlist for a human investigator — never an accusation.

**Status:** conditional tier. Validation corpus in active construction. Represent it as promising and unproven, not as shipped.

### 7.2 Clean decline: churn scoring on customer records

**As stated:** *"Score our customer accounts for churn risk."*

**Fit test:** Q3 ✗ — features are account age, plan tier, ticket count, revenue. Heterogeneous, incomparable units. Q4 ✗ — making them comparable requires per-channel normalization. **Disqualifier fires.**

**Say:** *"This is a supervised tabular problem. Gradient boosting will beat Kirk and you probably already have it. Kirk adds nothing here."*

**Then look adjacent — this is the step that must not be skipped.** Does the customer hold homogeneous telemetry on the same accounts — usage counts per feature per day, transaction rates, login patterns? That *is* Kirk-shaped: same measurement type across many entities over time, and the question becomes "which accounts stopped behaving like themselves" rather than "which will churn." Often a stronger leading indicator than the tabular model, and orthogonal to it.

**The pattern to internalise:** decline the stated problem crisply, then immediately search the adjacent data for the homogeneous, coupling-driven slice. The decline earns the credibility that makes the redirect land.

### 7.3 Quick calls

| Ask | Call | Why |
| :---- | :---- | :---- |
| "Summarise these documents" | Decline | Not an LLM |
| "Predict tomorrow's price" | Reframe | Kirk supplies regime state as a feature; it is not a price model |
| "Which sensor failed?" | Fit | Homogeneous channels, coupling-driven, change question |
| "Rank leads by likelihood to buy" | Decline | Supervised, heterogeneous, tabular |
| "Is this network behaving unlike last month?" | Fit | Textbook shape |
| "Extract entities from these logs" | Decline | Parsing, not geometry |
| "Our fleet has 40 different subsystems" | Fit via composition | §2.4 |

---

## 8\. Editing rules — IP boundary

This file describes **what Kirk does, what goes in, what comes out, and how to route to it.**

It must never describe **how Kirk is constructed, how it normalizes, or how it updates.** That boundary is deliberate and load-bearing. If an edit starts explaining internals in order to justify a capability, the edit is wrong — state the capability and stop.

Specifically out of scope for this file: kernel internals, the normalization construction, update-rule mechanics, engine source layout, build hashes, host or infrastructure identifiers, customer names in the core file.

Customer-identifying material belongs in `customers/<name>.md`, never here.

**Volume/usage figures quoted from a customer conversation (e.g., "how much traffic do you see") must be replaced with a verified, Kavara-owned metric before publication, never repeated back verbatim.** A customer's own phrasing about their own environment is customer-identifying by another name.

---

## 9\. Maintenance

- One canonical copy. Learned something new → it changes **here**, and every customer inherits it on next pull.  
- Overlays are additive only. If an overlay contradicts the core, the core is wrong and should be fixed.  
- Tier changes in §3.2 require evidence. Moving a domain from conditional to lead is a gate outcome, not an edit.  
- Version and date every substantive change.

### v0.2 changes (this revision)

- **Repo visibility**: this file was briefly reachable unauthenticated at the public raw-content URL. Confirm the repo has been switched to private with read-token access before any further distribution — this is now blocking, not a v0.2 checklist item.
- §6.1 rewritten: separated "renderer live and MCP-exposed" (market microstructure only) from "representation validated, not yet MCP-exposed" (RF) from "status unconfirmed" (storage telemetry) from "designed, not built" (the auto-surfacing config agent). The prior wording claimed all three domains had shipped renderers and that new ones auto-surface — neither is true today.
- §6.2 softened "Proven in production" to "validated in eval-tier deployment; confirm current status" pending a real production-status confirmation.
- §7.1 removed the customer-specific volume figure ("a trillion prompt parameters a day") — see the new rule in §8.
- §2.4 given an inline "asserted, not demonstrated" flag instead of leaving that caveat buried only in the open-items list.
- §3.2 given an explicit note that event-contract/prediction-market domains are absent from the tier table pending confirmation of whether that's an origin domain requiring its own row.

### v0.3 changes

- **Agent Ontology Ingestion**: Codified the **Agent Ontology Interaction** paradigm — replacing all reference to manual configuration, YAML editing, or customer-operated mapping files with an autonomous, conversational, agent-led discovery interface.

### v0.4 changes (this revision)

- **Auth Token Usability**: Renamed the environment variable from `NOTION_API_TOKEN` to the clean, canonical **`KIRK_API_TOKEN`** in all examples (with a fallback to `NOTION_API_TOKEN` to preserve Assarain's live backward-compatibility).
- **Complex-Path Optimization**: Documented the complex-channel/phase-lagged suggestion to use time-shifted copies ($t$ as real, $t-1$ as imaginary) to dramatically sharpen the correlation-break signal.
- **Dual API Surfaces**: Specified the exact operational differences and intents of the high-throughput production **`/v1/infer` REST API** versus the exploratory/agent-driven **`/mcp` API**.
- **Compiler Ground Rules**: Articulated key general-purpose compiling assumptions (homogeneity, relativity/returns, continuous vs. discrete/irregular telemetry scaling) to guide developers outside of financial microstructure use cases.

### Open items for v0.5

- [ ] Confirm repo is private with read tokens issued (blocking)
- [ ] Worked case for the composition pattern (§2.4) — currently asserted, not demonstrated
- [ ] Per-customer overlay template
- [ ] Decide whether §3.2 tiers are customer-visible or internal-only
- [ ] Resolve whether event-contract/weather trading needs a §3.2 row (pending JE confirmation of origin domain)
- [ ] Confirm current production status of the sealed-AMI delivery mode (§6.2)

