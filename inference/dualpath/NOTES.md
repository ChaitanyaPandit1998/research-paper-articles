# DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference

**Paper:** https://arxiv.org/abs/2602.21548
**Authors:** Yongtong Wu, Shaoyuan Chen, Yinmin Zhong, Rilin Huang, Yixuan Tan, Wentao Zhang, Liyue Zhang, Shangyan Zhou, Yuxuan Liu, Shunfeng Zhou, Mingxing Zhang, Xin Jin, Panpan Huang
**Submitted:** February 25–26, 2026

---

## The Problem

Modern LLMs are increasingly deployed as **agents** — systems that reason across multiple turns, call external tools, process long outputs, and maintain large conversation histories. A single agentic task might span 10+ turns with 50K–100K tokens of accumulated context.

**Disaggregated inference** (splitting prefill and decode onto separate machines) is the standard architecture for serving LLMs at scale. In this setup:

- **Prefill engines** read the full context, compute attention over it, and produce the KV-Cache.
- **Decode engines** auto-regressively generate tokens one step at a time.

For agentic workloads, the KV-Cache from previous turns is stored on disk (SSD/NVMe) and loaded back at the start of each new turn rather than recomputed from scratch.

**The bottleneck:** All KV-Cache loads go from storage → prefill engine. The prefill engine's storage NIC gets saturated under long-context agentic workloads. Meanwhile, the decode engine's storage NIC sits completely idle. The system is storage-bandwidth-bound, not compute-bound.

---

## Root Cause Analysis

```
Traditional disaggregated inference:

  [SSD/NVMe Storage]
        |
        | (storage NIC — SATURATED)
        ↓
  [Prefill Engine]  ——RDMA——→  [Decode Engine]
                                  (storage NIC — IDLE)
```

The asymmetry exists because:
1. KV-Cache prefetching is prefill's job — so all storage traffic flows to prefill machines
2. Prefill NICs serve dual duty: KV-Cache I/O + model weight communication
3. At long context lengths (agentic tasks), KV-Cache I/O dominates and saturates the NIC
4. Decode machines have storage NICs connected to the same SSD tier but receive zero KV-Cache traffic

This is a **resource underutilisation problem**, not a hardware capacity problem.

---

## DualPath: The Solution

DualPath introduces a second route for KV-Cache loading:

```
Path 1 (traditional):
  Storage → Prefill NIC → Prefill Engine

Path 2 (new):
  Storage → Decode NIC → Decode Engine → (RDMA compute network) → Prefill Engine
```

By splitting KV-Cache loads across both NICs, DualPath distributes the storage bandwidth pressure across two machines. The decode engine acts as a relay — it receives KV-Cache from storage and forwards it to the prefill engine via the internal RDMA compute network.

### Why the second hop is cheap

The RDMA compute network (connecting prefill and decode engines) has:
- High bandwidth — designed to carry KV-Cache prefill-to-decode transfers anyway
- Spare capacity when prefill is busy doing storage I/O

So the extra internal hop in Path 2 adds minimal latency while nearly doubling available storage NIC bandwidth.

---

## Global Scheduler

A **global scheduler** dynamically picks between Path 1 and Path 2 per request based on:

- Current storage NIC utilisation on prefill engines
- Available storage NIC capacity on decode engines
- Current load on the RDMA compute network
- Latency SLO requirements for the request

The scheduler ensures Path 2 traffic is throttled to avoid interfering with latency-sensitive model weight communications on the compute network. This is critical — model weight transfers are on the critical path of token generation; saturating the compute network with KV-Cache relay traffic would introduce tail latency.

---

## Key Results

| Scenario | Throughput Improvement |
|---|---|
| Offline inference (throughput-optimised) | Up to **1.87×** |
| Online serving (with SLO guarantees) | Up to **1.96×** |

Nearly **2× throughput** on the same hardware, with no SLO violations in the online setting.

---

## Core Insight

> In agentic LLM inference, the bottleneck shifts from compute to storage bandwidth. Disaggregated architectures have idle storage NICs on decode machines that are never used for KV-Cache loading. DualPath exploits this asymmetry by routing some KV-Cache loads through decode engines, effectively doubling available storage bandwidth without any new hardware.

The broader lesson: **optimisation targets must evolve with workload patterns**. Standard LLM inference is compute-bound; agentic LLM inference is I/O-bound. Infrastructure assumptions that held for chatbots break down for long-running agent tasks.

---

## Architecture Summary

```
                        ┌──────────────────────────────┐
                        │        Global Scheduler        │
                        │  (monitors NIC utilisation,    │
                        │   routes per-request)          │
                        └────────┬──────────┬───────────┘
                                 │          │
                         Path 1  │          │  Path 2
                                 ▼          ▼
  ┌──────────┐   NIC    ┌──────────────┐  ┌──────────────┐   NIC    ┌──────────┐
  │  Storage │─────────▶│    Prefill   │  │    Decode    │◀─────────│  Storage │
  │  (SSD)   │          │    Engine    │  │    Engine    │          │  (SSD)   │
  └──────────┘          └──────┬───────┘  └──────┬───────┘          └──────────┘
                               │   RDMA           │
                               │◀─────────────────┘  (relay: decode forwards KV-Cache
                               │                       to prefill via compute network)
                               ▼
                          Token generation
```

---

## DualPath Mechanism — Explained as a Story

### The District

Picture a manufacturing district with two types of factories side by side, connected by roads and internal conveyor belts.

**Prefill Factories** are the planners. When a new job arrives — say, "here are 80,000 words of context, now process them" — the prefill factory reads all of it, builds a detailed work plan (the KV-Cache), and hands it to the next building.

**Decode Factories** are the assembly lines. They take the plan and produce the output, one word at a time. Precise, methodical, token by token.

Between every pair of factories is a **fast internal conveyor belt** — a high-bandwidth RDMA link. The prefill factory uses it to ship its work plan to the decode factory when a new job starts.

Behind both factories, across the street, is the **Central Warehouse** — NVMe SSD storage. It stores all the KV-Caches from previous turns of every ongoing conversation. Each factory has exactly one **loading dock** facing the warehouse — a storage NIC.

---

### The Old Way: One Loading Dock Does Everything

Before DualPath, the rule was simple: all warehouse deliveries go to the Prefill Factory's loading dock. The prefill factory needs the KV-Cache, so the warehouse trucks always drove to the prefill dock.

For short jobs, this was fine. A small box, quick delivery, no queue.

But agentic tasks changed everything.

---

### The Problem: Rush Hour at One Dock

Imagine it's peak hour. A hundred long-running agent jobs are all mid-conversation — coding agents, research agents, document review agents. Each on Turn 8, Turn 12, Turn 15. Each turn, the system needs to reload their KV-Cache from the warehouse.

A hundred warehouse trucks all queue up at the **Prefill Factory's loading dock**. One gate. One lane. Trucks backed up around the block.

Meanwhile, across the narrow alley, the **Decode Factory's loading dock** is completely empty. Its gate is open. Its dock workers are playing cards. Not a single truck in sight.

The Decode Factory has a loading dock connected to the same warehouse. But every driver has been told the same thing: *"Deliveries go to the Prefill dock. That's the rule."*

The system grinds — not because there's not enough cargo, not because the factories can't process it, but because one loading dock is a chokepoint while an identical one sits idle twenty metres away.

---

### The DualPath Idea: Open the Second Dock

A logistics engineer watches the queue at the Prefill dock and the empty Decode dock and has a thought: *"What if we let some trucks deliver to the Decode dock instead?"*

The obvious objection: the Decode Factory doesn't need the KV-Cache. Delivering to the wrong factory doesn't help.

The engineer smiles. *"The factories are connected by that conveyor belt, aren't they?"*

Yes — the same internal conveyor belt that normally carries work plans from Prefill to Decode. It runs the other direction too. And right now, when the Prefill dock is jammed, the conveyor belt is sitting at maybe 30% capacity.

**Path 1 (traditional):**
> Warehouse truck → Prefill dock → Prefill factory uses the cargo directly

**Path 2 (new):**
> Warehouse truck → Decode dock → Decode factory receives cargo → ships it down the conveyor belt → Prefill factory receives it

Path 2 has an extra step — but every step in Path 2 is happening on **currently idle infrastructure**. The Decode dock that was sitting empty, and the conveyor belt running at 30%. The Prefill factory gets its KV-Cache either way.

---

### How the Scheduler Decides

You can't randomly send all trucks to the Decode dock — that would just move the jam. So DualPath uses a **Traffic Controller** (the global scheduler) who watches everything in real time and asks four questions before assigning each truck:

1. **How backed up is the Prefill dock right now?** No queue → Path 1 is faster, use it.
2. **How much spare capacity does the Decode dock have?** If it's already filling up, sending more there shifts the jam.
3. **How busy is the internal conveyor belt?** It also carries urgent work plans (model weight communications) that can't wait. If it's near capacity, Path 2 is throttled.
4. **How urgent is this delivery?** Tight-deadline jobs get Path 1 — predictable latency over optimised throughput.

No truck picks its own route. Central coordination prevents the solution from creating new jams.

---

### A Concrete Evening at the Dock

**Without DualPath:**
- 120 trucks queue at the Prefill dock
- Average wait: 4 minutes per truck
- Decode dock: empty
- Throughput: 200 jobs/hour

**With DualPath:**
- Queue of 8 trucks at Prefill dock → controller sends next 5 to Decode via Path 2
- Decode dock receives cargo, loads onto conveyor belt → Prefill gets it ~40 seconds later than direct
- Prefill dock clears faster; next batch split again across both docks
- Conveyor belt runs at 65% — busy but not saturated, urgent work plans still get through

**Result: 390 jobs/hour — 1.96× improvement. Same buildings, same trucks, same conveyor belt. Just better routing.**

---

### Why the Extra Hop Is Acceptable

The bottleneck was never the conveyor belt — it was the single loading dock. Once two docks work in parallel, the 40-second relay via the conveyor belt is a small price compared to the 4-minute queue in the old system.

Same reason a highway with two toll booths handles twice the traffic as one, even if one booth requires a slight detour. The detour costs seconds; the queue costs minutes.

---

### The Moral of the Mechanism

DualPath doesn't add new roads, widen the conveyor belt, or build a bigger warehouse. It notices a road nobody was using — the one to the Decode factory's loading dock — and builds a Traffic Controller smart enough to use both roads without creating new jams.

> **Parallel loading docks + intelligent routing + throttled relay = double the effective storage bandwidth at zero hardware cost.**

---

## Use Cases and Real-World Relevance

### Where This Problem Actually Appears

DualPath targets **production agentic inference at scale** — specifically clusters running disaggregated prefill/decode architectures with KV-Cache offloading to NVMe. The paper evaluates on "production agentic workloads" on an in-house inference system (the authors' affiliated organisation), validated with DeepSeek-V3, Qwen, and similar large models.

The storage bandwidth bottleneck becomes meaningful at:
- **Long context lengths:** 32K+ tokens per turn (common in code agents, document agents)
- **High concurrency:** many agentic sessions running simultaneously, all triggering KV-Cache loads
- **Multi-turn depth:** 10+ turns per task (research agents, software engineering agents)

Below these thresholds, KV-Cache I/O is fast enough that NIC saturation doesn't occur and DualPath provides no benefit.

### Concrete Agentic Scenarios Where DualPath Matters

**1. Software Engineering Agents (SWE-bench style)**
An agent reads an entire codebase, identifies a bug, writes a fix, runs tests, reads the test output, revises the fix, re-runs tests. Each turn re-loads 40K–80K tokens of prior code context from the KV-Cache. At scale (hundreds of such agents running simultaneously), storage NICs saturate quickly. DualPath distributes these loads, keeping queues short.

**2. Web Research Agents (WebArena style)**
An agent browses multiple web pages, scrapes content, summarises, follows links, and synthesises findings across many steps. Each page visit appends thousands of tokens to the context. The accumulated KV-Cache grows turn by turn — exactly the I/O-heavy pattern DualPath is designed for.

**3. Long-Form Document Processing**
Agents that process legal contracts, scientific papers, or financial reports — reading, annotating, cross-referencing sections across many turns. Contexts of 100K+ tokens are common, and every new turn re-loads most of that history.

**4. Multi-Agent Pipelines (AutoGen / CrewAI style)**
Orchestrator agents spawn sub-agents, collect their outputs, synthesise, and iterate. Each agent maintains its own long context. When dozens of agents run in parallel on the same cluster, aggregate KV-Cache I/O can saturate storage NICs at the cluster level, not just per-machine. DualPath's global scheduler operates at cluster scope, distributing load across both prefill and decode machine storage NICs.

**5. Code Execution Loops**
Agents that write code, execute it, read stdout/stderr (potentially large), debug, and retry. Code output can be verbose (stack traces, data dumps), rapidly expanding the KV-Cache. These are naturally I/O-burst workloads — heavy load on every turn, brief compute gaps between turns.

### Systems This Builds On / Fits Into

| System | Role | How DualPath Fits |
|---|---|---|
| **vLLM / SGLang** | LLM serving framework | DualPath is a scheduling layer on top of these, not a replacement |
| **Mooncake** | KV-Cache disaggregation (ByteDance) | Mooncake offloads KV-Cache to disk; DualPath optimises the load path from that disk |
| **LayerKV** | Layer-wise KV-Cache management | Complementary — LayerKV decides what to store; DualPath decides how to load it back |
| **ChunkPrefill / Sarathi** | Prefill chunking to reduce decode stalls | Addresses compute-side prefill bottleneck; DualPath addresses the I/O-side bottleneck |
| **PagedAttention** | Non-contiguous KV-Cache blocks | DualPath works with paged layouts — pages are the unit loaded via either path |

### Who Would Deploy This Today

The most direct users are **large-scale LLM serving operators** who:
1. Already run disaggregated prefill/decode (PD disaggregation is a prerequisite)
2. Already offload KV-Cache to SSD/NVMe (KV-Cache offloading is a prerequisite)
3. Are serving agentic workloads with long, multi-turn contexts at high concurrency

This describes cloud providers running inference APIs for coding assistants (Cursor, GitHub Copilot backends), enterprise document agents, and AI search products — anywhere that agentic task completion, not single-turn chat, is the dominant workload pattern.

---

## Relation to Other Concepts

- **Disaggregated inference** (`PD disaggregation`) — prerequisite architecture; DualPath is a scheduling layer on top
- **KV-Cache offloading** — DualPath assumes KV-Cache is already being offloaded to disk; it optimises the *loading* path, not the storage decision
- **RDMA / compute network** — the existing prefill→decode transfer channel is repurposed as a KV-Cache relay channel in Path 2
- **Prefill-decode bottleneck shift** — complements work like ChunkPrefill and Sarathi that address compute-side prefill bottlenecks; DualPath addresses the I/O-side bottleneck that emerges at longer contexts
