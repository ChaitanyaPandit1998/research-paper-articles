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

## Relation to Other Concepts

- **Disaggregated inference** (`PD disaggregation`) — prerequisite architecture; DualPath is a scheduling layer on top
- **KV-Cache offloading** — DualPath assumes KV-Cache is already being offloaded to disk; it optimises the *loading* path, not the storage decision
- **RDMA / compute network** — the existing prefill→decode transfer channel is repurposed as a KV-Cache relay channel in Path 2
- **Prefill-decode bottleneck shift** — complements work like ChunkPrefill and Sarathi that address compute-side prefill bottlenecks; DualPath addresses the I/O-side bottleneck that emerges at longer contexts
