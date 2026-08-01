# Harness Engineering, Demystified: The Discipline of Building the Scaffolding Around a Model

**Pull quotes:**
- "The model is not the product. The model wrapped in tools, memory, permissions, and a feedback loop that lets it check its own work — that's the product."
- "Prompt engineering asks a model to say the right thing once. Harness engineering asks a system to keep doing the right thing across a thousand steps, most of which the model will get slightly wrong."
- "A harness doesn't make a model smarter. It makes the model's mistakes recoverable — which, at scale, is indistinguishable from making it smarter."

---

A harness is everything that sits around a language model so it can act in the world instead of just answering a single prompt: the tools it's allowed to call, the context it's fed on each turn, the memory that survives between turns, the permission system that decides what it can do unsupervised, and the loop that keeps it iterating instead of stopping after one shot. Harness engineering is the discipline of designing that scaffolding well. This article works through where the term comes from, what problem it actually solves, the anatomy of a real harness worked through with a concrete task, what it doesn't fix, and how it relates to prompt engineering and fine-tuning.

---

## Table of contents

1. [How it came to be](#1-how-it-came-to-be)
2. [What problem does it solve](#2-what-problem-does-it-solve)
3. [The anatomy of a harness](#3-the-anatomy-of-a-harness)
4. [A worked example, step by step](#4-a-worked-example-step-by-step)
5. [Design principles that recur across harnesses](#5-design-principles-that-recur-across-harnesses)
6. [What harness engineering doesn't solve](#6-what-harness-engineering-doesnt-solve)
7. [Alternatives and relatives](#7-alternatives-and-relatives)
8. [Summary table](#8-summary-table)
9. [Key takeaways](#9-key-takeaways)
10. [Further reading](#10-further-reading)

---

## 1. How it came to be

For most of the history of language models, the interesting engineering happened *inside* the model — architecture, training data, loss functions, RLHF. The thing a user typed and the thing the model typed back were the entire interface. Better answers meant a better model, full stop.

**The prompt-engineering era (2020–2022).** As models got large enough to follow instructions reliably, a lot of leverage moved to *how you asked* — few-shot examples, chain-of-thought prompting, careful system messages. This was real, but it was still a single-shot discipline: one prompt in, one completion out, no state carried forward, no ability for the model to check its own work or take an action in the world.

**The agent era (2023–present).** Once models could reliably emit structured tool calls, a new kind of system became possible: a loop where the model's output isn't the final answer but a request to *do something* — run a shell command, edit a file, query an API — whose result gets fed back in before the model continues. This changed the unit of engineering. It was no longer enough to write a good prompt; someone had to decide which tools the model could call, what happened when a tool failed, how much history stayed in context, when to ask a human before proceeding, and how the whole loop terminated. None of that lives in the model's weights — it lives in the code and configuration wrapped around the model, and that wrapping is what practitioners started calling the **harness** — a name that echoes test harnesses in software engineering (scaffolding that drives a thing under test through realistic conditions), though its exact first use in agent tooling is informal and hard to pin to one source.

**Why the name stuck.** "Agent framework" implies a fixed product; "harness" better captures that this scaffolding is mostly bespoke per task and per organization — the same underlying model performs very differently depending on what it's harnessed to. Coding agents like Claude Code, autonomous research agents, and customer-support bots built on the same base model can have wildly different reliability, purely as a function of harness design: what's in the system prompt, which tools exist, how errors surface, and how aggressively the loop is allowed to run before stopping to check in.

> **Note.** Harness engineering didn't replace prompt engineering or fine-tuning — it sits on top of both. A well-designed harness still needs good prompts inside it, and a well-tuned model still needs a harness if it's expected to act rather than just answer. The three are complementary layers, not competing ones.

---

## 2. What problem does it solve

A raw model, called once, has three structural limitations that no amount of prompting fixes on its own:

**It has no memory beyond its context window.** Every call is stateless. If a task takes more steps than fit in one context, or spans multiple sessions, something outside the model has to decide what to carry forward and what to discard.

**It can't act, only speak.** A model can describe the shell command that would fix a bug, but it cannot run that command, see the result, and adjust — unless something outside the model actually executes the command and feeds the output back in.

**It has no sense of consequence.** A model asked to "clean up the temp files" has no built-in notion that `rm -rf` is riskier than moving a file aside, or that pushing to a shared branch affects other people. Left alone, it will act on the most literal reading of the instruction.

**What a harness actually is.** A harness is the code that closes all three gaps: it manages what enters the model's context on each turn (memory, retrieved files, prior tool results), it exposes a fixed set of tools the model can request and then actually executes them, and it enforces policy — what requires human confirmation, what's auto-approved, what's forbidden outright — independent of what the model itself thinks is fine. The model proposes; the harness disposes.

It's worth being precise about what harness engineering is *not*: it is not about making the underlying model reason better, and it does not change the model's weights at all. It only shapes what the model sees, what it's allowed to do, and how its actions get verified and fed back. A separate discipline — model training and fine-tuning — improves the model itself. Harness engineering answers "what can this model see, do, and be checked against"; training answers "how good is the model's judgment in the first place."

> **Worth remembering.** The same base model, given a bad harness (unlimited unsupervised shell access, no memory, no verification step) and a good harness (scoped tools, persistent memory, sandboxed execution, a review loop) will produce wildly different real-world reliability on identical tasks. The gap between "impressive demo" and "trustworthy tool" is very often a harness gap, not a model gap.

---

## 3. The anatomy of a harness

Every non-trivial agent harness, regardless of domain, is built from the same handful of parts.

![Diagram of one turn through an agent harness: the context builder feeds the model, the model emits a tool request, the permission gate approves or denies it, approved requests reach tool execution, and the result feeds back into the next turn's context, with memory read from and written back to the loop](harness-engineering-assets/harness-loop.svg)

*The diagram above simplifies two of the six pieces described below. Termination and escalation logic isn't drawn at all, since it governs whether another turn happens rather than anything that flows through this one. Tool interface also has no box of its own — it's the contract behind the "tool request" arrow, not a stage the flow passes through.*

**Context builder.** Decides what the model actually sees this turn: the system prompt, relevant memory, recent conversation, and any files or search results retrieved for the task. This is where a harness compensates for the model's statelessness — what isn't assembled here simply doesn't exist for the model.

**Tool interface.** The fixed menu of actions the model is allowed to *request*. A tool definition is a contract: a name, a schema for its arguments, and a promise about what it does. The model never executes anything itself — it emits a structured request, and the harness decides whether and how to fulfill it.

**Permission gate.** The policy layer that decides, per tool call, whether to run it automatically, ask a human first, or refuse outright. This is where the harness encodes judgment the model doesn't have on its own — that editing a file is reversible but force-pushing to `main` is not, for instance.

**Execution and feedback.** The harness actually runs the approved action — in a sandbox, in a real shell, against a real API — and captures the result (success, error, output) to feed back into the next turn's context. This closing of the loop is what separates an agent from a chatbot: the model's next decision is conditioned on the real outcome of its last one, not just on what it predicted the outcome would be.

**Memory.** Whatever is meant to survive past the current context window — user preferences, prior decisions, project state — has to be deliberately written somewhere and deliberately read back in later. Nothing persists by default; persistence is a harness feature, not a model feature.

**Termination and escalation logic.** The rule for when the loop stops: task complete, error budget exhausted, risk threshold crossed, or a fixed step limit reached. Without this, a model with tool access and a vague goal will keep taking actions indefinitely, which is a well-known failure mode in early agent systems.

> **Note.** None of these six pieces requires the model to be different from a plain chat model. A single frontier model can be the "brain" behind a customer-support bot, a coding agent, and a research assistant — the entire behavioral difference between those three products can live in how these six pieces are configured, not in the model itself.

---

## 4. A worked example, step by step

Take a concrete task: "Fix the failing test in `payment_test.go` and open a PR." Walk it through a harness the way Section 5 of the backpropagation article walked a forward and backward pass through a network — one concrete step at a time.

**Turn 1 — context assembly.** The harness builds the initial context: a system prompt describing the model's role and constraints, the user's instruction, and (from memory) the fact that this repository requires PRs to include a test-plan checklist. The model has not seen the failing test yet — it hasn't been given it, because the harness doesn't dump the whole repository into context by default; it waits to see what the model asks for.

**Turn 2 — a read-only tool call.** The model requests `run_tests(path="payment_test.go")`. The permission gate checks this against policy: read-only test execution is auto-approved, no confirmation needed. The harness runs it in a sandboxed shell and returns the failure output — a stack trace pointing at a nil-pointer dereference.

**Turn 3 — an edit.** The model requests an edit to `payment.go` to add a nil check. The permission gate treats file edits inside the working tree as auto-approved (reversible, local, no shared state touched) and applies it.

**Turn 4 — re-verification.** The model requests `run_tests` again rather than assuming the fix worked — this is the harness's feedback loop doing its job: the model's next move is conditioned on an actual rerun, not on its own confidence. Tests pass.

**Turn 5 — an escalation.** The model requests `git push` and `create_pull_request`. The permission gate treats *pushing to a remote and opening a PR* differently from a local edit — these are visible to other people and harder to reverse — so it pauses and asks the human for confirmation instead of auto-approving, even though the model itself is confident the fix is correct.

**Turn 6 — memory write.** Once approved, the harness records to memory that this project requires a test-plan checklist on PRs (useful context for next time), while discarding the verbose test-run logs from context, since they have no value beyond this task.

Nothing in this sequence required the model to be smarter than a model given the same instruction in a single stateless call. What made the difference was entirely outside the model: the harness decided what the model could see when, which actions it could take without asking, which it couldn't, and that a claimed fix gets re-verified rather than trusted.

---

## 5. Design principles that recur across harnesses

**Match autonomy to reversibility, not to task difficulty.** The riskiest moment in a harness is rarely the hardest reasoning step — it's the action that's hard to undo. A well-designed permission gate cares more about "can this be walked back" (editing a local file vs. force-pushing to a shared branch) than about how intellectually demanding the task was.

**Scope tools narrowly.** A tool named `run_shell(command: str)` gives the model unlimited surface area and pushes all the safety reasoning onto the prompt, where it's unreliable. A tool named `run_tests(path: str)` structurally cannot do anything else, which pushes the safety guarantee into the harness's code instead of the model's judgment. Narrower tools are easier to reason about and easier to sandbox.

**Don't trust the model's own account of what it did.** The re-verification in Turn 4 above is the general pattern: a harness that lets the model report "tests pass" without actually rerunning them is trusting a claim it could cheaply check. Wherever a claim can be verified mechanically, a well-built harness verifies it rather than accepting it.

**Treat context as a scarce, curated resource, not a dumping ground.** Every extra token in context is something the model has to attend past to find what matters, and something that pushes earlier, possibly more relevant, information further from the point it's needed. A harness that stuffs entire files or full tool-output logs into context indiscriminately degrades the same model that would perform well on a curated context.

**Make failure recoverable, not just detectable.** A harness that stops dead on the first tool error forces a human to reconstruct state and resume manually. A harness that surfaces the error back to the model as a normal turn — the same way a real error would surface to a human engineer — lets the loop try an alternative approach, which is usually cheaper than an escalation.

> **The one thing worth remembering from this section.** Every principle above is really the same idea applied in a different place: push judgment about risk, verification, and scope into the harness's structure wherever possible, rather than relying on the model to reliably infer it fresh from a paragraph of prompt every single time.

---

## 6. What harness engineering doesn't solve

A well-designed harness compensates for what a model structurally lacks — but it cannot manufacture capability the model doesn't have.

**It doesn't fix a model that reasons badly.** No amount of tool scoping or memory design turns a model that consistently misunderstands a domain into one that understands it. A harness can catch and surface an error faster; it cannot make the underlying judgment correct.

**It doesn't eliminate the need for good prompts.** A system prompt inside a well-architected harness that's vague, contradictory, or missing key constraints still produces unreliable behavior — the harness governs *what the model can do*, not *how well it decides what to do*, which still depends heavily on how the task and constraints are communicated.

**It doesn't make unsafe actions safe by allowing them slower.** Adding a confirmation step before a destructive action is a mitigation, not a fix, if the human approving it doesn't actually understand what they're approving. A permission gate that trains users to reflexively click "approve" is barely better than no gate at all.

**It doesn't remove the cost of running many steps.** Every tool call, context rebuild, and re-verification in the Section 4 walkthrough costs latency and, for hosted models, money. A harness that re-verifies everything exhaustively is more reliable and also slower and more expensive — that tradeoff doesn't disappear, it just becomes an explicit design knob (how aggressively to re-check, how big a step budget to allow) instead of an invisible one.

**It doesn't substitute for evaluation.** Knowing a harness is well-designed in principle is different from knowing it performs well on the actual tasks it will face. Harnesses, like models, need to be benchmarked against realistic tasks before being trusted — a good design on paper can still fail in practice against tasks its designer didn't anticipate.

> **The takeaway to hold onto.** Every limitation above is a limitation of what the harness was *handed* — a weak model, a vague prompt, an unevaluated design — not a limitation of the harnessing approach itself. If an agent behaves badly, the harness's plumbing is rarely the only place to look; the model's judgment and the clarity of its instructions usually deserve equal scrutiny.

---

## 7. Alternatives and relatives

Harness engineering is the dominant way to turn a model into an agent today, but it sits alongside a few adjacent approaches worth distinguishing.

**Fine-tuning the model for agentic behavior.** Instead of scaffolding a general model with tools and policy, train the model itself on trajectories of tool use, so it develops better native judgment about when and how to call tools. This can reduce how much the harness needs to compensate for, but it's expensive to iterate on (a training run vs. a config change) and doesn't remove the need for a permission layer — even a model trained to use tools well still shouldn't be trusted to unilaterally decide what's reversible in someone else's production system.

**Multi-agent orchestration.** Rather than one model looping inside one harness, split the task across several agents — a planner, a specialist executor, a reviewer — each with its own narrower harness, coordinated by an outer controller. This can make individual pieces easier to scope and verify, at the cost of new failure modes at the coordination boundary (agents disagreeing, or a plan that doesn't match what the executor can actually do).

**Pure prompt engineering, no tools.** For tasks that are genuinely single-shot — classify this text, summarize this document, answer this question from provided context — the full harness apparatus (tools, memory, permission gates) is often unnecessary overhead. Not every model deployment needs to be an agent; harness engineering earns its cost specifically when a task requires multiple steps, real-world action, or state that outlives one exchange.

**None of these has displaced harness engineering** for tasks that genuinely require multi-step, tool-using behavior, because the core problem — a stateless model needs something outside itself to hold memory, mediate action, and enforce policy — doesn't go away regardless of how good the model gets. A smarter model still needs to be told what it's allowed to touch.

> **Note.** It's tempting to treat "just use a smarter model" as a substitute for harness design, the same way "just use a bigger network" was once treated as a substitute for a better activation function. In both cases, the substitution only goes so far — capability and scaffolding solve different problems, and improving one doesn't remove the need for the other.

---

## 8. Summary table

| Concept | What it means | Where it lives |
|---|---|---|
| Context builder | Assembles what the model sees this turn — prompt, memory, retrieved data | Between the user/task and the model |
| Tool interface | Fixed, schema-defined actions the model may request | Between the model and the outside world |
| Permission gate | Policy deciding auto-approve / confirm / deny per action | Between a tool request and its execution |
| Execution & feedback | Actually runs approved actions and returns real results | Between the permission gate's approval and the next turn's context |
| Memory | What survives past the current context window | Read/written by the context builder |
| Termination and escalation logic | Rule for when the loop stops or escalates | Wraps the whole turn cycle |

---

## 9. Key takeaways

- **A harness compensates for what a model structurally lacks:** memory across turns, the ability to act rather than just speak, and any innate sense of which actions are reversible.
- **The model proposes, the harness disposes.** Every tool call is a request, not an execution — the harness decides whether, when, and how it actually happens.
- **Autonomy should scale with reversibility, not with how hard the task looked.** A local file edit and a push to a shared branch deserve very different levels of scrutiny even inside the same task.
- **Verification beats trust.** Wherever a claim the model makes ("tests pass," "the fix worked") can be mechanically re-checked, a well-built harness checks it rather than accepting it.
- **Context is a curated resource, not a dumping ground.** What the context builder chooses to leave out matters as much as what it includes.
- **A harness cannot manufacture capability.** It can make a model's mistakes cheaper to catch and recover from — it cannot make a poorly-reasoning model reason well, or a vague instruction unambiguous.
- **The dividing line is the model's weights.** A harness changes everything about what a model can see, do, and be checked against; it changes nothing about the model's underlying judgment — that's what training and fine-tuning are for.

---

## 10. Further reading

- **"ReAct: Synergizing Reasoning and Acting in Language Models"** (Yao et al., 2022) — one of the earliest formal descriptions of the reason-then-act loop that modern agent harnesses are built around: arxiv.org/abs/2210.03629

- **"Toolformer: Language Models Can Teach Themselves to Use Tools"** (Schick et al., 2023) — an early exploration of models learning when to invoke external tools, relevant background for why tool interfaces became a first-class harness component: arxiv.org/abs/2302.04761

- **SWE-bench** (Jimenez et al., 2023) — a benchmark specifically for evaluating coding agents on real GitHub issues, widely used to compare harness designs against each other on identical tasks and the same underlying models: swebench.com

- **The Claude Code documentation** — a concrete, shipping example of a coding harness: tool definitions, permission modes, memory, and sub-agents, described in the terms this article uses: docs.claude.com/en/docs/claude-code

- **The companion [backpropagation](backpropagation.md) and [activation functions](activation-functions.md) articles** — for the layer of the stack harness engineering deliberately does *not* touch: what makes the underlying model capable in the first place.

---

Harness engineering is, underneath all the tool schemas and permission configs, the same basic move every reliable system has always made around an unreliable component: don't ask the component to be perfect, build the scaffolding that catches it when it isn't. A model proposes an edit, a permission gate decides if that edit is safe to make unsupervised, a test run checks whether it actually worked, and a memory store carries forward whatever's worth remembering next time — none of that is the model thinking better, and all of it is the difference between an impressive single response and a system someone can actually depend on across a thousand steps.
