# `_can_record_outputs` — Output Capturing in Llama4

Llama4 introduces a declarative output-capturing mechanism to collect intermediate tensors (router logits, attention weights, hidden states) across layers without changing any forward method signatures. This is how `output_attentions=True`, `output_hidden_states=True`, and the MoE auxiliary loss work in Llama4.

---

## The Problem It Solves

In Llama3, collecting intermediate outputs requires threading extra flags through every forward signature:

```python
# Llama3 — every layer must explicitly handle and pass along these flags
def forward(self, hidden_states, ..., output_attentions=False, output_hidden_states=False):
    ...
    if output_attentions:
        all_attentions += (attn_weights,)
    ...
    return hidden_states, all_attentions, all_hidden_states
```

Every layer has to know about every output type. Adding a new output (e.g. router logits) means touching every layer's signature and return value.

Llama4 replaces this with a **registry + context manager** approach: each module *declares* what it can record, and a separate capture mechanism collects those outputs automatically without the modules coordinating with each other.

---

## The Declaration

On `Llama4TextModel`:

```python
class Llama4TextModel(Llama4PreTrainedModel):
    _can_record_outputs = {
        "attentions":    Llama4TextAttention,
        "hidden_states": Llama4TextDecoderLayer,
        "router_logits": Llama4TextMoe,
    }
```

This dict maps **output key name → the module class whose output should be captured**. It says:
- When collecting `"attentions"`, capture the return value of every `Llama4TextAttention` forward call
- When collecting `"hidden_states"`, capture the return value of every `Llama4TextDecoderLayer`
- When collecting `"router_logits"`, capture the return value of every `Llama4TextMoe`

---

## The Capture Decorator

`Llama4TextModel.forward` is decorated with `@capture_outputs`:

```python
@can_return_tuple
@merge_with_config_defaults
@capture_outputs
@auto_docstring
def forward(self, ...):
    ...
```

`@capture_outputs` wraps the forward method in a context that:
1. Reads the `output_attentions`, `output_hidden_states` flags from `kwargs`
2. Registers forward hooks on the module classes listed in `_can_record_outputs` for the flags that are `True`
3. Runs the actual forward pass (which doesn't know or care about being observed)
4. Collects all hook-captured tensors and attaches them to the output object

The modules themselves (`Llama4TextAttention`, `Llama4TextDecoderLayer`, `Llama4TextMoe`) return their normal outputs — no conditional branching on `output_attentions`, no extra return values. The hook intercepts the return value transparently.

---

## Router Logits and the Auxiliary Loss

`"router_logits": Llama4TextMoe` is the most practically important entry. `Llama4TextMoe.forward` returns `(hidden_states, router_logits)`. The capture hook collects `router_logits` from every MoE layer across the 48-layer stack.

These collected logits are used to compute the **load-balancing auxiliary loss**:

```python
# Conceptually, after forward:
all_router_logits = [logits from layer 0, logits from layer 2, ...]
aux_loss = load_balancing_loss_func(all_router_logits, num_experts, num_experts_per_tok)
total_loss = ce_loss + router_aux_loss_coef * aux_loss
```

Without this mechanism, the main forward pass would need to explicitly collect and thread router logits through every layer return. Instead, the decorator captures them silently.

**Why load balancing matters:** with top-1 routing and 16 experts, the model could degenerate — all tokens routing to expert 0, leaving experts 1–15 untrained. The auxiliary loss penalises unequal utilisation by measuring how concentrated the routing distribution is across experts. `router_aux_loss_coef=0.001` keeps this penalty small relative to the main language modelling loss but enough to encourage spread.

---

## Comparison with Llama3's Approach

| | Llama3 | Llama4 |
|---|---|---|
| How outputs collected | Each layer checks flags and conditionally appends to tuples | Forward hooks registered by `@capture_outputs` |
| Forward signature | Takes `output_attentions`, `output_hidden_states` explicitly | Flags consumed by decorator, not seen by forward body |
| Adding a new output type | Touch every layer's forward + return | Add one entry to `_can_record_outputs` |
| Router logits | N/A (no MoE) | Collected automatically, used for aux loss |

The Llama4 approach is more scalable — adding `"router_logits"` required zero changes to `Llama4TextDecoderLayer` or `Llama4TextMoe` forward methods.
