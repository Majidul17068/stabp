# STABP — Set-Theoretic Agent Boundary Protocol

**Stop context bleeding between agents in a single-tenant LLM multi-agent system**
by moving the confidentiality boundary out of the (unreliable) prompt and into a
deterministic interception middleware.

> A prompt *asks* the AI to behave. STABP *removes* the AI's ability to misbehave:
> private, labeled fields are stripped at the inter-agent hop, so a downstream
> agent can neither copy nor paraphrase what it never received.

## Headline result (real, on OpenAI models)
Relay benchmark — a coordinator summarizes team updates for an external client;
semantic leak scored by an **independent gpt-4o judge**, 40 runs, 95% CIs.

| Setting | gpt-4o-mini | gpt-3.5-turbo | **STABP (typed)** |
|---|---|---|---|
| naive (no rule) | 1% | **100%** | **0%** |
| prompt rule | 0% | 0% | **0%** |
| prompt rule **+ injection** | 0% | **100%** | **0%** |

Prompt-based safety swings 0–100% by model, instruction, and injection; **STABP is
invariantly 0%.** Judge validated at **0% FPR / 12% FNR** (n=16) → reported leak
rates are not inflated and are conservative.

## Files
| File | What |
|---|---|
| `stabp_core.py` | The protocol library (boundary matrix, typed channel, middleware, checkers). |
| `tenantbound.py` | The benchmark (direct + relay scenarios, injection, semantic LLM judge, cluster-bootstrap CIs). |
| `validate_judge.py` | Judge FPR/FNR validation on a labeled set. |
| `make_matrix_figure.py` | Builds the paper's Figure 3 from the result files. |
| `results*.json` | Raw results for each model × condition. |
| `PROJECT_PLAN.md` | Phased roadmap + reproduce commands. |
| `EXPLAINER.md`, `HOW_IT_WORKS.md` | Plain-language + mechanism docs. |

## Reproduce
```bash
# key in .env (gitignored): OPENAI_API_KEY=sk-...
python3 tenantbound.py --provider openai --model gpt-4o-mini  --judge-model gpt-4o --scenario relay --repeat 40
python3 tenantbound.py --provider openai --model gpt-3.5-turbo --judge-model gpt-4o --scenario relay --repeat 40 --out results_gpt35.json
python3 tenantbound.py --provider openai --model gpt-4o-mini  --judge-model gpt-4o --scenario relay --inject --repeat 40 --out results_gpt4omini_inject.json
python3 tenantbound.py --provider openai --model gpt-3.5-turbo --judge-model gpt-4o --scenario relay --inject --repeat 40 --out results_gpt35_inject.json
python3 validate_judge.py --judge-model gpt-4o
python3 make_matrix_figure.py --out ../stabp-paper/figure3_frontier.tex

# offline demo (no key): mock mode proves the mechanism, not real-LLM behavior
python3 tenantbound.py --provider mock --scenario relay --repeat 12
```

## Honesty
- STABP is an *instantiation* of information-flow control (Denning 1976; SAFEFLOW; FORGE), not a new enforcement primitive. The contributions are the developer-facing set-algebra model + the TenantBound benchmark + the measured model-dependence of prompt-based safety.
- The typed channel is a **provable** 0% (Proposition 1); free text falls back to bounded-recall checkers; **semantic bleeding** (paraphrase) is beyond any non-LLM filter and is why STABP strips data *at the source*.
- Leak metric is judge-relative (validated FPR/FNR); mock mode is a code demo, never evidence.
