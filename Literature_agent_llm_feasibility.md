# Literature verification — generative-LLM feasibility obstacles (Prof Ma feedback, 2026-08-04)

Round 8. Prof Ma asked that the claim in `email_to_prof_ma.md` — *"scores elicited from a
generative model are not stable, because the same filing can receive different scores under a small
change of prompt or a new model version"* — be "appropriately cited so as to assert its
credibility". The same paragraph appears as the scoping decision in `QnA_from_Prof_Ma.md` Q5.

Unlike Rounds 1–7, the `Literature/` and `Literature (new)/` PDF directories are **not on the
cluster**. Every source below was therefore downloaded from arXiv on this host and verified by text
extraction (`pdftotext`) against the claim it is being asked to support. Metadata (exact title,
full author list, publication date) came from the arXiv API, not from memory — this is the control
that caught the FinMTEB "Su et al." error in Round 7.

Legend: ✅ verified and used · ⚠️ verified but domain-limited, used with an explicit caveat ·
❌ verified and rejected.

---

## Layer 1 — Score instability (the claim as the email states it)

> ✅ **Sclar, M., Choi, Y., Tsvetkov, Y. and Suhr, A. (2024). "Quantifying Language Models'
> Sensitivity to Spurious Features in Prompt Design or: How I learned to start worrying about
> prompt formatting." ICLR 2024.** arXiv:2310.11324v2.

Verified from the abstract, quoted directly:

> *"We find that several widely used open-source LLMs are extremely sensitive to subtle changes in
> prompt formatting in few-shot settings, with performance differences of up to 76 accuracy points
> when evaluated using LLaMA-2-13B."*

This is the **direct** support for "a small change of prompt". The changes studied are explicitly
*meaning-preserving* formatting choices, which is the strongest possible form of the claim: not a
different question, the same question typeset differently. Note the paper is on open-source models,
which is the class this study would actually use.

> ⚠️ **Ouyang, S., Zhang, J. M., Harman, M. and Wang, M. (2024). "An Empirical Study of the
> Non-determinism of ChatGPT in Code Generation."** arXiv:2308.02828v2.

Verified from the abstract:

> *"the ratio of coding tasks with zero equal test output across different requests is 75.76%,
> 51.00%, and 47.56% for three different code generation datasets"* and *"setting the temperature
> to 0 does not guarantee determinism in code generation, although it indeed brings less
> non-determinism than the default configuration (temperature=1)."*

**Caveat, stated wherever this is cited:** the domain is code generation, not numeric scoring of
financial text. It supports "the same prompt returns different output, and temperature 0 does not
fix it" as a general property of the decoding process. It does not measure score dispersion on a
scoring task. Cite for the mechanism, not for a magnitude.

> ⚠️ **Wang, P., Li, L., Chen, L., Cai, Z., Zhu, D., Lin, B., Cao, Y., Liu, Q., Liu, T. and
> Sui, Z. (2023). "Large Language Models are not Fair Evaluators."** arXiv:2305.17926v2.

Verified from the abstract:

> *"the quality ranking of candidate responses can be easily hacked by simply altering their order
> of appearance in the context ... Vicuna-13B could beat ChatGPT on 66 over 80 tested queries with
> ChatGPT as an evaluator."*

**Caveat:** the setting is *pairwise comparison* of two candidate responses, not absolute scoring of
a single document, which is what this study would do. Adjacent evidence for scorer instability, not
direct. Used in `QnA_from_Prof_Ma.md` only, with the setting named.

> ❌ **Zheng, L. et al. (2023). "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena."**
> arXiv:2306.05685. Downloaded but **not verified in detail and therefore not cited.** The claim it
> would have supported (position and verbosity bias in LLM judges) is already carried by Wang et
> al. above, which was verified. Project standard is verified-or-omitted.

---

## Layer 2 — Version drift

> ✅ **Chen, L., Zaharia, M. and Zou, J. (2023). "How Is ChatGPT's Behavior Changing over Time?"**
> arXiv:2307.09009v3.

Verified from the abstract:

> *"GPT-4 (March 2023) was reasonable at identifying prime vs. composite numbers (84% accuracy) but
> GPT-4 (June 2023) was poor on these same questions (51% accuracy)"* and *"the behavior of the
> 'same' LLM service can change substantially in a relatively short amount of time, highlighting the
> need for continuous monitoring of LLMs."*

This is the direct support for "or a new model version", and the 84 → 51 figure is the concrete
number to quote. It also supplies the reproducibility argument: a feature built from a hosted model
is not a fixed measurement instrument.

---

## Layer 3 — Look-ahead contamination (the stronger obstacle, previously unstated)

Neither `email_to_prof_ma.md` nor `QnA_from_Prof_Ma.md` currently raises look-ahead as a reason to
scope out generative scoring, even though the study already measures encoder look-ahead at ~0.06 IC
(Q4) — several times the genuine text increment. The literature here is strong and finance-specific.

> ✅ **Glasserman, P. and Lin, C. (2023). "Assessing Look-Ahead Bias in Stock Return Predictions
> Generated by GPT Sentiment Analysis."** arXiv:2309.17322v1.

Verified from the abstract:

> *"backtesting produces biased results if the training and backtesting periods overlap. This bias
> can take two forms: a look-ahead bias, in which the LLM may have specific knowledge of the stock
> returns that followed a news article, and a distraction effect, in which general knowledge of the
> companies named interferes with the measurement of a text's sentiment."*

**MATERIAL CORRECTION to how this paper is usually invoked, and to my own earlier framing of it.**
The intuitive claim is "look-ahead inflates measured performance". That is *not* what they find
in-sample:

> *"In-sample (within the LLM training window), we find, surprisingly, that the anonymized headlines
> outperform, indicating that the distraction effect has a greater impact than look-ahead bias.
> This tendency is particularly strong for larger companies — companies about which we expect an
> LLM to have greater general knowledge."*

So the net contamination **cannot be signed in advance**: two mechanisms operate in opposite
directions, and which dominates depends on firm size and on the model's familiarity with the names.
For an S&P 500 panel — the largest, best-known firms, exactly where they report the distraction
effect is strongest — this is a *stronger* argument for caution than the naive version, because an
unsignable bias cannot be corrected for or bounded, only avoided.

Also relevant to study design: their proposed remedy is **removing the company's identifiers from
the text**, which is the anonymisation procedure adopted for the Round-8 LLM pilot. That design
choice is therefore published practice, not an invention of this study.

> ✅ **Wongchamcharoen, P. K. and Glasserman, P. (2025). "Do Large Language Models (LLMs)
> Understand Chronology?"** arXiv:2511.14214v2.

Verified from the abstract:

> *"prompt-based attempts against look-ahead bias implicitly assume that models understand
> chronology"* and *"Exact match rate drops sharply as sequences lengthen even while rank
> correlations stay high as LLMs largely preserve local order but struggle to maintain a single
> globally consistent timeline."*

This closes the obvious workaround. Telling the model "answer as of March 2020, ignore anything
later" only works if the model can place facts on a timeline; it largely cannot. Evaluated on
GPT-4.1, Claude-3.7 Sonnet and GPT-5, so it is not a small-model artifact.

> ✅ **Kong, Y., Lee, H., Hwang, Y., Lopez-Lira, A., Levy, B., Mehta, D., Wen, Q., Choi, C.,
> Lee, Y. and Zohren, S. (2026). "Evaluating LLMs in Finance Requires Explicit Bias
> Consideration."** arXiv:2602.14233v1.

Verified from the abstract:

> *"Finance-specific biases can inflate performance, contaminate backtests, and make reported
> results useless for any deployment claim. We identify five recurring biases in financial LLM
> applications. They include look-ahead bias, survivorship bias, narrative bias, objective bias, and
> cost bias ... We reviewed 164 papers from 2023 to 2025 and found that no single bias is discussed
> in more than 28 percent of studies."*

A position paper, so it is cited as evidence that the problem is recognised and unresolved in the
field, not as a measurement. The 28%-of-164-papers figure is the quotable number. Note it is a 2026
preprint without a journal reference yet — flagged in case a published version supersedes it before
submission.

---

## Net effect on the written position

The claim in the email ("scores are not stable") is **supportable but is the weaker of the two
available arguments**, and its strongest sources are outside finance (Sclar on prompt formatting,
Ouyang on decoding non-determinism). The stronger argument is look-ahead contamination, which is
finance-specific, has three verified sources, closes its own obvious workaround
(Wongchamcharoen & Glasserman), and connects to a number this study has already measured on its own
data (the ~0.06 IC encoder look-ahead premium in Q4).

Recommended ordering in both documents: **look-ahead first, instability second.**

## Sources downloaded to verify

`arxiv.org/pdf/{2309.17322, 2310.11324, 2305.17926, 2306.05685, 2307.09009, 2308.02828}`, plus
API metadata for 2511.14214 and 2602.14233. Retrieved 2026-08-04.
