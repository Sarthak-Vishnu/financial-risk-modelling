# Sunnie Feedback Analysis — Feedback 02 (Prof Tiejun Ma)

> **Context:** Feedback received after submitting the Dissertation Progress Report
> (`S2880814_Dissertation_Progress_Report.pdf`, July 2026), which is built from `study_extended.md`.
> Unlike Feedback 01 — which asked for a *structural* reframing of the whole project (benchmark
> framing, incremental-value question, three baseline groups, separable feature table) — this round
> **endorses the reframing** and asks only for four *framing* refinements, all in the
> Results/Conclusions. Nothing in the design, data, or numbers is questioned.
>
> Report locations below are given for `study_extended.md` (the living source that will be edited
> for the thesis) and the latex report. The submitted PDF is frozen; these changes are for the thesis draft, not the
> progress report.

---

## Working notes (Sarthak)

> **Note (workflow):** `S2880814_Dissertation_Progress_Report.pdf` in the GitHub repo is the **old**
> progress report — it is submitted and frozen. Following these edits/improvements from Sunnie, they
> are to be amended in the **working LaTeX file**, only on Sarthak's approval/confirmation. Do not
> treat any point below as auto-applied.

> **Note (yfinance / price sources — my reminder):** the `yfinance` fallback for the price data is to
> **not be mentioned anywhere** (supervisor's instruction — it is a minor source used for only a
> couple of tickers, and surfacing it invites unnecessary data-consistency questions later). Action:
> remove the `yfinance` mention from **Appendix D** of `study_extended.md` (and keep §0.2 / §3.2 as
> "two sources"), so the body and appendix agree. This is separate from Sunnie's framing feedback —
> recorded here so it is not lost.

---

## Sunnie's Full Reply

> Overall, I think the experiments are well aligned with the revised research question. The original
> problem has been reframed into a much more scientifically grounded and defensible one, and the
> experimental design is consistent with that framing throughout. The results directly address the
> revised RQ, showing that the structured baseline is the strongest predictor, that Item 1A
> disclosures provide a small but consistent incremental signal under the proposed evaluation
> protocol.
>
> The only suggestion I have is to slightly soften the explanation of why TF-IDF performs better.
> The experiments clearly demonstrate that dense encoders do not outperform TF-IDF out of sample,
> but the interpretation that the volatility signal is fundamentally "lexical" and that dense
> representations learn an "era-specific" mapping is still an inference rather than something
> conclusively established by the current evidence. Framing this as a plausible explanation supported
> by the results, rather than a definitive mechanism, would make the conclusions even more convincing.
>
> At a higher level, I think the experiments are well designed, but I would encourage making the
> scientific takeaway slightly more explicit. The revised research question is about the incremental
> predictive value of Item 1A disclosures, whereas much of the Results naturally focus on comparing
> text representations. Explicitly tying these comparisons back to the incremental-value question
> would further strengthen the overall story.
>
> Also for the negative result, rather than presenting it as "the encoder failed to outperform
> TF-IDF," I would frame it more positively as identifying the conditions under which simple lexical
> representations remain competitive. That makes the contribution feel more like a scientific finding
> than a comparison of models.

---

## Point 1 — Overall assessment: the reframing is endorsed

> *"The experiments are well aligned with the revised research question. The original problem has
> been reframed into a much more scientifically grounded and defensible one … The results directly
> address the revised RQ, showing that the structured baseline is the strongest predictor, that
> Item 1A disclosures provide a small but consistent incremental signal under the proposed evaluation
> protocol."*

### Reading
This is the headline outcome of the round, and it closes the loop opened by Feedback 01. Every
structural change Feedback 01 asked for — the benchmark framing, the incremental-value question, the
non-text/structured baseline, the separable firm-filing feature table — is now in place, and the
supervisor accepts them without reservation. He also restates the two load-bearing results back to
us correctly ("structured baseline strongest"; "small but consistent incremental signal under the
protocol"), which confirms the narrative is landing exactly as intended.

**No action.** This is validation. The three remaining points are all refinements of *tone*, not
corrections of substance — the design and results stand.

---

## Point 2 — Soften "why TF-IDF wins": interpretation, not established mechanism

> *"Slightly soften the explanation of why TF-IDF performs better … the interpretation that the
> volatility signal is fundamentally 'lexical' and that dense representations learn an 'era-specific'
> mapping is still an inference rather than something conclusively established … Framing this as a
> plausible explanation supported by the results, rather than a definitive mechanism, would make the
> conclusions even more convincing."*

### Is it correct?
**Yes, and it is the sharpest of the four.** The evidence establishes an *observation* decisively —
dense encoders do not beat TF-IDF out of sample, and a task-aligned encoder reaches in-period parity
but loses forward. The *explanation* for that observation (the signal is "lexical"; the dense mapping
is "era-specific" and drifts) is a well-motivated inference, consistent with concept-drift theory and
with Magner et al., but it is not something the experiments prove. We have not, for instance, isolated
the drift mechanism directly (e.g. by measuring year-to-year stability of the learned text→vol map),
so "era-specific" remains the best available reading rather than a demonstrated fact. An examiner is
more likely to challenge a stated mechanism than a clearly-labelled interpretation, so this softening
also *reduces attack surface* in the viva.

### Where the report currently over-commits
The word choice reads as settled fact in four places:
- **PDF Abstract:** *"a **mechanism** for why count-based text wins (the dense text-to-volatility map
  **is** era-specific and does not transfer forward …)."*
- **`study_extended.md` §III.3 / PDF §4.3.3:** *"The dense text-to-volatility mapping a supervised
  encoder learns **is** era-specific."* — asserted, then followed by the good hedge *"This reading is
  consistent with the concept-drift framework."* The hedge is right; the sentence before it isn't.
- **PDF §4.6 Synthesis:** *"The **underlying explanation is straightforward.** The volatility signal
  in Item 1A text **is** lexical-level …"*
- **`study_extended.md` §VI.2 / PDF §5:** contribution 2 is titled *"A characterisation of **why** the
  count model wins"* / *"the study **explains why** the count model wins."*

### What to change (concrete)
Keep the observation hard; make the explanation explicitly provisional. Minimal edits:
- "a mechanism for why" → **"a plausible explanation for why"**.
- "The dense … mapping **is** era-specific" → **"appears to be era-specific"** / **"is consistent with
  an era-specific mapping that drifts over time."**
- "The underlying explanation is straightforward" → **"The most plausible explanation"** (drop
  "straightforward").
- Contribution heading "A characterisation of why…" → **"A plausible account of why…"**; "the study
  explains why" → **"the study offers a well-supported explanation of why."**
- Optionally add one sentence naming what *would* settle it (a direct measurement of year-to-year
  drift in the learned mapping), which pre-empts the objection and doubles as future work.

This is a light-touch pass — roughly six word-level swaps — and it strengthens rather than weakens the
claim, because a hedged inference that survives scrutiny is more convincing than an over-stated one.

---

## Point 3 — Make the scientific takeaway explicit: tie comparisons back to incremental value

> *"The revised research question is about the incremental predictive value of Item 1A disclosures,
> whereas much of the Results naturally focus on comparing text representations. Explicitly tying
> these comparisons back to the incremental-value question would further strengthen the overall
> story."*

### Is it correct?
**Yes — and it is a presentation gap, not a design gap.** The RQ ("does Item 1A text add value over
structured controls?") is framed cleanly in the Introduction and answered cleanly in the anchor table
(§II.2 / §4.2.1: *"text on top of the structured baseline adds: struct+tfidf reaches 0.603 …"*). But
the moment the Results move into the **encoder grid (§II.3 / §4.2.2)** and the **backtests (§III /
§4.3)**, the prose slips into representation-vs-representation language ("no encoder beats TF-IDF",
"parity", "which representation ranks best"). A reader can lose the thread that every one of those
comparisons is still, underneath, answering *one* question: how much does each representation add on
top of the structured block. The information is all there; the connective tissue is thin.

### Where to reinforce it
- **§II.3 / §4.2.2 (encoder grid):** the section currently ends on "none of the parity rows beats the
  reference." Add a closing sentence that re-casts the whole grid in incremental-value terms, e.g.:
  *"In incremental-value terms the grid says one thing: no dense representation adds more over the
  structured baseline than the count model does, and the generic encoders add essentially nothing over
  it."*
- **§III.1 / §4.3.1 (backtest):** this section already does it well ("struct+tfidf adds +0.016 IC …
  over the fair structured [ridge] baseline"). Keep as the template for the others.
- **§III.3 / §4.3.3 (final verdict):** immediately after the bold verdict, restate it as an
  incremental-value conclusion (see Point 4).
- **Consider a single framing sentence at the head of §II / §4** ("Every comparison below is an
  increment over the structured baseline; the question is never which representation is best in the
  abstract, only which one adds the most over what is already known"). One sentence there inoculates
  the entire Results section against the drift Sunnie noticed.

This costs three or four sentences total and materially tightens the story.

---

## Point 4 — Reframe the negative result as a positive scientific finding

> *"For the negative result, rather than presenting it as 'the encoder failed to outperform TF-IDF,'
> I would frame it more positively as identifying the conditions under which simple lexical
> representations remain competitive. That makes the contribution feel more like a scientific finding
> than a comparison of models."*

### Is it correct?
**Yes — and it aligns exactly with the framing we already committed to for this document** (present
the outcome as a finding, never as something that "failed"). Sunnie is pushing in the same direction
we were: turn a contest ("X lost to Y") into a scientific statement ("here are the conditions under
which the simple method is preferable"). The material is already there — task alignment reaches parity
in-period, the frozen mapping loses forward — it just needs to be stated as a *condition* rather than
a *defeat*.

### Where the report currently reads as a contest
- **`study_extended.md` §III.3 / PDF §4.3.3 (bold verdict):** *"**no encoder configuration beats the
  count-based model under any admissible protocol.**"* — true, but phrased as a scoreboard.
- **PDF §4.2.2:** *"it reaches parity with the count model but does not beat it."*
- **§4.3.3:** *"Out of period, the encoder collapses."* — vivid, but "collapses" is contest language.

### What to change (concrete)
Keep the empirical verdict (it is the backbone of the contribution) but *lead or pair it* with the
positive statement. Suggested reframing to sit alongside the bold line:

> *"Stated as a finding rather than a contest: a simple lexical representation remains the stronger
> choice out of sample whenever the alternative is a dense encoder that must fix its text-to-volatility
> mapping on a bounded historical window. Task-aligned training reaches parity within its training
> period, but that parity does not survive forward transfer — so the conditions under which count-based
> text stays competitive are precisely the forward-looking, non-stationary ones that matter for
> deployment."*

And soften "the encoder collapses" → "the out-of-period advantage disappears." The contribution then
reads as *characterising a regime* (when is lexical enough?) rather than *reporting a loser*, which is
both stronger science and consistent with the no-failure framing we adopted.

---

## Summary of actions

| # | Point | Type | Effort | Where |
|---|---|---|---|---|
| 1 | Reframing endorsed; results land as intended | Validation | None | — |
| 2 | Soften "lexical"/"era-specific" from mechanism → plausible explanation | Word-level | ~6 swaps | Abstract, §III.3/§4.3.3, §4.6, §VI.2/§5 |
| 3 | Tie representation comparisons back to incremental-value RQ | Add connective sentences | ~4 sentences | head of §II/§4, end of §II.3/§4.2.2, §III.3/§4.3.3 |
| 4 | Reframe negative result as "conditions where lexical stays competitive" | Reframe verdict, keep the fact | ~2–3 sentences | §III.3/§4.3.3, §4.2.2 |

**Net:** no experiments, no numbers, no structural changes. Four framing edits, all in
Results/Conclusions, none touching the methodology or the data. Points 2 and 4 are the two that most
change how a reader receives the conclusions; Point 3 is cheap insurance for the whole Results
section; Point 1 is the win — the reframing is accepted.
