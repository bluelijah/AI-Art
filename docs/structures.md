# Hidden structures found in the four speeches

First pass, 26 Nov 2026. Speeches: Thunberg UN 2019, Thunberg COP24 2018,
Malala Yousafzai UN 2013, Martin Luther King Jr. 1963 (edited excerpt).
All numbers come from `output/<id>/structure/` and `output/compare/structure/`
(regenerate with `scripts/find_structure.py`). Correlations are Spearman rank
correlations (ρ, from −1 to +1); `*` marks p < 0.05. Samples are small
(34–125 sentences per speech, 1–10 applause moments), so these are structures
to look at and draw, not laws.

The findings fall into three layers, which map onto the project's three
signals: **how it is spoken**, **what is said (as the AI reads it)**, and
**how people respond**.

## 1. The shape of each speech

**Sections the machine found on its own.** Sliding a "novelty" detector down
each speech's self-similarity matrix (the technique music analysis uses to
find verse/chorus changes) splits every speech into 6–9 sections. The
detector never saw the rhetoric, only changes in meaning and delivery, yet
its boundaries fall on the speeches' real turns:

| Speech | Sections open on |
|---|---|
| King | "I have a dream that one day…", "I have a dream.", "I have a dream today.", "Sweet land of liberty…", "…let freedom ring" |
| Thunberg 2019 | "How dare you?" (twice), "50% may be acceptable to you", "…less than 350 gigatons", "You are failing us" |
| Thunberg 2018 | "…speak clearly, no matter how uncomfortable", "You are not mature enough…", "Maybe they will ask me about you", "You have ignored us…" |
| Malala | "Dear brothers and sisters, do remember one thing", "Be peaceful and love everyone", "…children [kept from] their schools", "We call upon…" |

**Arcs (AI-rated intensity, averaged per fifth of the speech, scale 0–3):**

| Speech | 1st | 2nd | 3rd | 4th | 5th | Shape |
|---|---|---|---|---|---|---|
| King | 1.2 | 2.0 | 2.3 | 2.3 | 2.5 | steady crescendo (intensity rises with position, ρ +0.39*), and he speeds up (+0.34*) |
| Malala | 0.8 | 2.0 | 1.8 | 1.8 | 2.1 | quiet greeting, then a long plateau that lifts at the end (ρ +0.48*) |
| Thunberg 2019 | 2.3 | 2.0 | 2.0 | 1.6 | 2.6 | starts hot, sinks into a "valley of numbers" (the carbon-budget passage), then the highest peak at the end; she also gets louder through the speech (+0.44*) |
| Thunberg 2018 | 0.9 | 2.4 | 1.6 | 2.0 | 1.8 | rises fast after the introduction, then holds |

## 2. Voice and words moving together (or not)

- **Thunberg 2019 puts her voice behind her rhetoric:** the more devices a
  sentence carries, the louder its peak (ρ +0.51*), and the more intense the
  wording, the louder (+0.46*).
- **Thunberg 2018 does the opposite with pitch:** her most intense sentences
  end *lower* (ρ −0.44*), a controlled, flattened delivery. Her breath-group
  shapes agree: 84% are the "flat" gesture (see below).
- **King's endings rise through the speech** (end pitch vs position +0.55*),
  and his loudest sentences end lowest (−0.53*).
- **Across all speakers:** louder endings are higher endings (+0.38*), and
  higher endings are followed by shorter pauses (−0.35*), i.e. a sentence
  that ends low is one the speaker lets sit.

## 3. Rhythm and gesture

- **Breath groups** (the runs of words between pauses, the speech's "lines"):
  Malala speaks in the shortest lines (median 3 words, 0.9 s, 421 lines);
  King in the longest (6 words, 2.2 s).
- **A shared gesture vocabulary.** Every line's pitch-and-loudness shape was
  clustered into six recurring gestures across all speakers: a flat line (0),
  a falling line (1), a rising line (2), a high arch that crashes (3), an arch
  with a steep final fall (4), and a falling cascade from high (5). Each
  speaker's mix is a signature:

| | flat | fall | rise | arch-crash | arch-fall | cascade |
|---|---|---|---|---|---|---|
| Thunberg 2019 | 41% | 40% | 5% | 3% | 11% | 0% |
| Thunberg 2018 | 84% | 12% | 0% | 2% | 2% | 0% |
| Malala | 15% | 15% | 21% | 20% | 16% | 12% |
| King* | 82% | 11% | 7% | 0% | 0% | 0% |

  Malala uses the whole vocabulary almost evenly; Thunberg 2018 almost only
  the flat line. *King's "flat" share is inflated: the 1963 recording is noisy
  and the pitch tracker loses his voice.

## 4. Echoes the AI hears between sentences

An embedding model places every sentence in a "meaning space"; sentences
close together are linked, even when they share no words.

**Across speeches** (speakers who never cite each other):

- Malala "And my dreams are the same." ↔ King "I have a dream today." (0.65)
- Malala "We all are equal." ↔ King "…all men are created equal." (0.62)
- Thunberg 2018 "…children can get headlines … just by not going to school" ↔
  Malala "…conflicts stop children to go to their schools" (0.66)
- Thunberg 2019 "People are suffering." ↔ Malala "Women and children are suffering…"
- Thunberg reusing herself: "not mature enough to tell it like it is" (0.92),
  "change is coming, whether you like it or not" (0.59), and money vs
  extinction (0.61) appear in both 2018 and 2019.

**Within a speech, far apart:** King's "let freedom ring" refrain (62 s
apart); Malala returning to "Dear sisters and brothers" across 5 minutes,
and to the Taliban who shot her ("…shot me on the left side of my forehead"
↔ "I do not even hate the Taliban who shot me", 117 s apart).

## 5. What predicts the room (and what doesn't)

- **Timing:** applause follows sentence ends within 0–2.5 s; for King, 57%
  of applause starts mid-sentence (the crowd breaks in), for the others 0%.
- **Continuous lead/lag is weak.** On the 100 ms timeline no sound signal
  (loudness, pitch, pause) leads applause with a clear peak; most "peaks" sit
  at the edge of the ±6 s window, i.e. slow trends, not cues. The brief's
  hypothesis (a pause after a device predicts applause ~2 s later) is not
  visible at this resolution with 1–10 applause events per speech.
- **The AI's applause prediction (consensus of 5 runs)** relates to real
  applause in Malala (ρ +0.28*) and weakly in Thunberg 2019 (+0.27), not in
  King or COP24. It catches most applause (67–80%) but most of what it
  predicts never gets applause (18–25% hit rate; 0% at COP24, where the room
  barely reacted).
- Applauded sentences: faster in Thunberg 2019 (+0.35*), lower-ending in
  Malala (−0.19*).

## 6. How the AI reads (a layer of its own)

Some of the strongest correlations are inside the AI's own reading, not in
the speech: intensity ↔ predicted applause (+0.59*), punchline ↔ predicted
applause (+0.60*), more devices ↔ more intense (+0.51*). These say the model
links "intense", "punchline" and "applause-worthy" together everywhere. That
is a structure of the machine's judgment, worth showing as such rather than
as a fact about the speeches. Its five runs agree 89–96% on applause
predictions but only 83–95% on emotion and 84–91% on devices.

## Caveats

- Small samples; treat ρ below about 0.3 as noise unless it repeats across speeches.
- The King recording is an edited excerpt: timing across its cuts is not real,
  and its pitch data is unreliable.
- Several features come from the AI (intensity, devices, applause_vote);
  correlations among them describe the model, not the speakers.
- Some pairs are related by construction and are excluded from the findings
  (e.g. sentence length vs word count, and pause-after vs applause, because
  applause fills the pause).
