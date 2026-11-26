# Structures found (auto-generated)

Numbers to check, not conclusions. Small samples: read p-values and n.

## Strongest sentence-level correlations (all speeches pooled, p < 0.05)

- applause_vote ~ has_punchline: rho +0.60 (p 0.000, n 243)
- intensity ~ applause_vote: rho +0.59 (p 0.000, n 243)
- intensity ~ device_count: rho +0.51 (p 0.000, n 243)
- intensity ~ has_punchline: rho +0.42 (p 0.000, n 243)
- applause_vote ~ device_count: rho +0.41 (p 0.000, n 243)
- end_loudness_rel_db ~ end_pitch_rel_st: rho +0.38 (p 0.000, n 228)
- position ~ has_personal_testimony: rho -0.37 (p 0.000, n 243)
- has_direct_address ~ has_warning: rho +0.36 (p 0.002, n 73)
- peak_loudness_rel_db ~ has_direct_address: rho +0.35 (p 0.000, n 243)
- has_accusation ~ has_direct_address: rho +0.35 (p 0.000, n 243)
- end_pitch_rel_st ~ pause_after_s: rho -0.35 (p 0.000, n 225)
- intensity ~ has_accusation: rho +0.35 (p 0.000, n 243)
- position ~ intensity: rho +0.34 (p 0.000, n 243)
- position ~ has_warning: rho +0.33 (p 0.004, n 73)
- end_pitch_rel_st ~ pace_wps_last5s: rho +0.30 (p 0.000, n 228)
- intensity ~ has_contrast: rho +0.30 (p 0.000, n 243)
- n_words ~ device_count: rho +0.30 (p 0.000, n 243)
- end_loudness_rel_db ~ intensity: rho +0.29 (p 0.000, n 243)
- has_accusation ~ has_rhetorical_question: rho +0.28 (p 0.000, n 198)
- intensity ~ has_warning: rho +0.27 (p 0.022, n 73)

## Lead/lag peaks (positive lag = first signal comes before the second)

- King 1963 (excerpt): applause_vote -> applause_prob: r -0.33 at -6.0 s  (edge of window: a trend, not a lead/lag)
- Thunberg 2018: applause_vote -> applause_prob: r -0.07 at +0.6 s
- Thunberg 2019: applause_vote -> applause_prob: r +0.38 at +6.0 s  (edge of window: a trend, not a lead/lag)
- Yousafzai 2013: applause_vote -> applause_prob: r +0.29 at +6.0 s  (edge of window: a trend, not a lead/lag)
- King 1963 (excerpt): intensity -> applause_prob: r -0.21 at -6.0 s  (edge of window: a trend, not a lead/lag)
- Thunberg 2018: intensity -> applause_prob: r -0.10 at +2.1 s
- Thunberg 2019: intensity -> applause_prob: r +0.24 at +6.0 s  (edge of window: a trend, not a lead/lag)
- Yousafzai 2013: intensity -> applause_prob: r +0.18 at +6.0 s  (edge of window: a trend, not a lead/lag)
- King 1963 (excerpt): intensity -> loudness_db: r +0.25 at +2.5 s
- Thunberg 2018: intensity -> loudness_db: r -0.07 at -0.1 s
- Thunberg 2019: intensity -> loudness_db: r +0.05 at +0.0 s
- Yousafzai 2013: intensity -> loudness_db: r +0.08 at +2.4 s
- King 1963 (excerpt): loudness_db -> applause_prob: r +0.19 at -0.6 s
- Thunberg 2018: loudness_db -> applause_prob: r +0.10 at +0.1 s
- Thunberg 2019: loudness_db -> applause_prob: r +0.13 at +5.3 s
- Yousafzai 2013: loudness_db -> applause_prob: r +0.30 at +0.1 s
- King 1963 (excerpt): loudness_db -> pause: r -0.79 at +0.0 s
- Thunberg 2018: loudness_db -> pause: r -0.87 at +0.0 s
- Thunberg 2019: loudness_db -> pause: r -0.76 at +0.0 s
- Yousafzai 2013: loudness_db -> pause: r -0.90 at +0.0 s
- King 1963 (excerpt): pause -> applause_prob: r -0.17 at -0.4 s
- Thunberg 2018: pause -> applause_prob: r -0.14 at +0.1 s
- Thunberg 2019: pause -> applause_prob: r -0.23 at +0.1 s
- Yousafzai 2013: pause -> applause_prob: r -0.28 at +0.1 s
- King 1963 (excerpt): pitch_hz -> applause_prob: r +0.08 at -0.9 s
- Thunberg 2018: pitch_hz -> applause_prob: r +0.31 at +0.1 s
- Thunberg 2019: pitch_hz -> applause_prob: r +0.20 at +1.7 s
- Yousafzai 2013: pitch_hz -> applause_prob: r +0.12 at -2.9 s

## Sections per speech

- Thunberg 2019: 8 sections; boundaries at 49s, 76s, 111s, 139s, 194s, 245s, 292s
- Thunberg 2018: 9 sections; boundaries at 16s, 39s, 69s, 93s, 128s, 153s, 184s, 208s
- Yousafzai 2013: 6 sections; boundaries at 172s, 459s, 629s, 829s, 1014s
- King 1963 (excerpt): 7 sections; boundaries at 61s, 114s, 176s, 230s, 300s, 366s

## Strongest meaning echoes across speeches

- 0.92: [thunberg_2019_un] "And you are still not mature enough to tell it like it is." <-> [thunberg_2018_cop24] "You are not mature enough to tell it like it is."
- 0.66: [thunberg_2018_cop24] "And if a few children can get headlines all over the world just by not going to school, then imagine what we could all do together if we really wanted to." <-> [malala_2013_un] "In many parts of the world, especially Pakistan and Afghanistan, terrorism, wars and conflicts stop children to go to their schools."
- 0.65: [malala_2013_un] "And my dreams are the same." <-> [mlk_1963_dream] "I have a dream today."
- 0.65: [malala_2013_un] "And my dreams are the same." <-> [mlk_1963_dream] "I have a dream today."
- 0.64: [malala_2013_un] "And my dreams are the same." <-> [mlk_1963_dream] "I have a dream."
- 0.63: [thunberg_2018_cop24] "And if a few children can get headlines all over the world just by not going to school, then imagine what we could all do together if we really wanted to." <-> [malala_2013_un] "We must not forget that millions of children are out of their schools."
- 0.62: [malala_2013_un] "We all are equal." <-> [mlk_1963_dream] "We hold these truths to be self-evident that all men are created equal."
- 0.61: [thunberg_2019_un] "We are in the beginning of a mass extinction, and all you can talk about is money and fairy tales of eternal economic growth." <-> [thunberg_2018_cop24] "Our civilization is being sacrificed for the opportunity of a very small number of people to continue making enormous amounts of money."
- 0.59: [thunberg_2019_un] "The world is waking up and change is coming whether you like it or not." <-> [thunberg_2018_cop24] "We have come here to let you know that change is coming, whether you like it or not."
- 0.59: [thunberg_2018_cop24] "And if a few children can get headlines all over the world just by not going to school, then imagine what we could all do together if we really wanted to." <-> [malala_2013_un] "to ensure free compulsory education all over the world for every child."

## Strongest echoes within a speech (far apart in time)

- 0.74, 62s apart [mlk_1963_dream]: "From every mountainside, let freedom ring." <-> "When we allow freedom ring, when we let it ring from every village and every hamlet, from every state and every city,"
- 0.72, 130s apart [malala_2013_un]: "Honourable UN Secretary General, Mr. Ban Ki-moon, respected President General Assembly," <-> "I fully support Mr. Ban Ki-moon, the Secretary General, in his Global Education First initiative, and the work of the UN Special Envoy, Mr. Gordon Brown, and the respected President General Assembly, Volk Yaramic."
- 0.70, 284s apart [malala_2013_un]: "Dear sisters and brothers," <-> "So dear sisters and brothers, now it's time to speak up."
- 0.70, 117s apart [malala_2013_un]: "Dear friends, on the 9th of October 2012, the Taliban shot me on the left side of my forehead." <-> "I do not even hate the Taliban who shot me."
- 0.69, 69s apart [thunberg_2018_cop24]: "I speak on behalf of climate justice now." <-> "I care about climate justice and a living planet."
- 0.68, 291s apart [malala_2013_un]: "Dear brothers and sisters, do remember one thing." <-> "Dear sisters and brothers,"
- 0.66, 119s apart [malala_2013_un]: "Honourable UN Envoy for Global Education, Mr. Gordon Brown, respected elders, and my dear brothers and sisters, Asalaamu Alaikum." <-> "I fully support Mr. Ban Ki-moon, the Secretary General, in his Global Education First initiative, and the work of the UN Special Envoy, Mr. Gordon Brown, and the respected President General Assembly, Volk Yaramic."
- 0.65, 160s apart [malala_2013_un]: "Malala Day is not my day." <-> "I am the same Malala."
- 0.63, 575s apart [malala_2013_un]: "Dear brothers and sisters, do remember one thing." <-> "So dear sisters and brothers, now it's time to speak up."
- 0.63, 250s apart [malala_2013_un]: "I am here to speak up for the right of education of every child." <-> "Islam says it's not only each child's right to get education, rather it's their duty and responsibility."

## Per-speech correlations (p < 0.05, non-trivial)

- **Thunberg 2019**: intensity ~ applause_vote +0.79; has_anaphora ~ has_rhetorical_question +0.72; applause_vote ~ device_count +0.63; intensity ~ device_count +0.60; applause_vote ~ has_direct_address +0.57; applause_vote ~ has_punchline +0.56; position ~ has_punchline +0.55; emotion_agreement ~ has_accusation +0.52
- **Thunberg 2018**: intensity ~ device_count +0.71; intensity ~ applause_vote +0.68; intensity ~ has_accusation +0.61; applause_vote ~ device_count +0.59; duration_s ~ intensity +0.57; applause_vote ~ has_punchline +0.55; duration_s ~ applause_vote +0.52; has_accusation ~ has_direct_address +0.52
- **Yousafzai 2013**: applause_vote ~ has_punchline +0.67; intensity ~ applause_vote +0.49; position ~ intensity +0.48; intensity ~ has_punchline +0.45; intensity ~ device_count +0.42; position ~ has_personal_testimony -0.37; has_anaphora ~ has_list_of_three +0.34; intensity ~ has_contrast +0.33
- **King 1963 (excerpt)**: intensity ~ applause_vote +0.66; intensity ~ device_count +0.56; position ~ end_pitch_rel_st +0.55; has_anaphora ~ has_punchline -0.55; applause_vote ~ has_punchline +0.53; peak_loudness_rel_db ~ end_pitch_rel_st -0.53; n_words ~ device_count +0.51; intensity ~ has_punchline +0.50

## Gesture vocabulary share per speaker

Gesture 0 is the flat shape (pitch moves < 1 semitone). Old or noisy recordings (King) read as flat
more often because the pitch tracker loses the voice.

```
                     gesture_0  gesture_1  gesture_2  gesture_3  gesture_4  gesture_5
Thunberg 2019             0.41       0.40       0.05       0.03       0.11       0.00
Thunberg 2018             0.84       0.12       0.00       0.02       0.02       0.00
Yousafzai 2013            0.15       0.15       0.21       0.20       0.16       0.12
King 1963 (excerpt)       0.82       0.11       0.07       0.00       0.00       0.00
```

