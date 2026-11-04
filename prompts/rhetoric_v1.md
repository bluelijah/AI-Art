# Annotation prompt: rhetorical devices and emotion (v1)

You are annotating the sentences of a political speech for a research project
on how delivery, wording and audience reaction line up in time. You see the
transcript only, never the audio or the audience. Annotate each numbered
sentence on its own terms, using the neighbouring sentences only as context.

Be literal and consistent. When several values apply, include all of them.

## devices (list)

Based on Max Atkinson's work on political rhetoric (*Our Masters' Voices*, 1984)
plus common classical devices.

| Value | Definition | Example |
|---|---|---|
| `CONTRAST` | Two parts set against each other (not X but Y; you say X, but Y) | "You say you hear us, but..." |
| `LIST_OF_THREE` | Three parallel items, phrases or clauses | "People are suffering. People are dying. Entire ecosystems are collapsing." (each sentence of the list gets the tag) |
| `ANAPHORA` | Repeats an opening word or phrase from a recent sentence | "How dare you..." after an earlier "How dare you" |
| `RHETORICAL_QUESTION` | A question not meant to be answered | "How dare you?" |
| `DIRECT_ADDRESS` | Speaks straight to the audience as "you" | "You have stolen my dreams" |
| `STATISTIC` | Cites a number, percentage or quantity | "a 50% chance" |
| `PERSONAL_TESTIMONY` | Speaker's own experience or position | "I should be back in school" |
| `ACCUSATION` | Assigns blame to the audience or a group | "You are failing us" |
| `PUZZLE_SOLUTION` | Sets up a puzzle in one part and resolves it in the next | |
| `WARNING` | Threat or warning about consequences | "We will never forgive you" |
| `PUNCHLINE` | Short, quotable line that lands a point; often a completion point | "Change is coming, whether you like it or not" |
| `NONE` | None of the above | |

## emotion (one value)

`ANGER`, `GRIEF`, `FEAR`, `DEFIANCE`, `RESOLVE`, `DISBELIEF`, `IRONY`, `HOPE`, `NEUTRAL`

## intensity (0-3)

How emotionally charged the wording is: 0 = flat or informational,
1 = mild, 2 = strong, 3 = peak.

## addressee (one value)

`LEADERS` (the people in the room / those in power), `YOUTH`, `WORLD`,
`SELF`, `NONE`

## topic (one value)

`PERSONAL_STAKES`, `SCIENCE_NUMBERS`, `INACTION`, `MORAL_JUDGEMENT`,
`FUTURE_WARNING`, `GREETING_CLOSING`, `OTHER`

## invites_applause (true / false)

Atkinson's "claptrap": does the sentence end on a clear completion point that
is built to invite applause (a punchline, the third item of a list, the
second half of a contrast, a quotable accusation)? Judge from the wording
only. This is a prediction we later compare with the real audience.

## note

One short phrase explaining the main device, e.g. "second 'How dare you'".
