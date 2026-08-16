# Persona-Guided Large Language Model Interfaces as Creativity Scaffolds

Supplementary materials for the doctoral thesis of the same name.

Three studies test what happens when divergent and convergent thinking are externalised as two
separately addressable LLM personas. The first assigns one persona for a whole session, the second
gives the user both and a switch under laboratory control, and the third deploys the same interface
in a four-day design hackathon. This repository holds the persona prompts, the measurement and
analysis specifications, the analysis code, and the derived result tables behind those studies.

## What is here

| Directory | Contents |
|---|---|
| `prompts/experiment-1/` | The four persona system prompts deployed in system V0, with the shared preamble and the challenge anchor that accompanied each |
| `prompts/experiments-2-3/` | The Taylor (divergent) and Alex (convergent) prompts deployed in system V1, and the undifferentiated control prompt |
| `methods/` | Measurement and pipeline specifications per study: stance instruments, idea extraction and originality pipelines, rubric and judge designs, semantic-trajectory formulas |
| `code/experiment-1/` | Analysis scripts for the assigned-persona study |
| `code/experiment-2/` | Analysis scripts and notebooks for the user-switchable study |
| `code/experiment-3/` | Analysis scripts for the field deployment |
| `derived/` | Derived result tables: the aggregated outputs the thesis reports, one directory per study |

## What is deliberately not here

**Raw conversation transcripts are not published.** Participants consented to their conversations
being logged and analysed, not to their release. The derived tables carry the aggregated measures
the thesis reports, and the analysis scripts document exactly how those measures were computed from
the transcripts, so every reported result can be traced to its computation without the underlying
messages being redistributed. Requests for access to the transcripts for verification purposes
should be directed to the author.

Third-party model weights and tokenizer files used by the local extraction and scoring pipelines are
also excluded. The scripts name the models they load.

## Reading this against the thesis

The persona prompts implement the stance contracts specified in the thesis's methodology chapter.
The `methods/` files carry the appendix material for each study, so a reader checking a reported
measure should start there and then follow the corresponding script in `code/`. Section numbers
inside `methods/` are those of the reviewed draft and are retained so that the two can be read side
by side.

Measures are computed within a study, never across studies: each study embeds ideas with a different
model, so no distance in one study's derived tables is comparable with another's.

## Note on the scripts

These are research analysis scripts, written over the course of the work rather than as a package.
They are published because the thesis's claims should be checkable, not as a reusable library. Some
scripts supersede others, and several analyses that were run and then dropped from the thesis remain
here; where the thesis reports a measure, the `methods/` specification names the script that
produced it.

Any API keys formerly present in these scripts have been removed. Scripts that call a hosted model
expect the relevant key in the environment.
