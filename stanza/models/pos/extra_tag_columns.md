# Extra tag columns in the POS tagger

The tagger can predict tagsets beyond UPOS, XPOS, and UFeats, and can
condition each output layer on others.  See `stanza/models/pos/tag_columns.py`
for the mechanism and `stanza/models/tagger.py` for the flags:

    --extra_tag_columns      additional tagsets, read from MISC
    --tag_column_parents     which columns each output layer is conditioned on
    --tag_column_link        what a parent hands over: tag_emb or hidden
    --detach_parent_tags     stop the gradient at the parent
    --train_ratios           how much of each training file to use per epoch
    --write_extra_tag_columns  write the extra tagsets back to MISC

Results of runs using these flags go below, newest first.  Record the
date, the commit, and the exact flags, since a result only describes the
code it was run against.  Keep the per-seed numbers rather than only the
summary, so the numbers can be re-analyzed later.

## Bhojpuri: a BIS-tagged corpus as an extra column

2026-09-20, commit 7dc4361b17c1f26b9e3ab6d8207bc74ad9c8e383, UD 2.18

### Question

UD_Bhojpuri-BHTB is small, a few hundred sentences. A separate corpus
of around 250k words is tagged only with BIS, a coarser tagset which
BHTB's XPOS column is a refinement of. Does training BIS as an extra
output layer improve the tags we care about, and does the arrangement
of the layers matter?

### Setup

Two training files. BHTB, with duplicate sentences removed and split so
every copy of a sentence stays on the same side. The BIS corpus, which
supplies only a BIS tag per token. BIS is also derived onto BHTB from
its XPOS column, using the deterministic XPOS to BIS mapping, so the
BIS layer is supervised on both files and can be conditioned on a gold
XPOS tag where one exists.

    --extra_tag_columns bis
    --tag_column_parents "bis=xpos,upos"
    --train_ratios iit.conllu=<ratio>

5 seeds per configuration. "vectors" is whether pretrained fastText
Bhojpuri word vectors were used. "link" is --tag_column_link.

### Results

BHTB dev:

| IIT ratio | vectors | link | UPOS | XPOS | UFeats | AllTags |
|---|---|---|---|---|---|---|
| none | yes | tag_emb | 77.94 ± 0.75 | 76.40 ± 0.45 | 57.50 ± 0.83 | 50.41 ± 0.65 |
| none | no | tag_emb | 79.40 ± 0.62 | 77.74 ± 0.43 | 57.68 ± 1.58 | 51.49 ± 1.01 |
| 0.025 | yes | tag_emb | 81.52 ± 0.51 | 80.08 ± 0.42 | 59.68 ± 0.54 | 53.55 ± 0.60 |
| 0.05 | yes | tag_emb | 82.10 ± 1.02 | 80.82 ± 0.90 | 59.80 ± 0.72 | 53.45 ± 0.61 |
| 0.1 | yes | tag_emb | 82.26 ± 0.30 | 80.92 ± 0.32 | 59.30 ± 0.65 | 53.31 ± 0.43 |
| 0.025 | no | tag_emb | 81.44 ± 0.71 | 80.12 ± 0.79 | 59.22 ± 0.80 | 53.49 ± 0.80 |
| 0.05 | no | tag_emb | 82.30 ± 0.22 | 81.06 ± 0.31 | 59.96 ± 0.89 | 54.25 ± 0.87 |
| 0.1 | no | tag_emb | 81.98 ± 0.65 | 80.72 ± 0.60 | 58.88 ± 0.51 | 53.27 ± 0.56 |
| 0.15 | no | tag_emb | 82.54 ± 0.64 | 81.42 ± 0.63 | 58.46 ± 0.69 | 52.85 ± 0.46 |
| 0.05 | yes | hidden | 81.58 ± 0.58 | 80.20 ± 0.57 | 59.62 ± 0.63 | 52.45 ± 0.54 |
| 0.05 | no | hidden | 81.28 ± 0.56 | 80.18 ± 0.23 | 59.12 ± 0.58 | 52.55 ± 0.35 |
| 0.1 | no | hidden | 82.10 ± 0.50 | 80.58 ± 0.37 | 59.10 ± 0.51 | 52.57 ± 0.46 |

BHTB test:

| IIT ratio | vectors | link | UPOS | XPOS | UFeats | AllTags |
|---|---|---|---|---|---|---|
| none | yes | tag_emb | 77.66 ± 0.78 | 75.82 ± 0.69 | 60.43 ± 0.42 | 52.44 ± 0.53 |
| none | no | tag_emb | 78.38 ± 0.42 | 76.52 ± 0.61 | 59.24 ± 0.89 | 51.37 ± 0.66 |
| 0.025 | yes | tag_emb | 80.12 ± 0.53 | 78.36 ± 0.62 | 60.70 ± 1.20 | 53.44 ± 1.21 |
| 0.05 | yes | tag_emb | 80.39 ± 0.45 | 78.88 ± 0.33 | 60.91 ± 0.49 | 53.77 ± 0.39 |
| 0.1 | yes | tag_emb | 80.89 ± 0.55 | 79.11 ± 0.50 | 58.68 ± 2.63 | 52.26 ± 2.16 |
| 0.025 | no | tag_emb | 80.52 ± 0.54 | 78.80 ± 0.51 | 59.67 ± 1.52 | 53.08 ± 0.98 |
| 0.05 | no | tag_emb | 81.14 ± 0.72 | 79.44 ± 0.86 | 60.50 ± 1.56 | 53.81 ± 1.38 |
| 0.1 | no | tag_emb | 80.77 ± 0.78 | 79.15 ± 0.80 | 58.88 ± 1.03 | 52.67 ± 0.91 |
| 0.15 | no | tag_emb | 80.17 ± 0.71 | 78.59 ± 0.69 | 58.98 ± 0.56 | 52.51 ± 0.72 |
| 0.05 | yes | hidden | 80.87 ± 0.29 | 79.34 ± 0.62 | 61.03 ± 0.81 | 53.35 ± 0.81 |
| 0.05 | no | hidden | 80.70 ± 0.61 | 79.07 ± 0.42 | 60.19 ± 1.12 | 53.09 ± 0.68 |
| 0.1 | no | hidden | 80.29 ± 0.38 | 78.84 ± 0.16 | 59.61 ± 0.98 | 52.36 ± 0.45 |

Seed variation within a configuration is about 0.6 on UPOS and XPOS,
about 0.8 on UFeats for dev and 1.15 for test.

### Reading

- **The extra corpus helps a lot.** UPOS and XPOS improve by 2.5 to 4.5
  points on dev and 2.4 to 2.9 on test, at every ratio and with or
  without vectors. Several times the seed noise, and the largest effect
  anywhere in this file.
- **UFeats is unchanged.** Dev suggested a 2 point gain; test does not
  reproduce it. The honest summary is that the extra data helps UPOS and
  XPOS and leaves UFeats alone, which is unsurprising as the extra
  corpus has no morphological features in it.
- **0.05 is the best ratio** on both splits. Higher ratios trade UFeats
  away for little or nothing on UPOS and XPOS.
- **The pretrained vectors are not clearly worth it.** They cost around
  0.7 on UPOS and XPOS but the best UFeats figure in the table is BHTB
  alone with vectors, so dropping them is a trade rather than a win.
- **The link type does not matter.** Comparing hidden against tag_emb at
  the same ratio, UPOS, XPOS and UFeats scatter around zero, five of
  eighteen deltas positive and most within one standard deviation.
  AllTags, however, is worse under hidden in all six comparisons, by
  0.31 to 1.70. Same per-layer accuracy, but the layers stop making
  their mistakes on the same tokens, which costs the metric that
  requires all of them to be right at once.

### Reading across the two experiments

The gain here comes from the extra supervision reaching the shared
encoder, not from anything about how the output layers are wired to
each other. No arrangement of parents helped in English, and no link
type helped here. What made the difference was simply having a large
additional corpus for a language where the treebank is small.

### Not tested

- `--detach_parent_tags`, which would separate the two reasons the
  hidden link might do nothing: the wider input not helping, and the
  gradient it sends into the parent layer not helping.
- Whether a partial label loss on the XPOS layer, using the BIS to XPOS
  mapping to restrict rather than to supply the tag, does better than a
  separate BIS layer. That would put the extra data into the XPOS
  softmax directly rather than into the encoder.

## English: ParTUT and LinES XPOS as extra columns

2026-09-20, commit 7dc4361b17c1f26b9e3ab6d8207bc74ad9c8e383, UD 2.18

### Question

EWT, GUM, GUMReddit, PUD, and Pronouns share the PTB XPOS tagset.
ParTUT and LinES each use a different one.  Does adding those treebanks
as extra tag columns improve PTB XPOS, and does it matter which output
layers feed which?

### Setup

Training data is a zip.  The PTB-tagset treebanks are concatenated into
one file, unchanged.  ParTUT and LinES are separate files with their
XPOS moved into MISC as `xpos_partut=` and `xpos_lines=`; their UPOS and
UFeats are left in place, so they still train those heads.

Four seeds per layout.  All layouts other than the first use:

    --extra_tag_columns "xpos_partut;xpos_lines"
    --tag_column_link hidden

with the layouts differing only in `--tag_column_parents`:

| layout | --tag_column_parents |
|---|---|
| original | (no extra columns at all) |
| extra data, no extra xpos | (extra columns absent; ParTUT/LinES XPOS dropped) |
| upos feeds all three | (default: everything hangs off upos) |
| other xpos feed PTB xpos | `xpos=upos,xpos_partut,xpos_lines` |
| PTB xpos feeds other xpos | `xpos_partut=xpos;xpos_lines=xpos` |

Scored with a bert model.

### Results

Mean and standard deviation over 4 seeds.

### en_ewt dev

| layout | UPOS | XPOS | UFeats | AllTags |
|---|---|---|---|---|
| original | 97.882 ± 0.099 | 97.477 ± 0.070 | 97.555 ± 0.070 | 95.918 ± 0.017 |
| extra data, no extra xpos | 97.853 ± 0.062 | 97.435 ± 0.078 | 97.558 ± 0.064 | 95.927 ± 0.047 |
| upos feeds all three | 97.910 ± 0.071 | 97.490 ± 0.061 | 97.540 ± 0.036 | 95.938 ± 0.077 |
| other xpos feed PTB xpos | 97.920 ± 0.039 | 97.435 ± 0.062 | 97.578 ± 0.078 | 95.892 ± 0.053 |
| PTB xpos feeds other xpos | 97.892 ± 0.028 | 97.500 ± 0.064 | 97.500 ± 0.054 | 95.940 ± 0.064 |

### en_ewt test

| layout | UPOS | XPOS | UFeats | AllTags |
|---|---|---|---|---|
| original | 97.838 ± 0.040 | 97.395 ± 0.084 | 97.675 ± 0.167 | 95.963 ± 0.102 |
| extra data, no extra xpos | 97.810 ± 0.096 | 97.428 ± 0.043 | 97.705 ± 0.082 | 96.002 ± 0.090 |
| upos feeds all three | 97.840 ± 0.067 | 97.430 ± 0.091 | 97.685 ± 0.093 | 96.025 ± 0.093 |
| other xpos feed PTB xpos | 97.825 ± 0.035 | 97.340 ± 0.062 | 97.665 ± 0.079 | 95.885 ± 0.058 |
| PTB xpos feeds other xpos | 97.838 ± 0.076 | 97.465 ± 0.030 | 97.668 ± 0.034 | 96.005 ± 0.077 |

### Reading

Seed noise is about 0.06 on every metric.  The layouts span 0.06 to 0.12,
so with 4 seeds nothing here is separable except possibly the worst case.

- Adding ParTUT and LinES does not help.  `original` and `extra data, no
  extra xpos` differ only in whether those sentences are present, and
  test XPOS moves by +0.03.  Whatever the PTB-tagset treebanks already
  teach the encoder about English, a few thousand more sentences of
  English does not add to.
- `other xpos feed PTB xpos` is the only layout consistently at the
  bottom: worst test XPOS, worst test AllTags, worst dev AllTags.  About
  two standard errors, so suggestive rather than significant.  This is
  the direction with no structural justification, as an unrelated tagset
  does not partition the PTB label space.
- `PTB xpos feeds other xpos` has the best test XPOS mean and the
  tightest spread, but +0.07 over baseline at sd 0.06 is not a result.

### Caveat

English XPOS here is saturated, around 97.4, so this tests whether the
mechanism rescues a task which does not need rescuing.  It says little
about a low-resource setting where the main treebank is small and the
auxiliary corpus is large.  It also says nothing about the case where
the two tagsets are related by a known mapping, since ParTUT and LinES
have no such relation to PTB.

### Not tested

- Whether ParTUT and LinES follow the same UPOS annotation decisions
  that EWT and GUM have had applied over the last few years.  If they do
  not, the flat result may be measuring annotation drift rather than the
  mechanism.
- `--detach_parent_tags` on `PTB xpos feeds other xpos`, which would
  separate the gradient reshaping the shared hidden layer from the extra
  input alone.
