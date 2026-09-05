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

## English: ParTUT and LinES XPOS as extra columns

run: 2026-09-05 on commit bd52d880b5b2095638f4c9881e7dd1060b17f699, with UD 2.18 English datasets

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
