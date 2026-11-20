

## 1. System claim and intended users

Our claim to test is that the app can identify and count standard Mahjong tiles
in phone photos where every tile face is visible. The intended users are Mahjong
players who want to enter their hands from a photo instead of typing each tile.
Our initial test covers the tile designs and backgrounds in the supplied photos;
it does not establish performance on every Mahjong set.
We are testing the web app's YOLO detector. The chatbot, CLIP option, and score
estimates are outside this test. Jokers and red fives are not supported.

## 2. Initial cases, including difficult examples

The photos are in `evaluation/images/`, and their expected tile counts are in
`evaluation/cases.json`.

| Cases | Photos | What they show |
| --- | --- | --- |
| T01-T04 | 4 | Single tiles covering circles, bamboo, characters, and winds |
| T05-T08 | 4 | Groups of two to five tiles, including touching and rotated tiles |
| T09-T10 | 2 | A sideways four-character tile and a sideways south wind |

The photos were chosen before running the detector. A teammate still needs to
check the labels. There are no empty-table or unrelated-object photos, so we
cannot test whether the app correctly returns no tiles. We also have no groups
with repeated tile identities. The photos share tile designs and backgrounds,
which limits how much the results can tell us about other sets and settings.
To include negative examples while keeping 10 cases, we can replace two group
photos with an empty table and an unrelated object. Those photos are still
needed; the current set contains difficult examples but no negative examples.

## 3. Ground truth and scoring instructions

The expected tile identities and counts in `evaluation/cases.json` are our
ground truth. They were labeled by visually inspecting each photo before
inference. A teammate will check them and resolve any disagreements before
we run the test. For example, `{"5p": 1}` means one five-circle tile. A verified
negative photo would have `{}` and should produce no detections.

A photo passes only when every tile identity and count matches the label.
Missing a tile, adding an extra tile, or naming a tile incorrectly makes it a
failure. The order of the results does not matter.

## 4. Primary metric and failure slice

Our primary metric is **exact-hand accuracy**: the number of fully correct
photos divided by the number of photos tested. For example, eight correct photos out of ten would be 80%.
Our failure slice is **sideways single tiles** (T09-T10). We will report how
many of these two photos pass and compare that with the four clear single-tile
photos (T01-T04). This checks whether orientation causes problems that an
overall score might hide. We will also look at mistakes in the groups. The six development examples will
have their own results.

## 5. Robustness check and expected behavior

We will make darker copies of the four clear single-tile photos using 70% of
their original brightness. As long as the tiles are still readable, we expect
the same tile names and counts. We will record how often the answers change
and whether they are correct before and after darkening. These copies are
extra checks, not additional test photos.

## Running the test

We will use AR-Mahjong YOLOv8n with a confidence threshold of 0.4, a 640-pixel
model input, and the web app's existing image processing. We will keep these
settings fixed during the test.

```sh
python evaluate.py --validate-only
python evaluate.py
```

Results go to `outputs/evaluation.json`. We will review each mistake, note a
likely cause, and decide what to try next. We will also save the environment
and repository version so the run can be reproduced:

```sh
python -m pip freeze > outputs/evaluation-environment.txt
git rev-parse HEAD > outputs/evaluation-commit.txt
```

