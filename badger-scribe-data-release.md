---
title: 'Badger Scribe data: publishing a release'
---

The fine-tuning episode's data (the German-letter part of the Badger Scribe competition data: 295 labelled train pages, 141 test pages, 436 JPEG scans plus a few CSVs) lives as a **GitHub Release** on the lesson repository, not in the git tree and not on Kaggle. Learners never need a Kaggle account. Each data version is one release with three zip assets:

| Asset | Contents | Size |
|---|---|---|
| `badger-scribe-v<N>-labels.zip` | `train.csv`, `test.csv`, `metadata.csv`, `metric.py`, `sample_submission.csv`, `transcription_conventions.md` | < 1 MB |
| `badger-scribe-v<N>-images-train.zip` | `images/<page_id>.jpg` for every page in `train.csv` | see release |
| `badger-scribe-v<N>-images-test.zip` | `images/<page_id>.jpg` for every page in `test.csv` | see release |

Unzipping all three into one directory recreates the original layout (`train.csv` next to `images/`), which is what the episode's training channel and the shared S3 bucket expect. GitHub allows 2 GiB per asset with free, unlimited download bandwidth; the images are split by train/test so each zip stays well under the limit even if the lesson later adds the other collections.

Why a release and not the alternatives: git would keep every version of every JPEG forever and slow every clone; Git LFS has a 1 GB free bandwidth cap that 50 clones exhaust in minutes; Google Drive throttles large downloads and `gdown` breaks regularly. A release is permanent, versioned, free, and sits next to the lesson.

## Publishing a new data version

Do this whenever the competition data changes (new pages pass review, labels are corrected). `<N>` is the lesson's own release number (1, 2, 3, ...); the release notes record which Kaggle dataset version it was taken from.

1. Download the current data from the competition's data tab into a folder, e.g. `~/Downloads/badger-scribe` (it should contain `train.csv` and `images/`).
2. Build the assets:

   ```bash
   python scripts/build_badger_scribe_release.py --src ~/Downloads/badger-scribe --version 1
   ```

   The script checks that every page id in the CSVs has an image, writes the three zips to `release/`, prints their sizes, and prints the publish command.
3. Write `release_notes_v1.md` (template below) and run the printed `gh release create ...` command, logged in to the GitHub CLI with write access to the repository that hosts the release (`carpentries-incubator/ML_with_AWS_SageMaker`; pass `--repo` to the build script to publish elsewhere, e.g. on a fork while developing). Uploading ~2 GB takes a few minutes.
4. Update the version number (`V=1`) in the episode's self-paced callout and in the staging step below. Earlier releases stay available, so a learner mid-way through an old version is not broken.

The lesson's URLs point at the release on `carpentries-incubator/ML_with_AWS_SageMaker`. A copy of the same release also exists on the `qualiaMachine/ML_with_Amazon_SageMaker` fork, where it was first published; the zips are byte-identical, so either URL serves the same data.

### Release notes template

```markdown
Lesson data release v1: the German-letter (Kade) pages of the Badger Scribe competition data, taken from Kaggle dataset version 9 (2026-09-17). 295 train pages, 141 test pages, all human-transcribed.

Three assets; unzip all three into one directory to recreate the layout
(`train.csv`, `test.csv`, `metadata.csv`, `metric.py`, `images/<page_id>.jpg`).
Used by the "Fine-tuning a Vision-Language Model with Training Jobs" episode.

Rights, per the holding institutions (see the competition data page for detail):

| Collection | Held by | Rights statement |
|---|---|---|
| Max Kade Institute German letters | Max Kade Institute for German-American Studies, UW–Madison | Copyright Undetermined (UND) |
| Dominy Craftsmen papers | Winterthur Library, Joseph Downs Collection | No Copyright – United States (NoC-US) |
| Wisconsin Land Survey Field Notes | Wisconsin Board of Commissioners of Public Lands | In Copyright – Educational Use Permitted (InC-EDU) |

Digitized by the UW–Madison Digital Collections Center; may be used for
non-profit educational and research purposes. German transcriptions by the
Max Kade Institute Kurrent Transcription Group. Scoring code, conventions and
documentation are MIT licensed, © 2026 ML+X, University of Wisconsin–Madison.
Not for redistribution or commercial use; do not crawl the UWDC catalogs.
```

## Staging the release in the shared bucket (before each term)

Students read the data from a shared read-only bucket; nothing is downloaded from GitHub during class. From any machine with the AWS CLI (CloudShell works):

```bash
V=1
R=https://github.com/carpentries-incubator/ML_with_AWS_SageMaker/releases/download/badger-scribe-data-v$V
mkdir -p badger-scribe && cd badger-scribe
for a in labels images-train images-test; do
  curl -L -o $a.zip $R/badger-scribe-v$V-$a.zip && unzip -q -o $a.zip && rm $a.zip
done
cd ..
aws s3 sync badger-scribe/ s3://<shared-bucket>/badger-scribe/
```

Then attach the read-only bucket policy from the fine-tuning cost page, and add your full fine-tune outputs under `badger-scribe/instructor/full-run/`.

## Self-paced learners

The episode tells learners working on their own account to download the same three assets and `aws s3 sync` them into a bucket of their own with the same `badger-scribe/` prefix. No Kaggle account is involved; the Kaggle page only comes up in the homework, as the optional place to submit a predictions file.
