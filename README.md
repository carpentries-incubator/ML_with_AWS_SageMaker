# Machine Learning on AWS with SageMaker and Bedrock

A Carpentries-style lesson on running machine learning and AI workflows on AWS, developed for the UW-Madison Machine Learning Marathon and Research Bazaar. The materials can also be used on their own with an AWS account.

**Rendered lesson:** https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/

## What the lesson covers

This workshop teaches researchers how to run ML and AI work on AWS while keeping the compute cost under control. It covers data storage, model training, fine-tuning, hyperparameter tuning, inference and retrieval-augmented generation (RAG), with hands-on exercises built on research problems.

Throughout the workshop, a small SageMaker notebook instance is the controller. It prepares data, launches jobs and reads the results. The demanding computation runs on separate instances that SageMaker starts for each job and shuts down when the job finishes. For inference, the lesson also uses Amazon Bedrock, which serves hosted models through pay-per-token APIs.

Cost management is built into the exercises. Learners estimate the cost before launching a job, set runtime limits, tag resources, and compare the estimate with the actual bill. The workshop finishes with resource monitoring and cleanup.

| Episodes | What they cover | Hands-on problem |
|---|---|---|
| 1 to 5 | AWS orientation, S3, setting up the controller notebook, reading and writing data in S3, and optional GitHub version control | Titanic survival data |
| 6, 7 and 9 | XGBoost and PyTorch training jobs, CPU and GPU compared, multi-instance training, and hyperparameter tuning | Titanic survival classifier |
| 8 | LoRA fine-tuning of CHURRO-3B, with a cost estimate and evaluation before and after training | [Badger Scribe](https://www.kaggle.com/competitions/badger-scribe), historical German handwriting transcription |
| 10 to 13 | RAG on Bedrock, multimodal retrieval, GPU Processing jobs, and optional self-hosted inference on a GPU notebook | [WattBot](https://www.kaggle.com/competitions/WattBot2026), energy-literature Q&A |
| 14 to 15 | Optional AWS CLI workflows, resource management, cost monitoring and cleanup | |

## Who it is for

Researchers and students who use machine learning or AI and want to run that work in the cloud. Participants should be comfortable with Python and Jupyter and know the basics of model training, hyperparameter tuning, fine-tuning, embeddings and RAG.

The focus is on running these workflows on AWS, not on teaching the ML and AI methods. No previous AWS experience is needed.

## Cost controls

The exercises are cheap. Run as written, the core workshop costs each learner a few dollars at the documented instance prices, CPU and GPU training jobs, hyperparameter tuning, LoRA fine-tuning and RAG inference included.

Every compute-heavy exercise comes with a cost estimate, a runtime limit, and a check against the actual billable time. Resources are tagged so that spending can be attributed to each learner and project. The final episode covers resource monitoring, cleanup, and the account restrictions that block the expensive mistakes on a shared AWS account.

## What it does not cover

- **General ML/AI instruction.** The workshop assumes familiarity with the methods and does not teach machine learning, fine-tuning or RAG from scratch.
- **Deploying models as endpoints.** Serving trained or fine-tuned models behind always-on APIs. A model trained elsewhere can still be run here for batch inference, weights in S3 and a Processing job, which the RAG with Processing Jobs episode shows.
- **Agents and tool use.** LLM applications that call external tools, or agent frameworks such as Bedrock Agents.

## Repository layout

| Path | What it is |
|---|---|
| `episodes/*.md` | The lesson text. **Edit these.** The rendered site and the notebooks are generated from them. |
| `notebooks/*.ipynb` | Jupyter notebooks generated from the episodes by `scripts/md_to_ipynb.py`. Regenerated automatically on every push to `main`. Do not edit by hand. |
| `scripts/` | The Python the episodes import or launch: training, fine-tuning and RAG scripts, cost estimation and launch checks in `helpers.py`, the notebook generator and the data preparation tools. Learners clone this repository onto their notebook instance. There is no separate helper repository. |
| `learners/` | Setup instructions, glossary, and the instance-selection guide. |
| `colab/` | A free, standalone notebook version of the fine-tuning episode for Google Colab's T4. It teaches LoRA with the same data, model and scripts and no AWS account. Edit the notebook directly; it is not generated. |
| `instructors/` | Instructor notes on account preparation, service quotas, cost estimates, data staging, and post-workshop cleanup with aws-nuke. |
| `data/data.zip` | The Titanic CSVs for the training exercises, and the WattBot metadata and training questions for the RAG exercises. |
| `config.yaml` | Lesson configuration and episode order. See the [Carpentries Workbench documentation](https://carpentries.github.io/sandpaper-docs/). |

## Data

The workshop uses three datasets, each prepared differently.

- **Titanic (training episodes).** Included in `data/data.zip`. Learners download it before the workshop and upload the CSVs to their own S3 bucket.
- **WattBot (RAG episodes).** The training questions (`train_QA.csv`) and paper metadata (`metadata.csv`) are also in `data/data.zip`. The PDF corpus is built with `scripts/build_wattbot_corpus.py`. Instructors stage the questions, the metadata and the papers in a shared read-only S3 bucket before class, so learners do not build the corpus themselves.
- **Badger Scribe (fine-tuning episode).** Distributed as a [GitHub Release](https://github.com/carpentries-incubator/ML_with_AWS_SageMaker/releases) (`badger-scribe-data-v1`), with 295 labelled training pages and 141 test pages from the Anna Seifert correspondence held by the Max Kade Institute, and human transcriptions of the training pages. Instructors stage the dataset and the CHURRO-3B model weights in the shared bucket before class. See `instructors/badger-scribe-data-release.md` for preparing a release and `instructors/shared-data-bucket.md` for staging.

Each learner creates one S3 bucket in the introductory episodes. Every later exercise writes its data, intermediate results, model artifacts and predictions to that bucket under its own prefix. Instructor-provided datasets and model weights are read from the shared bucket.

## Running the workshop

Before class, follow `instructors/aws-account-prep.md` to prepare the AWS account, including service quotas, Bedrock model access, cost-allocation tags and the launch guard rails. The other pages in `instructors/` cover episode-specific preparation, expected costs, and the datasets and models to stage in S3.

GPU quota increases can take weeks, so request them well ahead of the workshop. The instructor notes also say which exercises can use precomputed results to cut GPU demand during class.

After the workshop, follow `instructors/instructor-notes.md` to clean up the shared account, including with aws-nuke.

The fine-tuning episode also exists as a [free Colab notebook](https://colab.research.google.com/github/carpentries-incubator/ML_with_AWS_SageMaker/blob/main/colab/LoRA-finetuning-on-a-free-GPU.ipynb) that needs no AWS account. It runs the same LoRA fine-tune on Colab's T4 with the frozen weights in 4-bit.

Learners working on their own can follow the [rendered lesson](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/) on their own AWS account. They will need to stage the shared datasets and model files described in the instructor pages.

## Contributing

The lesson is written in Markdown for the Carpentries Workbench. To contribute, edit the relevant file in `episodes/` and open a pull request. The site and the notebooks are rebuilt automatically.

Code blocks in the episodes become notebook cells, so keep each block runnable on its own and keep the order of variables and imports across cells.

For issues, suggestions, or requests for other ML and AI workflows, [open a GitHub issue](https://github.com/carpentries-incubator/ML_with_AWS_SageMaker/issues) or email [endemann@wisc.edu](mailto:endemann@wisc.edu).

Development takes place on the [qualiaMachine fork](https://github.com/qualiaMachine/ML_with_Amazon_SageMaker), and changes are synced to the Carpentries Incubator repository by pull request.

## Citation and license

Please cite the lesson using `CITATION.cff`.

The instructional material is licensed under CC BY 4.0 and the code under the MIT license. See `LICENSE.md` for details.

The Badger Scribe page images and transcriptions have their own rights statements, documented in the release notes and in `instructors/badger-scribe-data-release.md`.
