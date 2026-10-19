---
title: Learner Profiles
---

# Learner Profiles

This page describes the target learners for this lesson. Both are researchers, not cloud engineers; both have a dataset and a question, and neither has access to a GPU cluster with someone else paying the bill.

## Priya, a graduate student with a lab dataset

Priya is a third-year PhD student in an engineering lab with about 20 GB of tabular sensor data and a few thousand labeled images. She has trained scikit-learn and PyTorch models on her laptop and on the department's one shared workstation, where a hyperparameter sweep means leaving a job overnight and hoping nobody reboots it. Her lab has a small AWS allocation through the university, and her advisor has asked her to "just use the cloud", but she has never created a bucket and is worried about running up a bill she cannot explain. She has a GitHub account and uses git for her own code, without much ceremony.

From the lesson she gets a workflow she can repeat without a cluster: data in one S3 bucket, a $0.05-an-hour notebook that launches training and tuning jobs on instances that exist only while the job runs, a rule for choosing CPU vs. GPU for her data, and the habit of writing down an estimate, capping the runtime, tagging the job and comparing the estimate to the bill. By the end she can run a parallel tuning job on her own data and tell her advisor what it cost.

## Marcus, a digital-collections librarian

Marcus manages digitization at a university library and has tens of thousands of scanned pages: letters, ledgers and reports, many in old handwriting, plus a growing set of PDFs of published papers that patrons ask questions about. He is comfortable with Python for metadata cleanup and has used a commercial OCR service, but the per-page pricing does not scale to the backlog and the output on historical handwriting is poor. He wants a pipeline his small team can rerun each quarter as new material is scanned, with a cost per page he can put in a budget request, and he needs to be sure patron data never ends up somewhere it should not.

From the lesson he gets two concrete pipelines and the cost of each: retrieval-augmented question answering over a paper corpus on Bedrock (pay per token, nothing left running) and as a Processing job serving an open-weight model, and a LoRA fine-tune of a small vision-language model on a few hundred transcribed pages that improves transcription of exactly his kind of handwriting. He also gets the operational pieces a repeatable pipeline needs: one bucket with prefixes per step, tags that make the quarterly bill legible, and a cleanup checklist.
