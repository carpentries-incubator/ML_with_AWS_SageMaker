# Exploring Fact-Based QA with RAG: Romeo and Juliet
### Chris Endemann, endemann@wisc.edu
### [Nexus version](https://uw-madison-datascience.github.io/ML-X-Nexus/Learn/Notebooks/2025-05-07_RAG-Romeo-Juliet.html)
### Categories
- Notebooks
- RAG
- Retrieval
- NLP
- LLM
- Embeddings
- Text analysis
- Deep learning
- Prompt engineering
- Code-along


### New Cloud Stuff

There are several general approaches for setting up a Retrieval-Augmented Generation (RAG) workflow on AWS, each suited to different scales and constraints.

1. **Use long-lived inference endpoints for online RAG workloads.** For applications that need low-latency, interactive RAG (APIs, chatbots, dashboards), you can deploy your own embedding and generation models as SageMaker inference endpoints (or Bedrock-like managed endpoints) and call them from your retrieval service. This gives you control over the model, scaling policies, and autoscaling, but it’s also the most expensive option if traffic is low or bursty, since you’re keeping capacity online even when no one is querying the system.

2. **Run everything inside a single GPU-backed notebook instance**
For small- to medium-sized models (< 8 B), it's often simplest to just pick a GPU instance (e.g., [p3.2xlarge](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/instances-for-ML.html)), load your embedding and generation models directly in the notebook, and run RAG end-to-end there. This keeps the architecture simple and avoids extra moving parts, as long as you're disciplined about shutting down the instance when you’re done so you don’t leak cost. It's also possible to do this with larger models, but the costs to use more powerful GPUs (e.g., $15/hour) may be a limiting factor.

3. **Use SageMaker Estimators to run batch jobs for embeddings and/or generation.**
For large corpora or workflows where you want repeatable, offline computation, you can treat parts of the RAG pipeline—especially embedding—like a batch processing job rather than a live model. Instead of deploying an inference endpoint, you run a short-lived Hugging Face Estimator job that spins up a GPU instance, loads your embedding or generation model, processes all the chunked text in one shot, and saves the results back to S3. This pattern is ideal for “compute once, use many times” workloads, such as generating embeddings for thousands of documents or producing long-form outputs that don’t require low latency. Because the Estimator only exists while the batch completes, you avoid the continuous cost of an always-on endpoint. However, this approach is not suited for per-query RAG retrieval—launching an Estimator per user request would be far too slow, since starting a training job can take several minutes.

4. **Use Amazon Bedrock for managed embedding and generation APIs.** If you prefer fully managed foundation models and don’t want to own model hosting at all, Bedrock lets you call embedding and generation models via API from your RAG pipeline. You still manage your vector store and retrieval logic (for example in Lambda, ECS, or SageMaker), but you outsource the heavy model lifecycle work—at the trade-off of less control over architectures and sometimes higher per-token cost.



[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UW-Madison-DataScience/ML-X-Nexus/blob/main/Learn/Notebooks/2025-05-07_RAG-Romeo-Juliet.ipynb)

This notebook demonstrates the use of a Retrieval-Augmented Generation (RAG) system to answer factual questions from Shakespeare's *Romeo and Juliet*. Our long-term goal is to build a RAG-powered chatbot that supports literary exploration—helping readers investigate character dynamics, thematic development, and emotional subtext.

In this first part of the demo, we focus on low-hanging fruit: factual, quote-supported questions that a RAG pipeline can answer reliably. These examples will help us introduce key RAG components, and set a performance baseline before tackling more interpretive questions. 

### Learning objectives

By the end of this notebook, you should be able to:

- Identify the key components of a basic Retrieval-Augmented Generation (RAG) system.
- Use a sentence-transformer model to create embeddings from text passages.
- Run simple retrieval using vector similarity and evaluate retrieved chunks.
- Generate answers to factual questions using retrieved content as context.
- Understand early limitations of RAG pipelines and motivate future improvements.

### Step-by-step overview

1. **Load the corpus**
   - We use Shakespeare texts from the workshop's `data.csv` file.

2. **Split text into chunks**
   - Long texts are broken into smaller passages (~200 words) so they're easier to search and analyze.

3. **Create embeddings**
   - Each chunk is converted into a vector — a mathematical representation of its meaning — using a pretrained model from `sentence-transformers`.

4. **Retrieve relevant chunks**
   - When you ask a question, we embed the question and compare it to the embedded text chunks to find the most similar passages.

5. **Ask a language model**
   - We take the most relevant passages and feed them (along with your question) into a pretrained language model (like GPT-2) to generate an answer.

This is not training a model from scratch — it's a lightweight, modular way to build smart question-answering tools on top of your own text collection.

We'll explore the strengths and limitations of this approach along the way.

## Step 1: Load the corpus
In this example, we'll use "Romeo and Juliet" as our text corpus. This text is freely available via [Project Gutenberg](https://uw-madison-datascience.github.io/ML-X-Nexus/Toolbox/Data/Gutenberg.html).

Preview the file


```python
# Download Romeo and Juliet from Project Gutenberg
import requests

url = 'https://www.gutenberg.org/files/1112/1112-0.txt'
response = requests.get(url)
file_contents = response.text

# Preview first 3000 characters
preview_len = 3000
print(file_contents[:preview_len])
```

    *** START OF THE PROJECT GUTENBERG EBOOK 1112 ***
    
    
    Executive Director's Notes:
    
    In addition to the notes below, and so you will *NOT* think all
    the spelling errors introduced by the printers of the time have
    been corrected, here are the first few lines of Hamlet, as they
    are presented herein:
    
      Barnardo. Who's there?
      Fran. Nay answer me: Stand & vnfold
    your selfe
    
       Bar. Long liue the King
    
           *       *       *       *       *
    
    As I understand it, the printers often ran out of certain words
    or letters they had often packed into a "cliche". . .this is the
    original meaning of the term cliche. . .and thus, being unwilling
    to unpack the cliches, and thus you will see some substitutions
    that look very odd. . .such as the exchanges of u for v, v for u,
    above. . .and you may wonder why they did it this way, presuming
    Shakespeare did not actually write the play in this manner. . . .
    
    The answer is that they MAY have packed "liue" into a cliche at a
    time when they were out of "v"'s. . .possibly having used "vv" in
    place of some "w"'s, etc.  This was a common practice of the day,
    as print was still quite expensive, and they didn't want to spend
    more on a wider selection of characters than they had to.
    
    You will find a lot of these kinds of "errors" in this text, as I
    have mentioned in other times and places, many "scholars" have an
    extreme attachment to these errors, and many have accorded them a
    very high place in the "canon" of Shakespeare.  My father read an
    assortment of these made available to him by Cambridge University
    in England for several months in a glass room constructed for the
    purpose.  To the best of my knowledge he read ALL those available
    . . .in great detail. . .and determined from the various changes,
    that Shakespeare most likely did not write in nearly as many of a
    variety of errors we credit him for, even though he was in/famous
    for signing his name with several different spellings.
    
    So, please take this into account when reading the comments below
    made by our volunteer who prepared this file:  you may see errors
    that are "not" errors. . . .
    
    So. . .with this caveat. . .we have NOT changed the canon errors,
    here is the Project Gutenberg Etext of Shakespeare's The first
    Part of Henry the Sixt.
    
    Michael S. Hart
    Project Gutenberg
    Executive Director
    
           *       *       *       *       *
    
    Scanner's Notes:
    
    What this is and isn't. This was taken from a copy of
    Shakespeare's first folio and it is as close as I can come in
    ASCII to the printed text.
    
    The elongated S's have been changed to small s's and the
    conjoined ae have been changed to ae. I have left the spelling,
    punctuation, capitalization as close as possible to the printed
    text. I have corrected some spelling mistakes (I have put
    together a spelling dictionary devised from the spellings of
    the Geneva Bible and Shakespeare's First Folio and have unified
    spellings according to this template), typo's and expanded
    abbreviations as I have come across them. Everything within
    br


## Step 2: Split text into "chunks"
Next, we define a function to split the corpus into smaller chunks based on word count. The simplest "chunking" approach is to chunk by word count or character count.


```python
def chunk_text(text, max_words=200):
    import re  # Regular expressions will help us split the text more precisely

    # Use regex to tokenize the text:
    # This pattern splits the text into:
    #   - words (\w+)
    #   - whitespace (\s+)
    #   - punctuation or other non-whitespace symbols ([^\w\s])
    words = re.findall(r'\w+|\s+|[^\w\s]', text)

    chunks = []  # List to store the resulting text chunks
    chunk = []   # Temporary buffer to build up each chunk

    # Iterate through each token (word, space, or punctuation)
    for word in words:
        chunk.append(word)  # Add token to the current chunk
        if len(chunk) >= max_words:
            # Once we reach the max word count, join tokens into a string and store the chunk
            chunks.append("".join(chunk))  # Use "".join() to preserve punctuation/spacing
            chunk = []  # Reset for the next chunk

    # If there's leftover content after the loop, add the final chunk
    if chunk:
        chunks.append("".join(chunk))

    return chunks  # Return list of chunks
```

We then apply our chunking function to the corpus.


```python
# Apply the chunking function to your full text file
chunks = chunk_text(file_contents, max_words=200)

# Show how many chunks were created
print(f"Number of chunks: {len(chunks)}")

# Preview one of the chunks (by index)
chunk_ex_ind = 1  # Feel free to change this number to explore different parts of the text
print(f"Chunk {chunk_ex_ind} \n{chunks[chunk_ex_ind]}")
```

    Number of chunks: 291
    Chunk 1 
     packed into a "cliche". . .this is the
    original meaning of the term cliche. . .and thus, being unwilling
    to unpack the cliches, and thus you will see some substitutions
    that look very odd. . .such as the exchanges of u for v, v for u,
    above. . .and you may wonder why they did it this way, presuming
    Shakespeare did not actually write the play in this manner. . . .
    
    The answer is that they MAY have packed "liue" into a cliche at a
    time when they were out



```python
import json

# Convert chunks into JSONL for the Estimator input job
with open("chunks.jsonl", "w", encoding="utf-8") as f:
    for ch in chunks:      # or speaker_chunks if preferred
        f.write(json.dumps({"text": ch}, ensure_ascii=False) + "\n")

print("Wrote", len(chunks), "chunks to chunks.jsonl")

```

    Wrote 291 chunks to chunks.jsonl



```python
import json
import boto3
import sagemaker
import os

from sagemaker.huggingface import HuggingFace

# init session with region explicitly defined
region = "us-east-1" # United States (Virginia). Make sure this matches what you see near top right of AWS Console menu
boto_session = boto3.Session(region_name=region) # Create a Boto3 session that ensures all AWS service calls (including SageMaker) use the specified region
session = sagemaker.Session(boto_session=boto_session)

# print role
role = sagemaker.get_execution_role()
print(f'role = {role}')


s3_client = boto3.client("s3")

bucket_name = "chris-rag"          # <-- you choose this
prefix = "chunks/"                   # logical folder
local_file = "chunks.jsonl"

# Upload to S3
s3_uri = f"s3://{bucket_name}/{prefix}chunks.jsonl"
s3_client.upload_file(local_file, bucket_name, f"{prefix}chunks.jsonl")

print("Uploaded:", s3_uri)

```

    INFO:botocore.credentials:Found credentials from IAM Role: BaseNotebookInstanceEc2InstanceRole


    role = arn:aws:iam::183295408236:role/ml-sagemaker-use
    Uploaded: s3://chris-rag/chunks/chunks.jsonl



```python
# sagemaker assigns automatic default bucket per account where job artifacts get stored (zipped code)
session.default_bucket()
```




    'sagemaker-us-east-1-183295408236'




```python
# List and print all objects in the bucket
response = s3_client.list_objects_v2(Bucket=bucket_name)

# Check if there are objects in the bucket
if 'Contents' in response:
    for obj in response['Contents']:
        print(obj['Key'])  # Print the object's key (its path in the bucket)
else:
    print("The bucket is empty or does not exist.")
```

    chunks/chunks.jsonl
    embed/embeddings.npy


## Step 3: Embed chunks with sentence transformers

To enable semantic search, we need to convert our text chunks into numerical vectors—high-dimensional representations that capture meaning beyond simple keyword overlap. This process is called *embedding*, and it allows us to compare the semantic similarity between a user's question and the contents of a document.

This is done using an **encoder-only transformer model**. Unlike decoder or encoder-decoder models, encoder-only models are not designed to generate text. Instead, they are optimized for understanding input sequences and producing meaningful vector representations. These models take in text and output fixed-size embeddings that capture semantic content—ideal for tasks like search, retrieval, and clustering.

We'll use:

- The [`sentence-transformers`](https://www.sbert.net/) library  
  - A widely used library that wraps encoder-only transformer models for generating sentence- and paragraph-level embeddings.
  - It provides a simple interface (`model.encode()`) and is optimized for performance and batching, making it well-suited for retrieval-augmented generation (RAG) workflows.
  - It supports both short queries and longer document chunks, embedding them into the same shared vector space.

- A pretrained model: [`multi-qa-MiniLM-L6-cos-v1`](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1)  
  - A compact encoder-only model (6 layers) designed for semantic search and question answering.
  - Trained using contrastive learning on query-passage pairs, so it learns to embed related questions and answers close together in vector space.
  - It's efficient enough to run on CPUs or entry-level GPUs, making it great for experimentation and prototyping.

### Why embeddings matter in RAG

In a RAG system, embeddings are the foundation for connecting a user's question to the most relevant content in your corpus.

Rather than relying on exact keyword matches, embeddings represent both queries and document chunks in the same semantic space. When a user asks a question, we:

1. Convert the user's question into a vector using the same encoder-only embedding model that was used to encode the document chunks.
2. Compute similarity scores (e.g., cosine similarity) between the query vector and each chunk vector.
3. Retrieve the top-matching chunks to pass along as context to the language model.

This allows the system to surface text that is meaningfully related to the question—even if it doesn't use the same words. For example, a question like "*What does Juliet think of Romeo?*" might retrieve a passage describing her inner turmoil or emotional reaction, even if the words "think" or "Romeo" aren't explicitly present. Embedding-based retrieval improves relevance, flexibility, and ultimately the quality of the answers your language model can generate.

### (Optional) SageMaker training & deployment with Hugging Face Estimators

If you want to **train or fine‑tune** the embedding and generation models on SageMaker
and then deploy them as endpoints, you can use the Hugging Face `Estimator` API.

The next cell shows minimal skeletons for:

- an **embedding Estimator** (feature extraction / sentence embeddings)
- a **generation Estimator** (causal LM for text generation)

You can skip this cell if you already have endpoints deployed and just want to call them.


**Note**: had to create requirments.txt file and place in code direction to add sentence-transformers to env. Weird that this estimator doesn't come with that??

#### get notebook tags so we can tag processing jobs (we should apply this to training jobs as well!)


```python
prefix = 'chunks'
```


```python
from sagemaker.huggingface import HuggingFaceProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

script_path = "embedding_inference.py"
embed_model = "all-mpnet-base-v2"
base_job_name = f"RAG-embedding-{embed_model}"

input_path = f"s3://{bucket_name}/{prefix}/chunks.jsonl"
output_path = f"s3://{bucket_name}/embed/"

instance_type = "ml.g5.xlarge"
instance_count = 1
max_run = 2 * 60 * 60

```


```python
import json
import boto3
import os

sm_client = boto3.client("sagemaker")

def get_notebook_instance_tags():
    # 1. Get the instance name from the metadata file
    meta_path = "/opt/ml/metadata/resource-metadata.json"
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found at {meta_path}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    # This key name is typical; adjust if AWS changes format
    notebook_instance_name = meta.get("ResourceName")
    if not notebook_instance_name:
        raise RuntimeError("Could not find notebook instance name in metadata")

    # 2. Describe notebook to get its ARN
    desc = sm_client.describe_notebook_instance(
        NotebookInstanceName=notebook_instance_name
    )
    arn = desc["NotebookInstanceArn"]

    # 3. List tags on the notebook instance
    resp = sm_client.list_tags(ResourceArn=arn)
    return resp.get("Tags", [])


def filter_project_tags(tags):
    wanted_keys = {"Project", "Name", "Purpose"}
    return [t for t in tags if t["Key"] in wanted_keys]

base_tags = filter_project_tags(get_notebook_instance_tags())

print(base_tags)


```

    [{'Key': 'Project', 'Value': 'NA'}, {'Key': 'Purpose', 'Value': 'Test Rag and Bedrock'}, {'Key': 'Name', 'Value': 'Chris Endemann'}]



```python
# Replace Purpose tag with job-specific purpose
embedding_tags = [
    t if t["Key"] != "Purpose" else {"Key": "Purpose", "Value": base_job_name}
    for t in base_tags
]
print(embedding_tags)
```

    [{'Key': 'Project', 'Value': 'NA'}, {'Key': 'Purpose', 'Value': 'RAG-embedding-all-mpnet-base-v2'}, {'Key': 'Name', 'Value': 'Chris Endemann'}]



```python

embedding_processor = HuggingFaceProcessor(
    base_job_name=base_job_name,
    role=role,
    instance_type=instance_type,
    instance_count=instance_count,
    transformers_version="4.56",
    pytorch_version="2.8",
    py_version="py312",
    sagemaker_session=session,
    max_runtime_in_seconds=max_run,
    tags=embedding_tags,
)


```

    INFO:sagemaker.image_uris:image_uri is not presented, retrieving image_uri based on instance_type, framework etc.



```python
def print_processor_config(p):
    print("=== Processor Config ===")
    print(f"base_job_name:       {p.base_job_name}")
    print(f"role:                {p.role}")
    print(f"instance_type:       {p.instance_type}")
    print(f"instance_count:      {p.instance_count}")
    print(f"max_runtime_seconds: {p.max_runtime_in_seconds}")
    print(f"pytorch_version:     {p.pytorch_version}")
    print(f"py_version:          {p.py_version}")
    print(f"tags:                {p.tags}")
    print(f"volume_size_gb:      {p.volume_size_in_gb}")

print_processor_config(embedding_processor)

print(f"input path: {input_path}")
print(f"output path: {output_path}")
```

    === Processor Config ===
    base_job_name:       RAG-embedding-all-mpnet-base-v2
    role:                arn:aws:iam::183295408236:role/ml-sagemaker-use
    instance_type:       ml.g5.xlarge
    instance_count:      1
    max_runtime_seconds: 7200
    pytorch_version:     2.8
    py_version:          py312
    tags:                [{'Key': 'Project', 'Value': 'NA'}, {'Key': 'Purpose', 'Value': 'RAG-embedding-all-mpnet-base-v2'}, {'Key': 'Name', 'Value': 'Chris Endemann'}]
    volume_size_gb:      30
    input path: s3://chris-rag/chunks/chunks.jsonl
    output path: s3://chris-rag/embed/



```python
embedding_processor.run(
    # This is the script SageMaker will execute *inside the processing container*.
    # It will be placed in /opt/ml/processing/input/code (automatically by SageMaker)
    # when you pass code= and source_dir=.
    code=script_path,

    # Folder on your local machine/notebook that contains:
    #   - embedding_inference.py
    #   - requirements.txt  (optional)
    #
    # SageMaker will TAR this directory and upload it to the default SageMaker bucket,
    # then unpack it *inside the container* at:
    #   /opt/ml/processing/input/code/
    source_dir="code/",

    # ---------------------
    # PROCESSING INPUTS
    # ---------------------
    inputs=[
        ProcessingInput(
            # Your input data in YOUR bucket.
            # This path must exist in S3:
            #   s3://<bucket>/<prefix>/chunks.jsonl
            source=input_path,

            # This is NOT your bucket — this is a fixed path inside the container.
            # SageMaker automatically mounts the S3 object here.
            #
            # Everything in /opt/ml/processing/* is created by SageMaker at runtime.
            # You do NOT create or manage these directories yourself.
            destination="/opt/ml/processing/input",
        )
    ],

    # ---------------------
    # PROCESSING OUTPUTS
    # ---------------------
    outputs=[
        ProcessingOutput(
            output_name="embeddings",

            # Again, /opt/ml/processing/output is a container-local directory.
            # Your script will write embeddings to this folder.
            # SageMaker will automatically upload anything here to the S3 destination below.
            source="/opt/ml/processing/output",

            # This IS your bucket/prefix.
            # All output artifacts will land at:
            #   s3://<bucket>/<prefix>/embeddings/
            destination=output_path,
        )
    ],

    # ---------------------
    # SCRIPT ARGUMENTS
    # Passed to embedding_inference.py
    # ---------------------
    arguments=[
        "--model_id", f"sentence-transformers/{embed_model}",

        # The input filename we expect inside the container-mounted input directory:
        "--input_filename", "chunks.jsonl",

        # JSON key to pull the text from:
        "--text_key", "text",

        # Matches the ProcessingInput destination above
        "--input_dir", "/opt/ml/processing/input",

        # Matches the ProcessingOutput source above
        "--output_dir", "/opt/ml/processing/output",
    ],
)

```

    INFO:sagemaker.processing:Uploaded code/ to s3://sagemaker-us-east-1-183295408236/RAG-embedding-all-mpnet-base-v2-2025-11-21-18-48-22-921/source/sourcedir.tar.gz
    INFO:sagemaker.processing:runproc.sh uploaded to s3://sagemaker-us-east-1-183295408236/RAG-embedding-all-mpnet-base-v2-2025-11-21-18-48-22-921/source/runproc.sh
    INFO:sagemaker:Creating processing-job with name RAG-embedding-all-mpnet-base-v2-2025-11-21-18-48-22-921


    .....................................[34mCodeArtifact repository not specified. Skipping login.[0m
    [34mFound existing installation: typing 3.7.4.3[0m
    [34mUninstalling typing-3.7.4.3:
      Successfully uninstalled typing-3.7.4.3[0m
    [34mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.[0m
    [34mCollecting sentence-transformers (from -r requirements.txt (line 1))
      Downloading sentence_transformers-5.1.2-py3-none-any.whl.metadata (16 kB)[0m
    [34mCollecting faiss-cpu (from -r requirements.txt (line 2))
      Downloading faiss_cpu-1.13.0-cp39-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (7.7 kB)[0m
    [34mRequirement already satisfied: transformers<5.0.0,>=4.41.0 in /usr/local/lib/python3.12/site-packages (from sentence-transformers->-r requirements.txt (line 1)) (4.56.2)[0m
    [34mRequirement already satisfied: tqdm in /usr/local/lib/python3.12/site-packages (from sentence-transformers->-r requirements.txt (line 1)) (4.67.1)[0m
    [34mRequirement already satisfied: torch>=1.11.0 in /usr/local/lib/python3.12/site-packages (from sentence-transformers->-r requirements.txt (line 1)) (2.8.0+cu129)[0m
    [34mRequirement already satisfied: scikit-learn in /usr/local/lib/python3.12/site-packages (from sentence-transformers->-r requirements.txt (line 1)) (1.7.2)[0m
    [34mRequirement already satisfied: scipy in /usr/local/lib/python3.12/site-packages (from sentence-transformers->-r requirements.txt (line 1)) (1.16.2)[0m
    [34mRequirement already satisfied: huggingface-hub>=0.20.0 in /usr/local/lib/python3.12/site-packages (from sentence-transformers->-r requirements.txt (line 1)) (0.35.3)[0m
    [34mRequirement already satisfied: Pillow in /usr/local/lib/python3.12/site-packages (from sentence-transformers->-r requirements.txt (line 1)) (11.3.0)[0m
    [34mRequirement already satisfied: typing_extensions>=4.5.0 in /usr/local/lib/python3.12/site-packages (from sentence-transformers->-r requirements.txt (line 1)) (4.15.0)[0m
    [34mRequirement already satisfied: filelock in /usr/local/lib/python3.12/site-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers->-r requirements.txt (line 1)) (3.13.1)[0m
    [34mRequirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.12/site-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers->-r requirements.txt (line 1)) (1.26.4)[0m
    [34mRequirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.12/site-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers->-r requirements.txt (line 1)) (24.2)[0m
    [34mRequirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.12/site-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers->-r requirements.txt (line 1)) (6.0.2)[0m
    [34mRequirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.12/site-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers->-r requirements.txt (line 1)) (2025.9.18)[0m
    [34mRequirement already satisfied: requests in /usr/local/lib/python3.12/site-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers->-r requirements.txt (line 1)) (2.32.5)[0m
    [34mRequirement already satisfied: tokenizers<=0.23.0,>=0.22.0 in /usr/local/lib/python3.12/site-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers->-r requirements.txt (line 1)) (0.22.1)[0m
    [34mRequirement already satisfied: safetensors>=0.4.3 in /usr/local/lib/python3.12/site-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers->-r requirements.txt (line 1)) (0.6.2)[0m
    [34mRequirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.12/site-packages (from huggingface-hub>=0.20.0->sentence-transformers->-r requirements.txt (line 1)) (2025.9.0)[0m
    [34mRequirement already satisfied: hf-xet<2.0.0,>=1.1.3 in /usr/local/lib/python3.12/site-packages (from huggingface-hub>=0.20.0->sentence-transformers->-r requirements.txt (line 1)) (1.1.10)[0m
    [34mRequirement already satisfied: setuptools in /usr/local/lib/python3.12/site-packages (from torch>=1.11.0->sentence-transformers->-r requirements.txt (line 1)) (80.9.0)[0m
    [34mRequirement already satisfied: sympy>=1.13.3 in /usr/local/lib/python3.12/site-packages (from torch>=1.11.0->sentence-transformers->-r requirements.txt (line 1)) (1.13.3)[0m
    [34mRequirement already satisfied: networkx in /usr/local/lib/python3.12/site-packages (from torch>=1.11.0->sentence-transformers->-r requirements.txt (line 1)) (3.3)[0m
    [34mRequirement already satisfied: jinja2 in /usr/local/lib/python3.12/site-packages (from torch>=1.11.0->sentence-transformers->-r requirements.txt (line 1)) (3.1.6)[0m
    [34mRequirement already satisfied: nvidia-cuda-nvrtc-cu12==12.9.86 in /usr/local/lib/python3.12/site-packages (from torch>=1.11.0->sentence-transformers->-r requirements.txt (line 1)) (12.9.86)[0m
    [34mRequirement already satisfied: nvidia-cuda-runtime-cu12==12.9.79 in /usr/local/lib/python3.12/site-packages (from torch>=1.11.0->sentence-transformers->-r requirements.txt (line 1)) (12.9.79)[0m
    [34mRequirement already satisfied: nvidia-cuda-cupti-cu12==12.9.79 in /usr/local/lib/python3.12/site-packages (from torch>=1.11.0->sentence-transformers->-r requirements.txt (line 1)) (12.9.79)[0m
    [34mRequirement already satisfied: nvidia-cudnn-cu12==9.10.2.21 in /usr/local/lib/python3.12/site-packages (from torch>=1.11.0->sentence-transformers->-r requirements.txt (line 1)) (9.10.2.21)[0m
    [34mRequirement already satisfied: nvidia-cublas-cu12==12.9.1.4 in /usr/local/lib/python3.12/site-packages (from torch>=1.11.0->sentence-transformers->-r requirements.txt (line 1)) (12.9.1.4)[0m
    [34mRequirement already satisfied: nvidia-cufft-cu12==11.4.1.4 in /usr/local/lib/python3.12/site-packages (from torch>=1.11.0->sentence-transformers->-r requirements.txt (line 1)) (11.4.1.4)[0m
    [34mRequirement already satisfied: nvidia-curand-cu12==10.3.10.19 in /usr/local/lib/python3.12/site-packages (from torch>=1.11.0->sentence-transformers->-r requirements.txt (line 1)) (10.3.10.19)[0m
    [34mRequirement already satisfied: nvidia-cusolver-cu12==11.7.5.82 in /usr/local/lib/python3.12/site-packages (from torch>=1.11.0->sentence-transformers->-r requirements.txt (line 1)) (11.7.5.82)[0m
    [34mRequirement already satisfied: nvidia-cusparse-cu12==12.5.10.65 in /usr/local/lib/python3.12/site-packages (from torch>=1.11.0->sentence-transformers->-r requirements.txt (line 1)) (12.5.10.65)[0m
    [34mRequirement already satisfied: nvidia-cusparselt-cu12==0.7.1 in /usr/local/lib/python3.12/site-packages (from torch>=1.11.0->sentence-transformers->-r requirements.txt (line 1)) (0.7.1)[0m
    [34mRequirement already satisfied: nvidia-nccl-cu12==2.27.3 in /usr/local/lib/python3.12/site-packages (from torch>=1.11.0->sentence-transformers->-r requirements.txt (line 1)) (2.27.3)[0m
    [34mRequirement already satisfied: nvidia-nvtx-cu12==12.9.79 in /usr/local/lib/python3.12/site-packages (from torch>=1.11.0->sentence-transformers->-r requirements.txt (line 1)) (12.9.79)[0m
    [34mRequirement already satisfied: nvidia-nvjitlink-cu12==12.9.86 in /usr/local/lib/python3.12/site-packages (from torch>=1.11.0->sentence-transformers->-r requirements.txt (line 1)) (12.9.86)[0m
    [34mRequirement already satisfied: nvidia-cufile-cu12==1.14.1.1 in /usr/local/lib/python3.12/site-packages (from torch>=1.11.0->sentence-transformers->-r requirements.txt (line 1)) (1.14.1.1)[0m
    [34mRequirement already satisfied: triton==3.4.0 in /usr/local/lib/python3.12/site-packages (from torch>=1.11.0->sentence-transformers->-r requirements.txt (line 1)) (3.4.0)[0m
    [34mRequirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.12/site-packages (from sympy>=1.13.3->torch>=1.11.0->sentence-transformers->-r requirements.txt (line 1)) (1.3.0)[0m
    [34mRequirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.12/site-packages (from jinja2->torch>=1.11.0->sentence-transformers->-r requirements.txt (line 1)) (3.0.3)[0m
    [34mRequirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.12/site-packages (from requests->transformers<5.0.0,>=4.41.0->sentence-transformers->-r requirements.txt (line 1)) (3.4.3)[0m
    [34mRequirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.12/site-packages (from requests->transformers<5.0.0,>=4.41.0->sentence-transformers->-r requirements.txt (line 1)) (3.10)[0m
    [34mRequirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.12/site-packages (from requests->transformers<5.0.0,>=4.41.0->sentence-transformers->-r requirements.txt (line 1)) (2.5.0)[0m
    [34mRequirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.12/site-packages (from requests->transformers<5.0.0,>=4.41.0->sentence-transformers->-r requirements.txt (line 1)) (2025.8.3)[0m
    [34mRequirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.12/site-packages (from scikit-learn->sentence-transformers->-r requirements.txt (line 1)) (1.5.2)[0m
    [34mRequirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.12/site-packages (from scikit-learn->sentence-transformers->-r requirements.txt (line 1)) (3.6.0)[0m
    [34mDownloading sentence_transformers-5.1.2-py3-none-any.whl (488 kB)[0m
    [34mDownloading faiss_cpu-1.13.0-cp39-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (23.6 MB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 23.6/23.6 MB 153.3 MB/s  0:00:00[0m
    [34mInstalling collected packages: faiss-cpu, sentence-transformers[0m
    [34mSuccessfully installed faiss-cpu-1.13.0 sentence-transformers-5.1.2[0m
    [34mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.[0m
    [34m[notice] A new release of pip is available: 25.2 -> 25.3[0m
    [34m[notice] To update, run: pip install --upgrade pip[0m
    [34m/usr/local/lib/python3.12/site-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
      import pynvml  # type: ignore[import][0m
    [34m#015Batches:   0%|          | 0/10 [00:00<?, ?it/s]#015Batches:  10%|█         | 1/10 [00:00<00:02,  3.74it/s]#015Batches:  30%|███       | 3/10 [00:00<00:00,  7.94it/s]#015Batches:  50%|█████     | 5/10 [00:00<00:00, 10.04it/s]#015Batches:  70%|███████   | 7/10 [00:00<00:00, 11.04it/s]#015Batches:  90%|█████████ | 9/10 [00:00<00:00, 11.83it/s]#015Batches: 100%|██████████| 10/10 [00:00<00:00, 11.23it/s][0m
    [34mSaved embeddings to: /opt/ml/processing/output/embeddings.npy[0m
    [34mSaved FAISS index to: /opt/ml/processing/output/faiss.index[0m
    [34mSaved IDs to: /opt/ml/processing/output/ids.jsonl[0m
    



```python
# List and print all objects in the bucket
response = s3_client.list_objects_v2(Bucket=bucket_name)

# Check if there are objects in the bucket
if 'Contents' in response:
    for obj in response['Contents']:
        print(obj['Key'])  # Print the object's key (its path in the bucket)
else:
    print("The bucket is empty or does not exist.")
```

    chunks/chunks.jsonl
    embed/embeddings.npy
    embed/faiss.index
    embed/ids.jsonl



- **chunks/chunks.jsonl** — Line-delimited JSON file containing the original text chunks. The line order matches the embedding order.

- **embed/embeddings.npy** — NumPy array of all chunk embeddings with shape `[num_chunks, dim]`. Row *i* corresponds to chunk *i* in `chunks.jsonl`.

- **embed/faiss.index** — Serialized FAISS index built from the embeddings, used for fast top-k similarity search.

- **embed/ids.jsonl** — Line-delimited JSON file mapping each embedding row to an integer ID for easy reference in downstream steps.

### Callout: vector databases

After generating embeddings, you need a way to perform fast similarity search. FAISS (Facebook AI Similarity Search) is a popular open-source library from Meta for efficient vector search and nearest-neighbor lookup; it works well when your whole index can fit in memory and you're running retrieval inside a notebook or small API. For larger or multi-user systems, you might use a managed vector database instead. These become useful when you need concurrent queries, metadata filtering, persistent storage, or indexes that exceed a single machine’s memory. In those cases, OpenSearch with k-NN, Aurora PostgreSQL with pgvector, or Bedrock Knowledge Bases can provide scalable vector search at the cost of additional setup and ongoing management. For most projects, storing `embeddings.npy` in S3 and using FAISS or a similar local index is usually enough, especially when the corpus is small and you mainly need retrieval inside a notebook or simple API.

### Similarity search and generation

First, we'll save out a file containing a batch of queries we want to test. We'll upload this to our bucket.




```python
import json
import os
import boto3

# Assumes your corpus in chunks/chunks.jsonl is Romeo and Juliet text

queries = [
    {"id": 1, "query": "Summarize the main conflict in Romeo and Juliet."},
    {"id": 2, "query": "Who are the key characters in Romeo and Juliet, and how are they related?"},
    {"id": 3, "query": "What events lead to the deaths of Romeo and Juliet?"},
    {"id": 4, "query": "How does the feud between the Montagues and Capulets drive the plot?"},
    {"id": 5, "query": "What themes about love and fate appear in Romeo and Juliet?"},
    {"id": 6, "query": "Who kills Mercutio?"}, # Tybalt kills Mercutio
]

local_queries_path = "queries.jsonl"

with open(local_queries_path, "w", encoding="utf-8") as f:
    for q in queries:
        f.write(json.dumps(q, ensure_ascii=False) + "\n")

print(f"Wrote {len(queries)} queries to {local_queries_path}")

s3 = boto3.client("s3")

# reuse your existing bucket_name and prefix
queries_s3_key = f"{prefix}/queries/queries.jsonl"
s3.upload_file(local_queries_path, bucket_name, queries_s3_key)

queries_s3_uri = f"s3://{bucket_name}/{queries_s3_key}"
print("Uploaded Romeo & Juliet queries.jsonl to:")
print(queries_s3_uri)

```

    Wrote 6 queries to queries.jsonl
    Uploaded Romeo & Juliet queries.jsonl to:
    s3://chris-rag/chunks/queries/queries.jsonl



```python
# List and print all objects in the bucket
response = s3_client.list_objects_v2(Bucket=bucket_name)

# Check if there are objects in the bucket
if 'Contents' in response:
    for obj in response['Contents']:
        print(obj['Key'])  # Print the object's key (its path in the bucket)
else:
    print("The bucket is empty or does not exist.")
```

    chunks/chunks.jsonl
    chunks/queries/queries.jsonl
    embed/embeddings.npy
    embed/faiss.index
    embed/ids.jsonl



```python
from sagemaker.huggingface import HuggingFaceProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

# ------------------------------------------------------------------
# Paths we already have / will create
# ------------------------------------------------------------------

# Existing from embedding step:
# input_path  -> s3://bucket/prefix/chunks.jsonl
# output_path -> s3://bucket/prefix/embeddings/  (contains embeddings.npy, faiss.index)

# New: where queries live and where to put batch RAG outputs
queries_s3_uri = f"s3://{bucket_name}/{prefix}/queries/queries.jsonl"
batch_output_path = f"s3://{bucket_name}/{prefix}/answers/"

# Script name inside your local "code/" folder
batch_script_path = "batch_generation.py"
gen_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
# ------------------------------------------------------------------
# Processor for batch generation
# ------------------------------------------------------------------

batch_job_name = "rag-batch-generation-job"

generation_processor = HuggingFaceProcessor(
    base_job_name=batch_job_name,
    role=role,
    instance_type=instance_type,      # same as embedding job for now
    instance_count=instance_count,
    transformers_version="4.56",
    pytorch_version="2.8",
    py_version="py312",
    sagemaker_session=session,
    max_runtime_in_seconds=max_run,
    tags=embedding_tags,              # or define a new generation_tags with Purpose="batch_generation"
)

# Optional sanity check
print_processor_config(generation_processor)
print(f"corpus path:   {input_path}")
print(f"emb/index dir: {output_path}")
print(f"queries path:  {queries_s3_uri}")
print(f"batch output:  {batch_output_path}")


```

    INFO:sagemaker.image_uris:image_uri is not presented, retrieving image_uri based on instance_type, framework etc.


    === Processor Config ===
    base_job_name:       rag-batch-generation-job
    role:                arn:aws:iam::183295408236:role/ml-sagemaker-use
    instance_type:       ml.g5.xlarge
    instance_count:      1
    max_runtime_seconds: 7200
    pytorch_version:     2.8
    py_version:          py312
    tags:                [{'Key': 'Project', 'Value': 'NA'}, {'Key': 'Purpose', 'Value': 'RAG-embedding-all-mpnet-base-v2'}, {'Key': 'Name', 'Value': 'Chris Endemann'}]
    volume_size_gb:      30
    corpus path:   s3://chris-rag/chunks/chunks.jsonl
    emb/index dir: s3://chris-rag/embed/
    queries path:  s3://chris-rag/chunks/queries/queries.jsonl
    batch output:  s3://chris-rag/chunks/answers/



```python

embed_path = 's3://chris-rag/embed/'

```


```python
# ------------------------------------------------------------------
# Run batch RAG generation
# ------------------------------------------------------------------

generation_processor.run(
    code=batch_script_path,
    source_dir="code/",  # contains batch_generation.py + requirements.txt

    inputs=[
        # Corpus (chunks.jsonl)
        ProcessingInput(
            source=input_path,
            destination="/opt/ml/processing/corpus",
        ),
        # Embeddings matrix
        ProcessingInput(
            source=f"{embed_path}embeddings.npy",
            destination="/opt/ml/processing/emb",
        ),
        # FAISS index
        ProcessingInput(
            source=f"{embed_path}faiss.index",
            destination="/opt/ml/processing/faiss",
        ),
        # Queries
        ProcessingInput(
            source=queries_s3_uri,
            destination="/opt/ml/processing/queries",
        ),
    ],

    outputs=[
        ProcessingOutput(
            output_name="batch_results",
            source="/opt/ml/processing/output",
            destination=batch_output_path,
        )
    ],

    arguments=[
        # Models
        "--embed_model_id", f"sentence-transformers/{embed_model}",
        "--gen_model_id", gen_model_id,  # e.g. "mistralai/Mistral-7B-Instruct-v0.2"

        # Filenames *relative to* the input_dir we’ll set below
        "--corpus_filename", "corpus/chunks.jsonl",
        "--corpus_text_key", "text",

        "--queries_filename", "queries/queries.jsonl",
        "--query_key", "query",

        "--index_filename", "faiss/faiss.index",
        "--embeddings_filename", "emb/embeddings.npy",

        # Root I/O dirs inside the container
        "--input_dir", "/opt/ml/processing",
        "--output_dir", "/opt/ml/processing/output",

        # Retrieval + generation knobs
        "--top_k", "5",
        "--max_new_tokens", "128",
        "--temperature", "0.0",
    ],
)

```

    INFO:sagemaker.processing:Uploaded code/ to s3://sagemaker-us-east-1-183295408236/rag-batch-generation-job-2025-11-21-19-25-22-254/source/sourcedir.tar.gz
    INFO:sagemaker.processing:runproc.sh uploaded to s3://sagemaker-us-east-1-183295408236/rag-batch-generation-job-2025-11-21-19-25-22-254/source/runproc.sh
    INFO:sagemaker:Creating processing-job with name rag-batch-generation-job-2025-11-21-19-25-22-254


    .......................................[34mCodeArtifact repository not specified. Skipping login.[0m
    [34mFound existing installation: typing 3.7.4.3[0m
    [34mUninstalling typing-3.7.4.3:
      Successfully uninstalled typing-3.7.4.3[0m
    [34mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.[0m
    [34mCollecting sentence-transformers (from -r requirements.txt (line 1))
      Downloading sentence_transformers-5.1.2-py3-none-any.whl.metadata (16 kB)[0m
    [34mCollecting faiss-cpu (from -r requirements.txt (line 2))
      Downloading faiss_cpu-1.13.0-cp39-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (7.7 kB)[0m
    [34mRequirement already satisfied: transformers<5.0.0,>=4.41.0 in /usr/local/lib/python3.12/site-packages (from sentence-transformers->-r requirements.txt (line 1)) (4.56.2)[0m
    [34mRequirement already satisfied: tqdm in /usr/local/lib/python3.12/site-packages (from sentence-transformers->-r requirements.txt (line 1)) (4.67.1)[0m
    [34mRequirement already satisfied: torch>=1.11.0 in /usr/local/lib/python3.12/site-packages (from sentence-transformers->-r requirements.txt (line 1)) (2.8.0+cu129)[0m
    [34mRequirement already satisfied: scikit-learn in /usr/local/lib/python3.12/site-packages (from sentence-transformers->-r requirements.txt (line 1)) (1.7.2)[0m
    [34mRequirement already satisfied: scipy in /usr/local/lib/python3.12/site-packages (from sentence-transformers->-r requirements.txt (line 1)) (1.16.2)[0m
    [34mRequirement already satisfied: huggingface-hub>=0.20.0 in /usr/local/lib/python3.12/site-packages (from sentence-transformers->-r requirements.txt (line 1)) (0.35.3)[0m
    [34mRequirement already satisfied: Pillow in /usr/local/lib/python3.12/site-packages (from sentence-transformers->-r requirements.txt (line 1)) (11.3.0)[0m
    [34mRequirement already satisfied: typing_extensions>=4.5.0 in /usr/local/lib/python3.12/site-packages (from sentence-transformers->-r requirements.txt (line 1)) (4.15.0)[0m
    [34mRequirement already satisfied: filelock in /usr/local/lib/python3.12/site-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers->-r requirements.txt (line 1)) (3.13.1)[0m
    [34mRequirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.12/site-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers->-r requirements.txt (line 1)) (1.26.4)[0m
    [34mRequirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.12/site-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers->-r requirements.txt (line 1)) (24.2)[0m
    [34mRequirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.12/site-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers->-r requirements.txt (line 1)) (6.0.2)[0m
    [34mRequirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.12/site-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers->-r requirements.txt (line 1)) (2025.9.18)[0m
    [34mRequirement already satisfied: requests in /usr/local/lib/python3.12/site-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers->-r requirements.txt (line 1)) (2.32.5)[0m
    [34mRequirement already satisfied: tokenizers<=0.23.0,>=0.22.0 in /usr/local/lib/python3.12/site-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers->-r requirements.txt (line 1)) (0.22.1)[0m
    [34mRequirement already satisfied: safetensors>=0.4.3 in /usr/local/lib/python3.12/site-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers->-r requirements.txt (line 1)) (0.6.2)[0m
    [34mRequirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.12/site-packages (from huggingface-hub>=0.20.0->sentence-transformers->-r requirements.txt (line 1)) (2025.9.0)[0m
    [34mRequirement already satisfied: hf-xet<2.0.0,>=1.1.3 in /usr/local/lib/python3.12/site-packages (from huggingface-hub>=0.20.0->sentence-transformers->-r requirements.txt (line 1)) (1.1.10)[0m
    [34mRequirement already satisfied: setuptools in /usr/local/lib/python3.12/site-packages (from torch>=1.11.0->sentence-transformers->-r requirements.txt (line 1)) (80.9.0)[0m
    [34mRequirement already satisfied: sympy>=1.13.3 in /usr/local/lib/python3.12/site-packages (from torch>=1.11.0->sentence-transformers->-r requirements.txt (line 1)) (1.13.3)[0m
    [34mRequirement already satisfied: networkx in /usr/local/lib/python3.12/site-packages (from torch>=1.11.0->sentence-transformers->-r requirements.txt (line 1)) (3.3)[0m
    [34mRequirement already satisfied: jinja2 in /usr/local/lib/python3.12/site-packages (from torch>=1.11.0->sentence-transformers->-r requirements.txt (line 1)) (3.1.6)[0m
    [34mRequirement already satisfied: nvidia-cuda-nvrtc-cu12==12.9.86 in /usr/local/lib/python3.12/site-packages (from torch>=1.11.0->sentence-transformers->-r requirements.txt (line 1)) (12.9.86)[0m
    [34mRequirement already satisfied: nvidia-cuda-runtime-cu12==12.9.79 in /usr/local/lib/python3.12/site-packages (from torch>=1.11.0->sentence-transformers->-r requirements.txt (line 1)) (12.9.79)[0m
    [34mRequirement already satisfied: nvidia-cuda-cupti-cu12==12.9.79 in /usr/local/lib/python3.12/site-packages (from torch>=1.11.0->sentence-transformers->-r requirements.txt (line 1)) (12.9.79)[0m
    [34mRequirement already satisfied: nvidia-cudnn-cu12==9.10.2.21 in /usr/local/lib/python3.12/site-packages (from torch>=1.11.0->sentence-transformers->-r requirements.txt (line 1)) (9.10.2.21)[0m
    [34mRequirement already satisfied: nvidia-cublas-cu12==12.9.1.4 in /usr/local/lib/python3.12/site-packages (from torch>=1.11.0->sentence-transformers->-r requirements.txt (line 1)) (12.9.1.4)[0m
    [34mRequirement already satisfied: nvidia-cufft-cu12==11.4.1.4 in /usr/local/lib/python3.12/site-packages (from torch>=1.11.0->sentence-transformers->-r requirements.txt (line 1)) (11.4.1.4)[0m
    [34mRequirement already satisfied: nvidia-curand-cu12==10.3.10.19 in /usr/local/lib/python3.12/site-packages (from torch>=1.11.0->sentence-transformers->-r requirements.txt (line 1)) (10.3.10.19)[0m
    [34mRequirement already satisfied: nvidia-cusolver-cu12==11.7.5.82 in /usr/local/lib/python3.12/site-packages (from torch>=1.11.0->sentence-transformers->-r requirements.txt (line 1)) (11.7.5.82)[0m
    [34mRequirement already satisfied: nvidia-cusparse-cu12==12.5.10.65 in /usr/local/lib/python3.12/site-packages (from torch>=1.11.0->sentence-transformers->-r requirements.txt (line 1)) (12.5.10.65)[0m
    [34mRequirement already satisfied: nvidia-cusparselt-cu12==0.7.1 in /usr/local/lib/python3.12/site-packages (from torch>=1.11.0->sentence-transformers->-r requirements.txt (line 1)) (0.7.1)[0m
    [34mRequirement already satisfied: nvidia-nccl-cu12==2.27.3 in /usr/local/lib/python3.12/site-packages (from torch>=1.11.0->sentence-transformers->-r requirements.txt (line 1)) (2.27.3)[0m
    [34mRequirement already satisfied: nvidia-nvtx-cu12==12.9.79 in /usr/local/lib/python3.12/site-packages (from torch>=1.11.0->sentence-transformers->-r requirements.txt (line 1)) (12.9.79)[0m
    [34mRequirement already satisfied: nvidia-nvjitlink-cu12==12.9.86 in /usr/local/lib/python3.12/site-packages (from torch>=1.11.0->sentence-transformers->-r requirements.txt (line 1)) (12.9.86)[0m
    [34mRequirement already satisfied: nvidia-cufile-cu12==1.14.1.1 in /usr/local/lib/python3.12/site-packages (from torch>=1.11.0->sentence-transformers->-r requirements.txt (line 1)) (1.14.1.1)[0m
    [34mRequirement already satisfied: triton==3.4.0 in /usr/local/lib/python3.12/site-packages (from torch>=1.11.0->sentence-transformers->-r requirements.txt (line 1)) (3.4.0)[0m
    [34mRequirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.12/site-packages (from sympy>=1.13.3->torch>=1.11.0->sentence-transformers->-r requirements.txt (line 1)) (1.3.0)[0m
    [34mRequirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.12/site-packages (from jinja2->torch>=1.11.0->sentence-transformers->-r requirements.txt (line 1)) (3.0.3)[0m
    [34mRequirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.12/site-packages (from requests->transformers<5.0.0,>=4.41.0->sentence-transformers->-r requirements.txt (line 1)) (3.4.3)[0m
    [34mRequirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.12/site-packages (from requests->transformers<5.0.0,>=4.41.0->sentence-transformers->-r requirements.txt (line 1)) (3.10)[0m
    [34mRequirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.12/site-packages (from requests->transformers<5.0.0,>=4.41.0->sentence-transformers->-r requirements.txt (line 1)) (2.5.0)[0m
    [34mRequirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.12/site-packages (from requests->transformers<5.0.0,>=4.41.0->sentence-transformers->-r requirements.txt (line 1)) (2025.8.3)[0m
    [34mRequirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.12/site-packages (from scikit-learn->sentence-transformers->-r requirements.txt (line 1)) (1.5.2)[0m
    [34mRequirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.12/site-packages (from scikit-learn->sentence-transformers->-r requirements.txt (line 1)) (3.6.0)[0m
    [34mDownloading sentence_transformers-5.1.2-py3-none-any.whl (488 kB)[0m
    [34mDownloading faiss_cpu-1.13.0-cp39-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (23.6 MB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 23.6/23.6 MB 134.3 MB/s  0:00:00[0m
    [34mInstalling collected packages: faiss-cpu, sentence-transformers[0m
    [34mSuccessfully installed faiss-cpu-1.13.0 sentence-transformers-5.1.2[0m
    [34mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.[0m
    [34m[notice] A new release of pip is available: 25.2 -> 25.3[0m
    [34m[notice] To update, run: pip install --upgrade pip[0m
    [34m/usr/local/lib/python3.12/site-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
      import pynvml  # type: ignore[import][0m
    [34mLoading embeddings and FAISS index...[0m
    [34mLoading embedding model: sentence-transformers/all-mpnet-base-v2[0m
    [34mLoading generation model: mistralai/Mistral-7B-Instruct-v0.2[0m
    [34m`torch_dtype` is deprecated! Use `dtype` instead![0m
    [34m#015Loading checkpoint shards:   0%|          | 0/3 [00:00<?, ?it/s]#015Loading checkpoint shards:  67%|██████▋   | 2/3 [00:00<00:00,  8.18it/s]#015Loading checkpoint shards: 100%|██████████| 3/3 [00:00<00:00, 10.94it/s][0m
    [34mEncoding 6 queries...[0m
    [34m#015Batches:   0%|          | 0/1 [00:00<?, ?it/s]#015Batches: 100%|██████████| 1/1 [00:00<00:00,  3.35it/s]#015Batches: 100%|██████████| 1/1 [00:00<00:00,  3.35it/s][0m
    [34mRunning retrieval + generation; writing results to /opt/ml/processing/output/rag_results.jsonl[0m
    [34mDone. Saved batch RAG results to: /opt/ml/processing/output/rag_results.jsonl[0m
    



```python
# List and print all objects in the bucket
response = s3_client.list_objects_v2(Bucket=bucket_name)

# Check if there are objects in the bucket
if 'Contents' in response:
    for obj in response['Contents']:
        print(obj['Key'])  # Print the object's key (its path in the bucket)
else:
    print("The bucket is empty or does not exist.")
```

    chunks/answers/rag_results.jsonl
    chunks/chunks.jsonl
    chunks/queries/queries.jsonl
    embed/embeddings.npy
    embed/faiss.index
    embed/ids.jsonl



```python
import json
import boto3

# Extract bucket + prefix
assert batch_output_path.startswith("s3://")
_, _, bkt_and_key = batch_output_path.partition("s3://")
bucket_name_results, _, prefix_results = bkt_and_key.partition("/")

# List all objects in the batch results directory
resp = s3.list_objects_v2(
    Bucket=bucket_name_results,
    Prefix=prefix_results
)

if "Contents" not in resp:
    raise RuntimeError("No results found in RAG batch output S3 folder.")

result_files = [
    obj["Key"] for obj in resp["Contents"]
    if obj["Key"].endswith(".jsonl")
]

print("Found result files:")
for rf in result_files:
    print("  -", rf)

# Load all results into a Python list
rag_results = []
for key in result_files:
    print(f"\nLoading {key} ...")
    obj = s3_client.get_object(Bucket=bucket_name_results, Key=key)
    for line in obj["Body"].read().decode("utf-8").splitlines():
        rag_results.append(json.loads(line))

print(f"\nLoaded {len(rag_results)} generated answers.")


```

    Found result files:
      - chunks/answers/rag_results.jsonl
    
    Loading chunks/answers/rag_results.jsonl ...
    
    Loaded 6 generated answers.



```python
rag_results
```




    [{'query': 'Summarize the main conflict in Romeo and Juliet.',
      'query_meta': {'id': 1,
       'query': 'Summarize the main conflict in Romeo and Juliet.'},
      'neighbor_ids': [18, 152, 75, 282, 107],
      'neighbor_scores': [0.7125013470649719,
       0.661830484867096,
       0.634709358215332,
       0.6318341493606567,
       0.6309801340103149],
      'retrieved_texts': [",\nHe swong about his head, and cut the windes,\nWho nothing hurt withall, hist him in scorne.\nWhile we were enterchanging thrusts and blowes,\nCame more and more, and fought on part and part,\nTill the Prince came, who parted either part\n\n   Wife. O where is Romeo, saw you him to day?\nRight glad am I, he was not at this fray\n\n   Ben. Madam, an houre before the worshipt Sun\nPeer'd forth the golden window of the East,\nA troubled mind draue me to walke abroad,\nWhere vnderneath the groue ",
       " vrg'd withall\nYour high displeasure: all this vttered,\nWith gentle breath, calme looke, knees humbly bow'd\nCould not take truce with the vnruly spleene\nOf Tybalts deafe to peace, but that he Tilts\nWith Peircing steele at bold Mercutio's breast,\nWho all as hot, turnes deadly point to point,\nAnd with a Martiall scorne, with one hand beates\nCold death aside, and with the other sends\nIt back to Tybalt, whose dexterity\nRetorts it: Romeo he cries aloud,\nHold Friends, Friends part, and swifter then his tongue,\nHis aged",
       "must complaine,\nAnd she steale Loues sweet bait from fearefull hookes:\nBeing held a foe, he may not haue accesse\nTo breath such vowes as Louers vse to sweare,\nAnd she as much in Loue, her meanes much lesse,\nTo meete her new Beloued any where:\nBut passion lends them Power, time, meanes to meete,\nTemp'ring extremities with extreame sweete.\nEnter Romeo alone.\n\n  Rom. Can I goe forward when my heart is here?\nTurne backe dull earth, and find thy Center out.\nEnter Benuolio, with Mercutio.\n\n  Ben. Romeo, my ",
       "place\nDoth make against me of this direfull murther:\nAnd heere I stand both to impeach and purge\nMy selfe condemned, and my selfe excus'd\n\n   Prin. Then say at once, what thou dost know in this?\n  Fri. I will be briefe, for my short date of breath\nIs not so long as is a tedious tale.\nRomeo there dead, was husband to that Iuliet,\nAnd she there dead, that's Romeos faithfull wife:\nI married them; and their stolne marriage day\nWas Tybalts Doomesday: whose vntimely death\nBanish'd the new-made ",
       " to his Fathers, I spoke with his man\n\n   Mer. Why that same pale hard-harted wench, that Rosaline\ntorments him so, that he will sure run mad\n\n   Ben. Tibalt, the kinsman to old Capulet, hath sent a Letter\nto his Fathers house\n\n   Mer. A challenge on my life\n\n   Ben. Romeo will answere it\n\n   Mer. Any man that can write, may answere a Letter\n\n   Ben. Nay, he will answere the Letters Maister how he\ndares, being dared\n\n   Mer. Alas poore Romeo, he is already dead stab'd with\na white wenches blacke eye"],
      'answer': 'The main conflict in Romeo and Juliet is the longstanding feud between the Montagues and Capulets, which leads to the tragic deaths of Romeo and Juliet. Despite their love for each other, they are unable to be together due to the hatred between their families. This results in a series of unfortunate events, including brawls, banishments, and ultimately, their own deaths.'},
     {'query': 'Who are the key characters in Romeo and Juliet, and how are they related?',
      'query_meta': {'id': 2,
       'query': 'Who are the key characters in Romeo and Juliet, and how are they related?'},
      'neighbor_ids': [18, 277, 76, 274, 153],
      'neighbor_scores': [0.5738745927810669,
       0.5589990615844727,
       0.5534552931785583,
       0.5304517149925232,
       0.5298224687576294],
      'retrieved_texts': [",\nHe swong about his head, and cut the windes,\nWho nothing hurt withall, hist him in scorne.\nWhile we were enterchanging thrusts and blowes,\nCame more and more, and fought on part and part,\nTill the Prince came, who parted either part\n\n   Wife. O where is Romeo, saw you him to day?\nRight glad am I, he was not at this fray\n\n   Ben. Madam, an houre before the worshipt Sun\nPeer'd forth the golden window of the East,\nA troubled mind draue me to walke abroad,\nWhere vnderneath the groue ",
       ". The ground is bloody,\nSearch about the Churchyard.\nGo some of you, who ere you find attach.\nPittifull sight, here lies the Countie slaine,\nAnd Iuliet bleeding, warme and newly dead\nWho here hath laine these two dayes buried.\nGo tell the Prince, runne to the Capulets,\nRaise vp the Mountagues, some others search,\nWe see the ground whereon these woes do lye,\nBut the true ground of all these piteous woes,\nWe cannot without circumstance descry.\nEnter Romeo's man.\n\n  Watch. Here's Romeo's man,\nWe found him ",
       'Cozen Romeo, Romeo\n\n   Merc. He is wise,\nAnd on my life hath stolne him home to bed\n\n   Ben. He ran this way and leapt this Orchard wall.\nCall good Mercutio:\nNay, Ile coniure too\n\n   Mer. Romeo, Humours, Madman, Passion, Louer,\nAppeare thou in the likenesse of a sigh,\nSpeake but one time, and I am satisfied:\nCry me but ay me, Prouant, but Loue and day,\nSpeake to my goship Venus one faire word,\nOne Nickname for her purblind Sonne and her,\nYoung Abraham Cupid he that shot so true',
       " that my Maister slew him\n\n   Fri. Romeo.\nAlacke, alacke, what blood is this which staines\nThe stony entrance of this Sepulcher?\nWhat meane these Masterlesse, and goarie Swords\nTo lie discolour'd by this place of peace?\nRomeo, oh pale: who else? what Paris too?\nAnd steept in blood? Ah what an vnkind houre\nIs guiltie of this lamentable chance?\nThe Lady stirs\n\n   Iul. O comfortable Frier, where's my Lord?\nI do remember well where I should be:\nAnd there I am, where is my Romeo?\n  Fri. I heare",
       " arme, beats downe their fatall points,\nAnd twixt them rushes, vnderneath whose arme,\nAn enuious thrust from Tybalt, hit the life\nOf stout Mercutio, and then Tybalt fled.\nBut by and by comes backe to Romeo,\nWho had but newly entertained Reuenge,\nAnd too't they goe like lightning, for ere I\nCould draw to part them, was stout Tybalt slaine:\nAnd as he fell, did Romeo turne and flie:\nThis is the truth, or let Benuolio die\n\n   Cap. Wi. He is a kinsman to the Mountague,\nAffection makes him false"],
      'answer': "The key characters in Romeo and Juliet include Romeo, Mercutio, Tybalt, the Prince, Benvolio, the Capulets (implied but not named), and the Montagues (implied but not named). Romeo and Juliet are the titular characters, who are from feuding families, the Capulets and Montagues. Mercutio is Romeo's friend, Tybalt is Juliet's cousin and a Capulet, and the Prince is the ruler of Verona trying to keep peace between the families. Benvolio is another friend of Rome"},
     {'query': 'What events lead to the deaths of Romeo and Juliet?',
      'query_meta': {'id': 3,
       'query': 'What events lead to the deaths of Romeo and Juliet?'},
      'neighbor_ids': [18, 282, 277, 152, 274],
      'neighbor_scores': [0.6565954685211182,
       0.6423623561859131,
       0.6164535284042358,
       0.6138817071914673,
       0.6039419174194336],
      'retrieved_texts': [",\nHe swong about his head, and cut the windes,\nWho nothing hurt withall, hist him in scorne.\nWhile we were enterchanging thrusts and blowes,\nCame more and more, and fought on part and part,\nTill the Prince came, who parted either part\n\n   Wife. O where is Romeo, saw you him to day?\nRight glad am I, he was not at this fray\n\n   Ben. Madam, an houre before the worshipt Sun\nPeer'd forth the golden window of the East,\nA troubled mind draue me to walke abroad,\nWhere vnderneath the groue ",
       "place\nDoth make against me of this direfull murther:\nAnd heere I stand both to impeach and purge\nMy selfe condemned, and my selfe excus'd\n\n   Prin. Then say at once, what thou dost know in this?\n  Fri. I will be briefe, for my short date of breath\nIs not so long as is a tedious tale.\nRomeo there dead, was husband to that Iuliet,\nAnd she there dead, that's Romeos faithfull wife:\nI married them; and their stolne marriage day\nWas Tybalts Doomesday: whose vntimely death\nBanish'd the new-made ",
       ". The ground is bloody,\nSearch about the Churchyard.\nGo some of you, who ere you find attach.\nPittifull sight, here lies the Countie slaine,\nAnd Iuliet bleeding, warme and newly dead\nWho here hath laine these two dayes buried.\nGo tell the Prince, runne to the Capulets,\nRaise vp the Mountagues, some others search,\nWe see the ground whereon these woes do lye,\nBut the true ground of all these piteous woes,\nWe cannot without circumstance descry.\nEnter Romeo's man.\n\n  Watch. Here's Romeo's man,\nWe found him ",
       " vrg'd withall\nYour high displeasure: all this vttered,\nWith gentle breath, calme looke, knees humbly bow'd\nCould not take truce with the vnruly spleene\nOf Tybalts deafe to peace, but that he Tilts\nWith Peircing steele at bold Mercutio's breast,\nWho all as hot, turnes deadly point to point,\nAnd with a Martiall scorne, with one hand beates\nCold death aside, and with the other sends\nIt back to Tybalt, whose dexterity\nRetorts it: Romeo he cries aloud,\nHold Friends, Friends part, and swifter then his tongue,\nHis aged",
       " that my Maister slew him\n\n   Fri. Romeo.\nAlacke, alacke, what blood is this which staines\nThe stony entrance of this Sepulcher?\nWhat meane these Masterlesse, and goarie Swords\nTo lie discolour'd by this place of peace?\nRomeo, oh pale: who else? what Paris too?\nAnd steept in blood? Ah what an vnkind houre\nIs guiltie of this lamentable chance?\nThe Lady stirs\n\n   Iul. O comfortable Frier, where's my Lord?\nI do remember well where I should be:\nAnd there I am, where is my Romeo?\n  Fri. I heare"],
      'answer': "The deaths of Romeo and Juliet result from a series of unfortunate events. Romeo and Tybalt engage in a fight, leading to Tybalt's death at Romeo's hands. Later, Romeo and Juliet secretly marry, but their union is discovered, and a brawl ensues. In the chaos, Tybalt's cousin, Mercutio, is killed by Tybalt, and Romeo, in turn, kills Tybalt in self-defense. Romeo is banished from Verona, and in despair, he and Juliet take their own lives"},
     {'query': 'How does the feud between the Montagues and Capulets drive the plot?',
      'query_meta': {'id': 4,
       'query': 'How does the feud between the Montagues and Capulets drive the plot?'},
      'neighbor_ids': [15, 200, 278, 144, 234],
      'neighbor_scores': [0.6035712361335754,
       0.5812185406684875,
       0.5801926851272583,
       0.5796860456466675,
       0.5786124467849731],
      'retrieved_texts': [" his Blade in spight of me.\nEnter old Mountague, & his wife.\n\n  Moun. Thou villaine Capulet. Hold me not, let me go\n  2.Wife. Thou shalt not stir a foote to seeke a Foe.\nEnter Prince Eskales, with his Traine.\n\n  Prince. Rebellious Subiects, Enemies to peace,\nProphaners of this Neighbor-stained Steele,\nWill they not heare? What hoe, you Men, you Beasts,\nThat quench the fire of your pernitious Rage,\nWith purple Fountaines issuing from your Veines:\nOn paine of Torture, from those bloody hands\nThrow your mistemper'd ",
       '\nEre he that should be Husband comes to woe:\nI pray you tell my Lord and Father Madam,\nI will not marrie yet, and when I doe, I sweare\nIt shall be Romeo, whom you know I hate\nRather then Paris. These are newes indeed\n\n   Mo. Here comes your Father, tell him so your selfe,\nAnd see how he will take it at your hands.\nEnter Capulet and Nurse.\n\n  Cap. When the Sun sets, the earth doth drizzle deaw\nBut for the Sunset of my Brothers Sonne,\nIt raines downright.\nHow now?',
       'in the Churchyard\n\n   Con. Hold him in safety, till the Prince come hither.\nEnter Frier, and another Watchman.\n\n  3.Wat. Here is a Frier that trembles, sighes, and weepes\nWe tooke this Mattocke and this Spade from him,\nAs he was comming from this Church-yard side\n\n   Con. A great suspition, stay the Frier too.\nEnter the Prince.\n\n  Prin. What misaduenture is so earely vp,\nThat calls our person from our mornings rest?\nEnter Capulet and his Wife.\n\n  Cap. What should it be that they so shrike abroad?\n  Wife. O ',
       ' reason of my loue,\nAnd so good Capulet, which name I tender\nAs dearely as my owne, be satisfied\n\n   Mer. O calme, dishonourable, vile submission:\nAlla stucatho carries it away.\nTybalt, you Rat-catcher, will you walke?\n  Tib. What wouldst thou haue with me?\n  Mer. Good King of Cats, nothing but one of your nine\nliues, that I meane to make bold withall, and as you shall\nvse me hereafter dry beate the rest of the eight. Will you\npluck your Sword out of his Pilcher by the eares? Make\n',
       "Enter old Capulet.\n\n  Cap. Come, stir, stir, stir,\nThe second Cocke hath Crow'd,\nThe Curphew Bell hath rung, 'tis three a clocke:\nLooke to the bakte meates, good Angelica,\nSpare not for cost\n\n   Nur. Go you Cot-queane, go,\nGet you to bed, faith youle be sicke to morrow\nFor this nights watching\n\n   Cap. No not a whit: what? I haue watcht ere now\nAll night for lesse cause, and nere beene sicke\n\n   La. I you haue bin a Mouse-hunt in your time,\nBut I will watch "],
      'answer': 'The feud between the Montagues and Capulets is the central conflict of the play. It drives the plot as the two families are sworn enemies, leading to various acts of violence and hostility towards each other. In the context provided, we see Capulet and Montague encountering each other unexpectedly, with both trying to prevent their children from marrying each other. This desire to maintain family honor and hatred towards each other ultimately leads to tragic consequences.'},
     {'query': 'What themes about love and fate appear in Romeo and Juliet?',
      'query_meta': {'id': 5,
       'query': 'What themes about love and fate appear in Romeo and Juliet?'},
      'neighbor_ids': [250, 75, 18, 168, 76],
      'neighbor_scores': [0.636717677116394,
       0.6359308958053589,
       0.6271967887878418,
       0.6071748733520508,
       0.6011824607849121],
      'retrieved_texts': [" ioyfull newes at hand:\nMy bosomes L[ord]. sits lightly in his throne:\nAnd all this day an vnaccustom'd spirit,\nLifts me aboue the ground with cheerefull thoughts.\nI dreamt my Lady came and found me dead,\n(Strange dreame that giues a dead man leaue to thinke,)\nAnd breath'd such life with kisses in my lips,\nThat I reuiu'd and was an Emperour.\nAh me, how sweet is loue it selfe possest,\nWhen but loues shadowes are so rich in ioy.\nEnter Romeo's man.\n\nNewes from Verona, ",
       "must complaine,\nAnd she steale Loues sweet bait from fearefull hookes:\nBeing held a foe, he may not haue accesse\nTo breath such vowes as Louers vse to sweare,\nAnd she as much in Loue, her meanes much lesse,\nTo meete her new Beloued any where:\nBut passion lends them Power, time, meanes to meete,\nTemp'ring extremities with extreame sweete.\nEnter Romeo alone.\n\n  Rom. Can I goe forward when my heart is here?\nTurne backe dull earth, and find thy Center out.\nEnter Benuolio, with Mercutio.\n\n  Ben. Romeo, my ",
       ",\nHe swong about his head, and cut the windes,\nWho nothing hurt withall, hist him in scorne.\nWhile we were enterchanging thrusts and blowes,\nCame more and more, and fought on part and part,\nTill the Prince came, who parted either part\n\n   Wife. O where is Romeo, saw you him to day?\nRight glad am I, he was not at this fray\n\n   Ben. Madam, an houre before the worshipt Sun\nPeer'd forth the golden window of the East,\nA troubled mind draue me to walke abroad,\nWhere vnderneath the groue ",
       " beguil'd,\nBoth you and I for Romeo is exild:\nHe made you for a high-way to my bed,\nBut I a Maid, die Maiden widowed.\nCome Cord, come Nurse, Ile to my wedding bed,\nAnd death not Romeo, take my Maiden head\n\n   Nur. Hie to your Chamber, Ile find Romeo\nTo comfort you, I wot well where he is:\nHarke ye your Romeo will be heere at night,\nIle to him, he is hid at Lawrence Cell\n\n   Iul. O find him, giue this Ring to my true Knight,\nAnd",
       'Cozen Romeo, Romeo\n\n   Merc. He is wise,\nAnd on my life hath stolne him home to bed\n\n   Ben. He ran this way and leapt this Orchard wall.\nCall good Mercutio:\nNay, Ile coniure too\n\n   Mer. Romeo, Humours, Madman, Passion, Louer,\nAppeare thou in the likenesse of a sigh,\nSpeake but one time, and I am satisfied:\nCry me but ay me, Prouant, but Loue and day,\nSpeake to my goship Venus one faire word,\nOne Nickname for her purblind Sonne and her,\nYoung Abraham Cupid he that shot so true'],
      'answer': "The context above illustrates the intense and consuming nature of love, as well as the role of fate in bringing Romeo and Juliet together despite their feuding families. Romeo experiences a deep joy and ecstasy when he is in love, even in the face of adversity. Juliet's dream of being reunited with Romeo after death also highlights the power of love to transcend the physical world. Fate plays a significant role in their relationship, as they are unable to be together openly due to their families' feud, yet they find ways to meet and express their love. Ultimately, their"},
     {'query': 'Who kills Mercutio?',
      'query_meta': {'id': 6, 'query': 'Who kills Mercutio?'},
      'neighbor_ids': [145, 148, 137, 154, 149],
      'neighbor_scores': [0.49320435523986816,
       0.45003509521484375,
       0.44290006160736084,
       0.4314851760864258,
       0.4057557284832001],
      'retrieved_texts': ['hast, least mine be about your eares ere it be out\n\n   Tib. I am for you\n\n   Rom. Gentle Mercutio, put thy Rapier vp\n\n   Mer. Come sir, your Passado\n\n   Rom. Draw Benuolio, beat downe their weapons:\nGentlemen, for shame forbeare this outrage,\nTibalt, Mercutio, the Prince expresly hath\nForbidden bandying in Verona streetes.\nHold Tybalt, good Mercutio.\n\nExit Tybalt.\n\n  Mer. I am hurt.\nA plague a both the Houses, I am sped:\nIs he gone and hath nothing?\n  Ben. What art thou hurt?\n  Mer. I, I, a scratch',
       " that an houre\nHath beene my Cozin: O Sweet Iuliet,\nThy Beauty hath made me Effeminate,\nAnd in my temper softned Valours steele.\nEnter Benuolio.\n\n\n  Ben. O Romeo, Romeo, braue Mercutio's is dead,\nThat Gallant spirit hath aspir'd the Cloudes,\nWhich too vntimely here did scorne the earth\n\n   Rom. This daies blacke Fate, on mo daies depend,\nThis but begins, the wo others must end.\nEnter Tybalt.\n\n  Ben. Here comes the Furious Tybalt backe againe\n\n   Rom. He gon in triumph, and Mercutio slaine?\nAway to heauen respectiue Lenitie",
       ' alone,\nTill holy Church incorporate two in one.\nEnter Mercutio, Benuolio, and men.\n\n  Ben. I pray thee good Mercutio lets retire,\nThe day is hot, the Capulets abroad:\nAnd if we meet, we shal not scape a brawle, for now these\nhot dayes, is the mad blood stirring\n\n   Mer. Thou art like one of these fellowes, that when he\nenters the confines of a Tauerne, claps me his Sword vpon\nthe Table, and sayes, God send me no need of thee: and by\nthe operation of the second cup, drawes ',
       ', he speakes not true:\nSome twenty of them fought in this blacke strife,\nAnd all those twenty could but kill one life.\nI beg for Iustice, which thou Prince must giue:\nRomeo slew Tybalt, Romeo must not liue\n\n   Prin. Romeo slew him, he slew Mercutio,\nWho now the price of his deare blood doth owe\n\n   Cap. Not Romeo Prince, he was Mercutios Friend,\nHis fault concludes, but what the law should end,\nThe life of Tybalt\n\n   Prin. And for that offence,\nImmediately we doe exile him hence:\nI haue an interest',
       ",\nAnd fire and Fury, be my conduct now.\nNow Tybalt take the Villaine backe againe\nThat late thou gau'st me, for Mercutios soule\nIs but a little way aboue our heads,\nStaying for thine to keepe him companie:\nEither thou or I, or both, must goe with him\n\n   Tib. Thou wretched Boy that didst consort him here,\nShalt with him hence\n\n   Rom. This shall determine that.\n\nThey fight. Tybalt falles.\n\n  Ben. Romeo, away be gone:\nThe Citizens are vp, and Tybalt slaine,\nStand not amaz'd, the Prince will"],
      'answer': 'Romeo kills Mercutio.'}]




```python
for i, r in enumerate(rag_results):
    print(f"\n=== RESULT {i} ===")
    print("Query:", r["query"])
    print("Answer:", r["answer"])
    print("Retrieved text sample:", r["retrieved_texts"][0][:300], "...")

```

    
    === RESULT 0 ===
    Query: Summarize the main conflict in Romeo and Juliet.
    Answer: The main conflict in Romeo and Juliet is the longstanding feud between the Montagues and Capulets, which leads to the tragic deaths of Romeo and Juliet. Despite their love for each other, they are unable to be together due to the hatred between their families. This results in a series of unfortunate events, including brawls, banishments, and ultimately, their own deaths.
    Retrieved text sample: ,
    He swong about his head, and cut the windes,
    Who nothing hurt withall, hist him in scorne.
    While we were enterchanging thrusts and blowes,
    Came more and more, and fought on part and part,
    Till the Prince came, who parted either part
    
       Wife. O where is Romeo, saw you him to day?
    Right glad am I,  ...
    
    === RESULT 1 ===
    Query: Who are the key characters in Romeo and Juliet, and how are they related?
    Answer: The key characters in Romeo and Juliet include Romeo, Mercutio, Tybalt, the Prince, Benvolio, the Capulets (implied but not named), and the Montagues (implied but not named). Romeo and Juliet are the titular characters, who are from feuding families, the Capulets and Montagues. Mercutio is Romeo's friend, Tybalt is Juliet's cousin and a Capulet, and the Prince is the ruler of Verona trying to keep peace between the families. Benvolio is another friend of Rome
    Retrieved text sample: ,
    He swong about his head, and cut the windes,
    Who nothing hurt withall, hist him in scorne.
    While we were enterchanging thrusts and blowes,
    Came more and more, and fought on part and part,
    Till the Prince came, who parted either part
    
       Wife. O where is Romeo, saw you him to day?
    Right glad am I,  ...
    
    === RESULT 2 ===
    Query: What events lead to the deaths of Romeo and Juliet?
    Answer: The deaths of Romeo and Juliet result from a series of unfortunate events. Romeo and Tybalt engage in a fight, leading to Tybalt's death at Romeo's hands. Later, Romeo and Juliet secretly marry, but their union is discovered, and a brawl ensues. In the chaos, Tybalt's cousin, Mercutio, is killed by Tybalt, and Romeo, in turn, kills Tybalt in self-defense. Romeo is banished from Verona, and in despair, he and Juliet take their own lives
    Retrieved text sample: ,
    He swong about his head, and cut the windes,
    Who nothing hurt withall, hist him in scorne.
    While we were enterchanging thrusts and blowes,
    Came more and more, and fought on part and part,
    Till the Prince came, who parted either part
    
       Wife. O where is Romeo, saw you him to day?
    Right glad am I,  ...
    
    === RESULT 3 ===
    Query: How does the feud between the Montagues and Capulets drive the plot?
    Answer: The feud between the Montagues and Capulets is the central conflict of the play. It drives the plot as the two families are sworn enemies, leading to various acts of violence and hostility towards each other. In the context provided, we see Capulet and Montague encountering each other unexpectedly, with both trying to prevent their children from marrying each other. This desire to maintain family honor and hatred towards each other ultimately leads to tragic consequences.
    Retrieved text sample:  his Blade in spight of me.
    Enter old Mountague, & his wife.
    
      Moun. Thou villaine Capulet. Hold me not, let me go
      2.Wife. Thou shalt not stir a foote to seeke a Foe.
    Enter Prince Eskales, with his Traine.
    
      Prince. Rebellious Subiects, Enemies to peace,
    Prophaners of this Neighbor-stained Steel ...
    
    === RESULT 4 ===
    Query: What themes about love and fate appear in Romeo and Juliet?
    Answer: The context above illustrates the intense and consuming nature of love, as well as the role of fate in bringing Romeo and Juliet together despite their feuding families. Romeo experiences a deep joy and ecstasy when he is in love, even in the face of adversity. Juliet's dream of being reunited with Romeo after death also highlights the power of love to transcend the physical world. Fate plays a significant role in their relationship, as they are unable to be together openly due to their families' feud, yet they find ways to meet and express their love. Ultimately, their
    Retrieved text sample:  ioyfull newes at hand:
    My bosomes L[ord]. sits lightly in his throne:
    And all this day an vnaccustom'd spirit,
    Lifts me aboue the ground with cheerefull thoughts.
    I dreamt my Lady came and found me dead,
    (Strange dreame that giues a dead man leaue to thinke,)
    And breath'd such life with kisses in m ...
    
    === RESULT 5 ===
    Query: Who kills Mercutio?
    Answer: Romeo kills Mercutio.
    Retrieved text sample: hast, least mine be about your eares ere it be out
    
       Tib. I am for you
    
       Rom. Gentle Mercutio, put thy Rapier vp
    
       Mer. Come sir, your Passado
    
       Rom. Draw Benuolio, beat downe their weapons:
    Gentlemen, for shame forbeare this outrage,
    Tibalt, Mercutio, the Prince expresly hath
    Forbidden band ...


**Note**: The shape of our embedding matrix is (283, 384) — representing the 283 chunks we prepared, and the 384 features describing each chunk. These are neural network derived features, lacking direct interpretability.

## Step 4: Retrieve Relevant Chunks

In this step, we demonstrate a core component of a RAG (Retrieval-Augmented Generation) pipeline — finding the most relevant pieces of text to answer a user's question. Here's how it works:

- We take the user's question and convert it into a vector embedding using the *same model* we used to embed the original text chunks.
- Then we use cosine similarity to compare the question's embedding to all text chunk embeddings.
- We select the top *N* most similar chunks to use as context for the language model.

### Are question embeddings and chunk embeddings really comparable?

We're assuming that the embedding model (e.g., `all-MiniLM-L6-v2`) was trained in such a way that *questions and answers occupy the same semantic space*. That is, if a question and a passage are semantically aligned (e.g., about the same topic or fact), their embeddings should be close. This assumption holds reasonably well for general-purpose models trained on sentence pairs, but it's not perfect — especially for very abstract or indirect questions. If a model was only trained to embed statements, it may not align questions correctly. You might retrieve chunks that are **related but not directly useful** for answering the question.


```python
from sklearn.metrics.pairwise import cosine_similarity

def retrieve_relevant_chunks(query, chunks, embeddings, top_n=3):
    """Embed the query with the SageMaker endpoint and return top-N matching chunks."""
    query_embedding = embed_texts_sagemaker([query])  # shape (1, d)
    scores = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = scores.argsort()[-top_n:][::-1]
    results = [(chunks[i], float(scores[i])) for i in top_indices]
    return results

```


```python
question = "Who kills Mercutio?" # Answer: Tybalt, Juliet's cousin
top_chunks = retrieve_relevant_chunks(question, chunks, embeddings)

for i, (chunk, score) in enumerate(top_chunks, 1):
    print(f"\n\n############ CHUNK {i} ############")
    print(f"Score: {score:.4f}")
    print(chunk)
```

### Summary: Retrieval results for factual query

The following output shows how a RAG system handles the factual question "Who kills Mercutio?" using a chunked version of *Romeo and Juliet*. While no chunk explicitly states "Tybalt kills Mercutio" in modern phrasing, the system successfully retrieves highly relevant context. The Project Gutenberg edition uses the older spelling "Tibalt", which the retriever still resolves semantically.

* **Chunk 1** is the most direct and useful. It captures the aftermath of the duel, with citizens exclaiming:
  * "*Which way ran he that kild Mercutio? Tibalt that Murtherer, which way ran he?*". Despite the archaic spelling and phrasing, this chunk effectively provides the answer when interpreted in context.

* **Chunk 2** sets up the conflict. It includes Mercutio and Benvolio discussing that:
  * "*Tibalt, the kinsman to old Capulet, hath sent a Letter*" ... "*A challenge on my life*". While it doesn't answer the question directly, it reinforces that Tibalt is the antagonist and establishes his role in escalating the violence.

* **Chunk 3** presents the Prince's legal judgment:
  * "*Romeo, Prince, he was Mercutios Friend… The life of Tibalt*." The Prince confirms that Tybalt (Tibalt) has been killed in consequence of Mercutio's death. This chunk emphasizes closure rather than causality, but still supports the factual chain.

### Observations

- Early modern spelling (e.g., *Tibalt*) doesn't hinder embedding-based retrieval — a strength of semantic models.
- No chunk contains a complete "question + answer" sentence, but together they establish who killed whom, why, and what happened next.
- The system retrieves scenes with narrative and legal resolution, not just the killing itself.

This result demonstrates how chunk-level RAG with sentence-transformer embeddings can surface relevant evidence across spelling and stylistic variation, even when chunk boundaries split key action and dialogue.

### Run a few additional queries & report top-ranked chunk


```python
# Run a few factual queries and inspect the top-ranked chunks
factual_questions = [
    "Who kills Mercutio?", # Tybalt
    "Where does Romeo meet Juliet?", # Capulet's masquerade ball (party), which takes place at the Capulet family home in Verona
    "What punishment does the Prince give Romeo?" # exile / banishment
]

for q in factual_questions:
    print(f"\n=== Query: {q} ===")
    results = retrieve_relevant_chunks(q, chunks, embeddings, top_n=1)
    for i, (chunk, score) in enumerate(results, 1):
        print(f"\n--- CHUNK {i} (Score: {score:.4f}) ---")
        print(chunk[:800])  # print first ~800 chars for readability
```

### Improving retrieved chunks

Before we move on to having a language model generate answers, we need to take a closer look at the quality of the retrieved content.

As we just saw, our current retrieval method brings back passages that are topically related but often miss the actual moment where the answer appears. In some cases, the correct chunk is nearby but not retrieved. In others, key information may be split across multiple chunks or surrounded by distracting dialogue.

To address this, we'll focus on a key area of improvement: **refining the chunking strategy**.

#### Why chunking matters

The current approach uses a simple method such as splitting the text by a fixed word count. While this works for general purposes, it often cuts across meaningful dramatic units:

- A character's speech may be interrupted mid-line
- A fight scene may be split just before or after a critical action
- A conversation between characters may be split across chunks

This leads to less coherent retrieval and lowers the chance that a single chunk can fully answer the question.

Here are two practical adjustments we can use to improve the retrievals:

1. **Group complete speaker turns into chunks**: Instead of arbitrary lengths, we can group text based on who is speaking. This ensures each chunk preserves the flow and tone of the conversation.
2. **Use scene- or event-aware chunking**: By chunking based on scene boundaries or key events (e.g. "Romeo kills Tybalt"), we improve the chance that retrieved content captures complete dramatic moments, not just pieces of them.

These changes don't require a new model—they just help the existing model work with more meaningful input.

Next, we'll apply dialogue-aware chunking and rerun one of our earlier factual queries to see whether the results improve.

### Refining chunking strategy
Our current chunks are only based on word length. Instead, we can create chunks that are more tuned to the dataset and potential questions we might ask by defining a chunk as a "dialogue block", i.e.,  as a group of N full speaker turns (e.g., JULIET. + her lines, ROMEO. + his lines, etc.).

Let's give this a shot to see how it impacts retrieval.


```python
import re

def chunk_by_speaker_blocks(text, block_size=4):
    # This regex matches short speaker tags at the beginning of lines, e.g., "Ben." or "Rom."
    # Followed by speech text (either same line or indented on next)
    speaker_line_pattern = re.compile(r'^\s{0,3}([A-Z][a-z]+)\.\s+(.*)', re.MULTILINE)

    dialogue_blocks = []
    current_speaker = None
    current_lines = []

    for line in text.splitlines():
        match = speaker_line_pattern.match(line)
        if match:
            # Save previous speaker block if one was accumulating
            if current_speaker:
                dialogue_blocks.append(f"{current_speaker}.\n" + "\n".join(current_lines).strip())
            current_speaker = match.group(1)
            current_lines = [match.group(2)]
        elif current_speaker and line.strip():
            # Indented continuation of the same speaker
            current_lines.append(line)
        else:
            # Blank line or noise: treat as boundary
            if current_speaker and current_lines:
                dialogue_blocks.append(f"{current_speaker}.\n" + "\n".join(current_lines).strip())
                current_speaker = None
                current_lines = []

    # Add last block if exists
    if current_speaker and current_lines:
        dialogue_blocks.append(f"{current_speaker}.\n" + "\n".join(current_lines).strip())

    # Chunk into groups of speaker turns
    grouped_chunks = []
    for i in range(0, len(dialogue_blocks), block_size):
        chunk = "\n\n".join(dialogue_blocks[i:i + block_size])
        grouped_chunks.append(chunk.strip())

    return grouped_chunks
```


```python
speaker_chunks = chunk_by_speaker_blocks(file_contents, block_size=4)
print(f"Total speaker_chunks: {len(speaker_chunks)}")
print(f"Preview of first chunk:\n\n{speaker_chunks[0]}")
```

Our chunks have now been improved so that we aren't cutting off any diagloue mid-sentence, and each chunk contains a few turns between speakers -- allowing us to better capture the overall semantics of short passages from *Romeo and Juliet*.


```python
dialogue_embeddings = model.encode(speaker_chunks, device=device)

print(f"Shape of dialogue_embeddings matrix: {np.array(dialogue_embeddings).shape}")
```


```python
# Run a few factual queries and inspect the top-ranked chunks
factual_questions = [
    "Who kills Mercutio?", # Tybalt
    "Where does Romeo meet Juliet?", # Capulet's masquerade ball (party), which takes place at the Capulet family home in Verona
    "What punishment does the Prince give Romeo?" # exile / banishment
]

for q in factual_questions:
    print(f"\n=== Query: {q} ===")
    results = retrieve_relevant_chunks(q, speaker_chunks, dialogue_embeddings, top_n=1)
    for i, (chunk, score) in enumerate(results, 1):
        print(f"\n--- CHUNK {i} (Score: {score:.4f}) ---")
        print(chunk)  # print first ~800 chars for readability
```

### Takeaway

Refining our chunking strategy to preserve full speaker turns—and grouping several turns together—has already improved the relevance of the chunks retrieved. The content is more coherent, more complete, and better aligned with the structure of a play. This shows how much retrieval quality depends not just on the model, but on the way we prepare and represent the source material.

That said, even with better chunks, retrieval doesn't always land on the exact moment that answers the question. Sometimes it gets close but stops short; other times it picks up a scene with similar characters or themes, but not the one we need.

This points to a deeper challenge: *semantic similarity alone doesn't always capture answer relevance*. The chunk that's closest in meaning isn't always the one that answers the question. One way to address this is through a process called **reranking**.

### What is reranking?

Reranking means retrieving a small set of candidate chunks—say, the top 5—and then using an additional method to determine which of those is the best fit for the question.

That method could be:

- A custom scoring function (e.g., based on keyword overlap, speaker identity, or chunk metadata),
- Or—more powerfully—a *separate language model*.

This separate model can be small or large, depending on your resource availability:

- A smaller open-source model (like `mistral`, `falcon`, or `phi`) can often handle basic ranking tasks at low cost.
- A larger LLM (like GPT-3.5 or GPT-4) may be better at reasoning through subtleties and weighing relevance when answers are indirect or distributed across lines.

You might ask this model something like:

> Here are three passages. Which one best answers the question: "Who kills Mercutio?"

At first, it might feel strange to use one language model to support another—but this layered setup is common in production RAG pipelines. It separates concerns:

- The retriever quickly narrows down the universe of text,
- The reranker evaluates those chunks more deeply, focusing on which is most likely to be useful.

We won't implement this yet, but it's worth introducing now. As we start exploring more ambiguous or emotionally driven questions in later sections, reranking becomes one of the key techniques for bridging the gap between retrieval and meaningful response.

For now, we've established a strong foundation: well-structured chunks that carry clear speaker information and preserve narrative flow. That's a critical step toward building a RAG system that doesn't just respond, but interprets.

### Upgrading our retrieval model

The model we've used so far, [`multi-qa-MiniLM-L6-cos-v1`](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1), is a solid starting point for retrieval-augmented generation (RAG) pipelines, it is relatively lightweight (22M parameters, ~500–800 MB GPU memory), which makes it efficient but less expressive than larger models.

However, larger embedding models have more capacity to capture subtle semantic relationships, including indirect phrasing or domain-specific language. This can make a dramatic difference in tasks like matching Shakespearean dialogue to modern questions—something smaller models often struggle with.

Let's try a slightly larger model with 109 M parameters, [`all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)


```python
# Reuse the same SageMaker embedding endpoint for the speaker-based chunks
dialogue_embeddings = embed_texts_sagemaker(speaker_chunks)

```


```python
# Run a few factual queries and inspect the top-ranked chunks
factual_questions = [
    "Who kills Mercutio?", # Tybalt
    "Where does Romeo meet Juliet?", # Capulet's masquerade ball (party), which takes place at the Capulet family home in Verona
    "What punishment does the Prince give Romeo?" # exile / banishment
]

for q in factual_questions:
    print(f"\n=== Query: {q} ===")
    results = retrieve_relevant_chunks(q, speaker_chunks, dialogue_embeddings, top_n=1)
    for i, (chunk, score) in enumerate(results, 1):
        print(f"\n--- CHUNK {i} (Score: {score:.4f}) ---")
        print(chunk)  # print first ~800 chars for readability
```

If you're interested in exploring more powerful options for RAG pipelines, consider:

- [`intfloat/e5-large-v2`](https://huggingface.co/intfloat/e5-large-v2): A 24‑layer encoder (335M params) fine-tuned for dense retrieval with `query:` / `passage:` formatting.
- [`BAAI/bge-large-en-v1.5`](https://huggingface.co/BAAI/bge-large-en-v1.5): A high-performing English retriever (335M params) that tops MTEB benchmarks.
- [`deepseek-ai/DeepSeek-V2`](https://huggingface.co/deepseek-ai/DeepSeek-V2): A large-scale mixture-of-experts model (236 B params) pioneering efficient retrieval architectures, but note it's not a small encoder model—it's listed here to showcase advanced retrieval methods.

All of these are trained for dot-product similarity and work best with a high-performance index like `faiss.IndexFlatIP`.

**Note:** We didn't use FAISS in this notebook, since our dataset is small enough for brute-force similarity search. But once you move to larger models or bigger corpora, FAISS becomes essential for scalable and efficient retrieval.

## Step 5: Generate answer using retrieved context

### Putting it all together: Answering a question with a language model

Now that we've improved our chunking and retrieval process, we're ready to pass the retrieved content to *yet another* language model and generate an answer.

This step completes the typical RAG (Retrieval-Augmented Generation) workflow:

1. Retrieve the top-ranked passage(s) using a retrieval language model to embed the corpus into a Q&A semantic space
2. Concatenate retrieved results them into a structured prompt
3. Ask a (generative) language model to answer the user's question using only that retrieved context

This approach grounds the model's answer in specific evidence from the text, making it more trustworthy than asking the model to "hallucinate" an answer from general pretraining.


#### The prompt format

We use a basic prompt like this:

```
Use only the following passage to answer this question.
BEGIN_PASSAGE: [Top retrieved chunk(s) go here] END_PASSAGE 
QUESTION: [your question]
ANSWER:
```

By framing the input this way, we signal to the model that it should focus only on the retrieved content. We're not asking it to draw from general knowledge of the play—just from the selected passages. 

Let's begin assembling the full prompt:


```python
question = "Who killed Mercutio?" # Tybalt/Tibalt
```


```python
top_dialgoue_chunks = retrieve_relevant_chunks(question, speaker_chunks, dialogue_embeddings, top_n=3)

# Extract only the chunk text from (chunk, score) tuples
context = "\n".join(chunk for chunk, score in top_dialgoue_chunks)
print(context)
```


```python
prompt = f"Use the following passage to answer this question.\nBEGIN_PASSAGE:\n{context}\nEND_PASSAGE\nQUESTION: {question}\nANSWER:"
print(prompt)
```

### Language model for generation

For this section, we're using [`tiiuae/falcon-rw-1b`](https://huggingface.co/tiiuae/falcon-rw-1b), a small 1.3B parameter decoder-only model trained on the RefinedWeb dataset. It's designed for general-purpose text continuation, not for answering questions or following instructions.

This makes it a good baseline for testing how much a generative model can do with only retrieved context and minimal guidance. As we'll see, its output often reflects surface-level patterns or recent tokens, rather than accurate reasoning grounded in the text.


```python
import json

# Use a SageMaker text-generation endpoint instead of a local transformers pipeline
generation_endpoint_name = "rj-gen-falcon-rw-1b"  # TODO: replace with your actual endpoint name

def generate_with_sagemaker(
    prompt,
    max_new_tokens=128,
    temperature=0.1,
    do_sample=False,
    endpoint_name=generation_endpoint_name,
):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
            "return_full_text": True,
        },
    }
    response = sm_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    outputs = json.loads(response["Body"].read())

    # Most Hugging Face text-generation endpoints return a list of dicts with 'generated_text'
    if isinstance(outputs, list) and isinstance(outputs[0], dict) and "generated_text" in outputs[0]:
        return outputs[0]["generated_text"]

    # Fallback for slightly different schemas
    return str(outputs)

```

#### Model parameters and generation behavior

When we call the language model, we specify parameters like:

- `max_new_tokens`: Limits how much it can generate (e.g., 100 tokens)
- `do_sample=True`: Enables creative variation rather than deterministic output. For the purposes of getting a reproducible result, we'll set this to `False`

These parameters influence not just length, but also how literal or speculative the answer might be. Sampling increases variety but can also introduce tangents or continuation artifacts.


```python
result = generate_with_sagemaker(prompt, max_new_tokens=10, do_sample=False)

```


```python
print(result)
```

### Why the model output inludes the prompt

When using a decoder-only language model (like Falcon or GPT) with the Hugging Face `pipeline("text-generation")`, the output will include the entire input prompt followed by the model's generated continuation.

This happens because decoder-only models are trained to predict the *next token given all previous tokens*, not to separate a prompt from a response. So when you pass in a prompt, the model simply continues generating text — it doesn't know where "input" ends and "output" begins.

As a result, the `pipeline` will return a string that contains both:
```
[prompt] + [generated text]
```

If you're only interested in the generated part (e.g., the model's answer), you'll need to remove the prompt manually after generation.

We can strip off the final answer / generated result with the next code cell.


```python
generated_answer = result[len(prompt):].strip()
print(generated_answer)
```

#### Why the output might drift or repeat

Even though we ask just one question, you might see the model:

- Answer multiple questions in a row
- Invent follow-up questions and answers
- Continue in a Q&A or list format beyond what was asked

This usually happens when:

- The passage is long or covers multiple narrative beats
- The model detects a repeated pattern (e.g., “Question: … Answer: …”) and keeps going

For example, with a passage that includes both a fight and a romantic scene, the model might output:
```
Question: Who kills Mercutio?
Answer: Romeo.
Question: What does Juliet say about fate?
Answer: She curses fortune.
```
Even though we only asked the first question.

To limit this behavior, you can:

- Set a lower `max_new_tokens`
- Add a `stop` sequence after the first answer (if supported)
- Use a tighter or more explicit prompt style


```python
result = generate_with_sagemaker(prompt, max_new_tokens=1, do_sample=False)  # adjust to include max of 1 new tokens
generated_answer = result[len(prompt):].strip()
print(generated_answer)

```

### Note on model accuracy and hallucination

Smaller decoder-only models like `tiiuae/falcon-rw-1b` are fast and lightweight, but they can make factual errors, especially when summarizing events from structured texts like plays or historical records. For example, when asked "Who killed Mercutio?", the model incorrectly responded:
```
"Romeo"
```
This is not correct. Mercutio is killed by Tybalt during a street duel. Romeo kills Tybalt afterward in retaliation.

Interestingly, the correct information was present in the top retrieved chunk, but the phrasing may have confused the model:

> Mer.  
> I am hurt.  
> A plague a both the Houses, I am sped:  
> Is he gone and hath nothing?

> Ben.  
> What art thou hurt?

> Prin.  
> Romeo slew him, he slew Mercutio,  
> Who now the price of his deare blood doth owe

> Cap.  
> Not Romeo Prince, he was Mercutio’s Friend,  
> His fault concludes, but what the law should end,  
> The life of Tybalt

#### Instruction tuning improves perfomance
To improve factual accuracy in your RAG pipeline, it's helpful to use an **instruction-tuned** model rather than a base language model. You've been using `falcon-rw-1b` (where "rw" stands for “Refined Web”), which is trained only to continue text — not to follow specific question-and-answer instructions. That's why it often hallucinates factual events.

A lightweight upgrade is to instead use `tiiuae/Falcon3-1B-Instruct`, an instruction-tuned version of Falcon. It still runs on modest hardware but is trained to follow prompts and answer questions in a focused way.


```python
# If you've deployed a larger SageMaker endpoint (e.g., falcon3-1b-instruct), you can swap to it here.
generation_endpoint_name = "rj-gen-falcon3-1b-instruct"  # TODO: replace with your larger endpoint name

```


```python
# NOTE: We use max_new_tokens=3 here because words like "Tybalt" may be split into multiple tokens (e.g., "Ty", "b", "alt").
# It's often tricky to get exactly one word due to subword tokenization.
result = generate_with_sagemaker(prompt, max_new_tokens=3, do_sample=False)

# extract answer from full result, as before
generated_answer = result[len(prompt):].strip()
print(generated_answer)

```

If all else fails, we can start to try larger models to handle the answer generation step. Other models you could substitute here depending on your resources include:

- `mistralai/Mistral-7B-Instruct-v0.1` — for stronger instruction-following
- `meta-llama/Meta-Llama-3-8B-Instruct` — for more fluent answers
- `openai/gpt-3.5-turbo` — via API (not Hugging Face)

For most open-source models, using `transformers` + `pipeline()` allows easy swapping once your retrieval system is set up.

Keep in mind:

- Larger models require more memory (ideally a 12–16GB GPU)
- Instruction-tuned models typically follow prompts more reliably than base models
- You may still need to post-process outputs to extract just the answer

If you're working in Colab, consider using quantized models (e.g., via `bitsandbytes`) or calling the model via Hugging Face's hosted Inference API.

## Concluding remarks

This notebook introduced a basic Retrieval-Augmented Generation (RAG) pipeline for factual question answering using *Romeo and Juliet*. The goal was to build a simple but functioning system and surface practical lessons about how to improve performance.

**For retrieval**, we explored and discussed improvements such as:

- Using stronger embedding models (e.g., upgrading from `MiniLM` to `all-mpnet-base-v2`).
- Adopting a question-aligned chunking strategy, where chunks were grouped by speaker turns to better match the structure of expected queries.
- Implementing cosine similarity retrieval, which better handles variation in chunk lengths and embedding magnitudes.
- Briefly mentioning reranking as a next step, though not yet implemented.

**For generation**, we found that:

- Instruction-tuned language models yield more precise and context-sensitive answers.
- Prompt formatting significantly affects the clarity and relevance of the generated output.
- Post-processing may be necessary for trimming or cleaning model responses, especially in short-form QA tasks.

While larger models consistently improve both retrieval and generation, thoughtful design choices—such as aligning chunk structure to question types, using the right embedding normalization, and writing effective prompts—can yield substantial gains, even in smaller pipelines.

This notebook serves as a first step in a broader RAG workflow. Future notebooks will experiment with more flexible chunking, incorporate reranking, and test the system’s ability to handle interpretive or subjective questions.

## Questions?
If you any lingering questions about this resource, please feel free to post to the [Nexus Q&A](https://github.com/UW-Madison-DataScience/ML-X-Nexus/discussions/categories/q-a) on GitHub. We will improve materials on this website as additional questions come in.

## See also
- [**Kaggle RAG Challenge**: WattBot](https://www.kaggle.com/competitions/WattBot2025/overview): Put your RAG methods to the test in this ML+X hosted Kaggle challenge (featured in [MLM25](https://ml-marathon.wisc.edu/)).
- [**Workshop**:Intro to Natural Language Processing (NLP)](https://uw-madison-datascience.github.io/ML-X-Nexus/Learn/Workshops/Intro-Deeplearning_PyTorch.html): Brush up on NLP basics before diving head-first into RAG pipelines.

