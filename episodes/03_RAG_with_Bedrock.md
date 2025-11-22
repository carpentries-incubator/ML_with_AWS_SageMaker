---
title: "RAG with Amazon Bedrock"
teaching: 20
exercises: 10
---
:::::::::::::::::::::::::::::::::::::: questions
- What is the goal of this RAG workflow?
- How do we run this RAG variant?

::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::: objectives
- Understand this RAG approach.
- Run the workflow end-to-end.

::::::::::::::::::::::::::::::::::::::::::::::::
# RAG with Amazon Bedrock (Embeddings + Generation)
This notebook demonstrates a Bedrock-based RAG pipeline using Titan embeddings and Claude 3 Haiku.

```python
import os, json, boto3, numpy as np, requests
region="us-east-1"
bedrock=boto3.client("bedrock-runtime", region_name=region)
EMBED_MODEL_ID="amazon.titan-embed-text-v2:0"
GEN_MODEL_ID="anthropic.claude-3-haiku-20240307-v1:0"

```

```python
url="https://www.gutenberg.org/cache/epub/1112/pg1112.txt"
raw_text=requests.get(url).text

def simple_paragraph_chunks(text,max_chars=800):
    paras=[p.strip() for p in text.split("\n\n") if p.strip()]
    chunks=[]; buf=""
    for p in paras:
        if buf and len(buf)+len(p)+2>max_chars:
            chunks.append(buf.strip()); buf=p
        else:
            buf=(buf+"\n\n"+p).strip() if buf else p
    if buf: chunks.append(buf.strip())
    return chunks

chunks=simple_paragraph_chunks(raw_text,800)
len(chunks)
```

```python
def embed_texts_bedrock(texts):
    vecs=[]
    for t in texts:
        body=json.dumps({"inputText":t})
        resp=bedrock.invoke_model(modelId=EMBED_MODEL_ID, body=body)
        vecs.append(json.loads(resp["body"].read())["embedding"])
    return np.array(vecs,dtype="float32")

chunk_embeddings=embed_texts_bedrock(chunks)
chunk_embeddings.shape
```

```python
def cosine_sim(q,mat):
    q=q/(np.linalg.norm(q)+1e-9)
    m=mat/(np.linalg.norm(mat,axis=1,keepdims=True)+1e-9)
    return m@q

def retrieve_top_k(query,k=5):
    qv=embed_texts_bedrock([query])[0]
    sims=cosine_sim(qv,chunk_embeddings)
    idx=np.argsort(-sims)[:k]
    return [{"index":int(i),"similarity":float(sims[i]),"text":chunks[i]} for i in idx]

retrieve_top_k("Who kills Mercutio?",3)[0]
```

```python
def rag_answer_bedrock_claude(query,k=5,temperature=0.0):
    retrieved=retrieve_top_k(query,k)
    ctx="\n\n---\n\n".join([f"[Chunk {r['index']}]\n{r['text']}" for r in retrieved])
    system="You answer ONLY using the provided Shakespeare context."
    user=(f"Context:\n{ctx}\n\nQuestion: {query}")
    body={
        "modelId":GEN_MODEL_ID,
        "messages":[{"role":"user","content":[{"text":user}]}],
        "system":[{"text":system}],
        "inferenceConfig":{"maxTokens":256,"temperature":temperature}
    }
    resp=bedrock.converse(modelId=GEN_MODEL_ID,
                          messages=body["messages"],
                          system=body["system"],
                          inferenceConfig=body["inferenceConfig"])
    out=resp["output"]["message"]["content"]
    return "\n".join([c["text"] for c in out if "text" in c])

print(rag_answer_bedrock_claude("Who kills Mercutio?",5))
```

## Notes
This notebook compares Bedrock-based RAG vs SageMaker training workflows.

::::::::::::::::::::::::::::::::::::: keypoints
- This episode demonstrates a complete RAG workflow.
- Components include retrieval, embedding, and generation.
