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


```python
!pip install -q sentence-transformers accelerate

```


```python
from sentence_transformers import SentenceTransformer
import numpy as np
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu' # make sure you have GPU enabled in colab to speed things up!
print(f'device={device}')

model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1', device=device)

```

    /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
      import pynvml  # type: ignore[import]


    device=cuda



```python
embeddings = model.encode(chunks, device=device)

print(f"Shape of embedding matrix: {np.array(embeddings).shape}")
```

    Shape of embedding matrix: (291, 384)


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

def retrieve_relevant_chunks(model, query, chunks, embeddings, top_n=3):
    query_embedding = model.encode([query],device=device)
    scores = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = scores.argsort()[-top_n:][::-1]
    results = [(chunks[i], scores[i]) for i in top_indices]
    return results
```


```python
question = "Who kills Mercutio?" # Answer: Tybalt, Juliet's cousin
top_chunks = retrieve_relevant_chunks(model, question, chunks, embeddings)

for i, (chunk, score) in enumerate(top_chunks, 1):
    print(f"\n\n############ CHUNK {i} ############")
    print(f"Score: {score:.4f}")
    print(chunk)
```

    
    
    ############ CHUNK 1 ############
    Score: 0.4397
    , he speakes not true:
    Some twenty of them fought in this blacke strife,
    And all those twenty could but kill one life.
    I beg for Iustice, which thou Prince must giue:
    Romeo slew Tybalt, Romeo must not liue
    
       Prin. Romeo slew him, he slew Mercutio,
    Who now the price of his deare blood doth owe
    
       Cap. Not Romeo Prince, he was Mercutios Friend,
    His fault concludes, but what the law should end,
    The life of Tybalt
    
       Prin. And for that offence,
    Immediately we doe exile him hence:
    I haue an interest
    
    
    ############ CHUNK 2 ############
    Score: 0.4059
    vnluckie Mannage of this fatall brall:
    There lies the man slaine by young Romeo,
    That slew thy kinsman braue Mercutio
    
       Cap. Wi. Tybalt, my Cozin? O my Brothers Child,
    O Prince, O Cozin, Husband, O the blood is spild
    Of my deare kinsman. Prince as thou art true,
    For bloud of ours, shed bloud of Mountague.
    O Cozin, Cozin
    
       Prin. Benuolio, who began this Fray?
      Ben. Tybalt here slaine, whom Romeo's hand did slay,
    Romeo that spoke him faire, bid him bethinke
    How nice the Quarrell was, and
    
    
    ############ CHUNK 3 ############
    Score: 0.4028
    hast, least mine be about your eares ere it be out
    
       Tib. I am for you
    
       Rom. Gentle Mercutio, put thy Rapier vp
    
       Mer. Come sir, your Passado
    
       Rom. Draw Benuolio, beat downe their weapons:
    Gentlemen, for shame forbeare this outrage,
    Tibalt, Mercutio, the Prince expresly hath
    Forbidden bandying in Verona streetes.
    Hold Tybalt, good Mercutio.
    
    Exit Tybalt.
    
      Mer. I am hurt.
    A plague a both the Houses, I am sped:
    Is he gone and hath nothing?
      Ben. What art thou hurt?
      Mer. I, I, a scratch


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
    results = retrieve_relevant_chunks(model, q, chunks, embeddings, top_n=1)
    for i, (chunk, score) in enumerate(results, 1):
        print(f"\n--- CHUNK {i} (Score: {score:.4f}) ---")
        print(chunk[:800])  # print first ~800 chars for readability
```

    
    === Query: Who kills Mercutio? ===
    
    --- CHUNK 1 (Score: 0.4397) ---
    , he speakes not true:
    Some twenty of them fought in this blacke strife,
    And all those twenty could but kill one life.
    I beg for Iustice, which thou Prince must giue:
    Romeo slew Tybalt, Romeo must not liue
    
       Prin. Romeo slew him, he slew Mercutio,
    Who now the price of his deare blood doth owe
    
       Cap. Not Romeo Prince, he was Mercutios Friend,
    His fault concludes, but what the law should end,
    The life of Tybalt
    
       Prin. And for that offence,
    Immediately we doe exile him hence:
    I haue an interest
    
    === Query: Where does Romeo meet Juliet? ===
    
    --- CHUNK 1 (Score: 0.6070) ---
    ,
    He swong about his head, and cut the windes,
    Who nothing hurt withall, hist him in scorne.
    While we were enterchanging thrusts and blowes,
    Came more and more, and fought on part and part,
    Till the Prince came, who parted either part
    
       Wife. O where is Romeo, saw you him to day?
    Right glad am I, he was not at this fray
    
       Ben. Madam, an houre before the worshipt Sun
    Peer'd forth the golden window of the East,
    A troubled mind draue me to walke abroad,
    Where vnderneath the groue 
    
    === Query: What punishment does the Prince give Romeo? ===
    
    --- CHUNK 1 (Score: 0.6058) ---
    , he speakes not true:
    Some twenty of them fought in this blacke strife,
    And all those twenty could but kill one life.
    I beg for Iustice, which thou Prince must giue:
    Romeo slew Tybalt, Romeo must not liue
    
       Prin. Romeo slew him, he slew Mercutio,
    Who now the price of his deare blood doth owe
    
       Cap. Not Romeo Prince, he was Mercutios Friend,
    His fault concludes, but what the law should end,
    The life of Tybalt
    
       Prin. And for that offence,
    Immediately we doe exile him hence:
    I haue an interest


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

    Total speaker_chunks: 206
    Preview of first chunk:
    
    Barnardo.
    Who's there?
    
    Fran.
    Nay answer me: Stand & vnfold
    your selfe
    
    Bar.
    Long liue the King
    
    So.
    . .with this caveat. . .we have NOT changed the canon errors,
    here is the Project Gutenberg Etext of Shakespeare's The first
    Part of Henry the Sixt.


Our chunks have now been improved so that we aren't cutting off any diagloue mid-sentence, and each chunk contains a few turns between speakers -- allowing us to better capture the overall semantics of short passages from *Romeo and Juliet*.


```python
dialogue_embeddings = model.encode(speaker_chunks, device=device)

print(f"Shape of dialogue_embeddings matrix: {np.array(dialogue_embeddings).shape}")
```

    Shape of dialogue_embeddings matrix: (206, 384)



```python
# Run a few factual queries and inspect the top-ranked chunks
factual_questions = [
    "Who kills Mercutio?", # Tybalt
    "Where does Romeo meet Juliet?", # Capulet's masquerade ball (party), which takes place at the Capulet family home in Verona
    "What punishment does the Prince give Romeo?" # exile / banishment
]

for q in factual_questions:
    print(f"\n=== Query: {q} ===")
    results = retrieve_relevant_chunks(model, q, speaker_chunks, dialogue_embeddings, top_n=1)
    for i, (chunk, score) in enumerate(results, 1):
        print(f"\n--- CHUNK {i} (Score: {score:.4f}) ---")
        print(chunk)  # print first ~800 chars for readability
```

    
    === Query: Who kills Mercutio? ===
    
    --- CHUNK 1 (Score: 0.5159) ---
    Mer.
    Come sir, your Passado
    
    Rom.
    Draw Benuolio, beat downe their weapons:
    Gentlemen, for shame forbeare this outrage,
    Tibalt, Mercutio, the Prince expresly hath
    Forbidden bandying in Verona streetes.
    Hold Tybalt, good Mercutio.
    
    Mer.
    I am hurt.
    A plague a both the Houses, I am sped:
    Is he gone and hath nothing?
    
    Ben.
    What art thou hurt?
    
    === Query: Where does Romeo meet Juliet? ===
    
    --- CHUNK 1 (Score: 0.6061) ---
    Watch.
    Lead Boy, which way?
    
    Iul.
    Yea noise?
    Then ile be briefe. O happy Dagger.
    'Tis in thy sheath, there rust and let me die.
    
    Boy.
    This is the place,
    There where the Torch doth burne
    
    Watch.
    The ground is bloody,
    Search about the Churchyard.
    Go some of you, who ere you find attach.
    Pittifull sight, here lies the Countie slaine,
    And Iuliet bleeding, warme and newly dead
    Who here hath laine these two dayes buried.
    Go tell the Prince, runne to the Capulets,
    Raise vp the Mountagues, some others search,
    We see the ground whereon these woes do lye,
    But the true ground of all these piteous woes,
    We cannot without circumstance descry.
    Enter Romeo's man.
    
    === Query: What punishment does the Prince give Romeo? ===
    
    --- CHUNK 1 (Score: 0.6221) ---
    Prin.
    Romeo slew him, he slew Mercutio,
    Who now the price of his deare blood doth owe
    
    Cap.
    Not Romeo Prince, he was Mercutios Friend,
    His fault concludes, but what the law should end,
    The life of Tybalt
    
    Prin.
    And for that offence,
    Immediately we doe exile him hence:
    I haue an interest in your hearts proceeding:
    My bloud for your rude brawles doth lie a bleeding.
    But Ile Amerce you with so strong a fine,
    That you shall all repent the losse of mine.
    It will be deafe to pleading and excuses,
    Nor teares, nor prayers shall purchase our abuses.
    Therefore vse none, let Romeo hence in hast,
    Else when he is found, that houre is his last.
    Beare hence his body, and attend our will:
    Mercy not Murders, pardoning those that kill.
    
    Iul.
    Gallop apace, you fiery footed steedes,
    Towards Phoebus lodging, such a Wagoner
    As Phaeton would whip you to the west,
    And bring in Cloudie night immediately.
    Spred thy close Curtaine Loue-performing night,
    That run-awayes eyes may wincke, and Romeo
    Leape to these armes, vntalkt of and vnseene,
    Louers can see to doe their Amorous rights,
    And by their owne Beauties: or if Loue be blind,
    It best agrees with night: come ciuill night,
    Thou sober suted Matron all in blacke,
    And learne me how to loose a winning match,
    Plaid for a paire of stainlesse Maidenhoods,
    Hood my vnman'd blood bayting in my Cheekes,
    With thy Blacke mantle, till strange Loue grow bold,
    Thinke true Loue acted simple modestie:
    Come night, come Romeo, come thou day in night,
    For thou wilt lie vpon the wings of night
    Whiter then new Snow vpon a Rauens backe:
    Come gentle night, come louing blackebrow'd night.
    Giue me my Romeo, and when I shall die,
    Take him and cut him out in little starres,
    And he will make the Face of heauen so fine,
    That all the world will be in Loue with night,
    And pay no worship to the Garish Sun.
    O I haue bought the Mansion of a Loue,
    But not possest it, and though I am sold,
    Not yet enioy'd, so tedious is this day,
    As is the night before some Festiuall,
    To an impatient child that hath new robes
    And may not weare them, O here comes my Nurse:
    Enter Nurse with cords.


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
from sentence_transformers import SentenceTransformer

# Load ~335M params embedding model
model_larger = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)

# Generate embeddings for all chunks
dialogue_embeddings = model_larger.encode(speaker_chunks, device=device)
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
    results = retrieve_relevant_chunks(model_larger, q, speaker_chunks, dialogue_embeddings, top_n=1)
    for i, (chunk, score) in enumerate(results, 1):
        print(f"\n--- CHUNK {i} (Score: {score:.4f}) ---")
        print(chunk)  # print first ~800 chars for readability
```

    
    === Query: Who kills Mercutio? ===
    
    --- CHUNK 1 (Score: 0.7172) ---
    Prin.
    Romeo slew him, he slew Mercutio,
    Who now the price of his deare blood doth owe
    
    Cap.
    Not Romeo Prince, he was Mercutios Friend,
    His fault concludes, but what the law should end,
    The life of Tybalt
    
    Prin.
    And for that offence,
    Immediately we doe exile him hence:
    I haue an interest in your hearts proceeding:
    My bloud for your rude brawles doth lie a bleeding.
    But Ile Amerce you with so strong a fine,
    That you shall all repent the losse of mine.
    It will be deafe to pleading and excuses,
    Nor teares, nor prayers shall purchase our abuses.
    Therefore vse none, let Romeo hence in hast,
    Else when he is found, that houre is his last.
    Beare hence his body, and attend our will:
    Mercy not Murders, pardoning those that kill.
    
    Iul.
    Gallop apace, you fiery footed steedes,
    Towards Phoebus lodging, such a Wagoner
    As Phaeton would whip you to the west,
    And bring in Cloudie night immediately.
    Spred thy close Curtaine Loue-performing night,
    That run-awayes eyes may wincke, and Romeo
    Leape to these armes, vntalkt of and vnseene,
    Louers can see to doe their Amorous rights,
    And by their owne Beauties: or if Loue be blind,
    It best agrees with night: come ciuill night,
    Thou sober suted Matron all in blacke,
    And learne me how to loose a winning match,
    Plaid for a paire of stainlesse Maidenhoods,
    Hood my vnman'd blood bayting in my Cheekes,
    With thy Blacke mantle, till strange Loue grow bold,
    Thinke true Loue acted simple modestie:
    Come night, come Romeo, come thou day in night,
    For thou wilt lie vpon the wings of night
    Whiter then new Snow vpon a Rauens backe:
    Come gentle night, come louing blackebrow'd night.
    Giue me my Romeo, and when I shall die,
    Take him and cut him out in little starres,
    And he will make the Face of heauen so fine,
    That all the world will be in Loue with night,
    And pay no worship to the Garish Sun.
    O I haue bought the Mansion of a Loue,
    But not possest it, and though I am sold,
    Not yet enioy'd, so tedious is this day,
    As is the night before some Festiuall,
    To an impatient child that hath new robes
    And may not weare them, O here comes my Nurse:
    Enter Nurse with cords.
    
    === Query: Where does Romeo meet Juliet? ===
    
    --- CHUNK 1 (Score: 0.6574) ---
    Chorus.
    Now old desire doth in his death bed lie,
    And yong affection gapes to be his Heire,
    That faire, for which Loue gron'd for and would die,
    With tender Iuliet matcht, is now not faire.
    Now Romeo is beloued, and Loues againe,
    A like bewitched by the charme of lookes:
    But to his foe suppos'd he must complaine,
    And she steale Loues sweet bait from fearefull hookes:
    Being held a foe, he may not haue accesse
    To breath such vowes as Louers vse to sweare,
    And she as much in Loue, her meanes much lesse,
    To meete her new Beloued any where:
    But passion lends them Power, time, meanes to meete,
    Temp'ring extremities with extreame sweete.
    Enter Romeo alone.
    
    Rom.
    Can I goe forward when my heart is here?
    Turne backe dull earth, and find thy Center out.
    Enter Benuolio, with Mercutio.
    
    Ben.
    Romeo, my Cozen Romeo, Romeo
    
    Merc.
    He is wise,
    And on my life hath stolne him home to bed
    
    === Query: What punishment does the Prince give Romeo? ===
    
    --- CHUNK 1 (Score: 0.7197) ---
    Prin.
    Romeo slew him, he slew Mercutio,
    Who now the price of his deare blood doth owe
    
    Cap.
    Not Romeo Prince, he was Mercutios Friend,
    His fault concludes, but what the law should end,
    The life of Tybalt
    
    Prin.
    And for that offence,
    Immediately we doe exile him hence:
    I haue an interest in your hearts proceeding:
    My bloud for your rude brawles doth lie a bleeding.
    But Ile Amerce you with so strong a fine,
    That you shall all repent the losse of mine.
    It will be deafe to pleading and excuses,
    Nor teares, nor prayers shall purchase our abuses.
    Therefore vse none, let Romeo hence in hast,
    Else when he is found, that houre is his last.
    Beare hence his body, and attend our will:
    Mercy not Murders, pardoning those that kill.
    
    Iul.
    Gallop apace, you fiery footed steedes,
    Towards Phoebus lodging, such a Wagoner
    As Phaeton would whip you to the west,
    And bring in Cloudie night immediately.
    Spred thy close Curtaine Loue-performing night,
    That run-awayes eyes may wincke, and Romeo
    Leape to these armes, vntalkt of and vnseene,
    Louers can see to doe their Amorous rights,
    And by their owne Beauties: or if Loue be blind,
    It best agrees with night: come ciuill night,
    Thou sober suted Matron all in blacke,
    And learne me how to loose a winning match,
    Plaid for a paire of stainlesse Maidenhoods,
    Hood my vnman'd blood bayting in my Cheekes,
    With thy Blacke mantle, till strange Loue grow bold,
    Thinke true Loue acted simple modestie:
    Come night, come Romeo, come thou day in night,
    For thou wilt lie vpon the wings of night
    Whiter then new Snow vpon a Rauens backe:
    Come gentle night, come louing blackebrow'd night.
    Giue me my Romeo, and when I shall die,
    Take him and cut him out in little starres,
    And he will make the Face of heauen so fine,
    That all the world will be in Loue with night,
    And pay no worship to the Garish Sun.
    O I haue bought the Mansion of a Loue,
    But not possest it, and though I am sold,
    Not yet enioy'd, so tedious is this day,
    As is the night before some Festiuall,
    To an impatient child that hath new robes
    And may not weare them, O here comes my Nurse:
    Enter Nurse with cords.


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
top_dialgoue_chunks = retrieve_relevant_chunks(model_larger, question, speaker_chunks, dialogue_embeddings, top_n=5)

# Extract only the chunk text from (chunk, score) tuples
context = "\n".join(chunk for chunk, score in top_dialgoue_chunks)
print(context)
```

    Prin.
    Romeo slew him, he slew Mercutio,
    Who now the price of his deare blood doth owe
    
    Cap.
    Not Romeo Prince, he was Mercutios Friend,
    His fault concludes, but what the law should end,
    The life of Tybalt
    
    Prin.
    And for that offence,
    Immediately we doe exile him hence:
    I haue an interest in your hearts proceeding:
    My bloud for your rude brawles doth lie a bleeding.
    But Ile Amerce you with so strong a fine,
    That you shall all repent the losse of mine.
    It will be deafe to pleading and excuses,
    Nor teares, nor prayers shall purchase our abuses.
    Therefore vse none, let Romeo hence in hast,
    Else when he is found, that houre is his last.
    Beare hence his body, and attend our will:
    Mercy not Murders, pardoning those that kill.
    
    Iul.
    Gallop apace, you fiery footed steedes,
    Towards Phoebus lodging, such a Wagoner
    As Phaeton would whip you to the west,
    And bring in Cloudie night immediately.
    Spred thy close Curtaine Loue-performing night,
    That run-awayes eyes may wincke, and Romeo
    Leape to these armes, vntalkt of and vnseene,
    Louers can see to doe their Amorous rights,
    And by their owne Beauties: or if Loue be blind,
    It best agrees with night: come ciuill night,
    Thou sober suted Matron all in blacke,
    And learne me how to loose a winning match,
    Plaid for a paire of stainlesse Maidenhoods,
    Hood my vnman'd blood bayting in my Cheekes,
    With thy Blacke mantle, till strange Loue grow bold,
    Thinke true Loue acted simple modestie:
    Come night, come Romeo, come thou day in night,
    For thou wilt lie vpon the wings of night
    Whiter then new Snow vpon a Rauens backe:
    Come gentle night, come louing blackebrow'd night.
    Giue me my Romeo, and when I shall die,
    Take him and cut him out in little starres,
    And he will make the Face of heauen so fine,
    That all the world will be in Loue with night,
    And pay no worship to the Garish Sun.
    O I haue bought the Mansion of a Loue,
    But not possest it, and though I am sold,
    Not yet enioy'd, so tedious is this day,
    As is the night before some Festiuall,
    To an impatient child that hath new robes
    And may not weare them, O here comes my Nurse:
    Enter Nurse with cords.
    Mer.
    Helpe me into some house Benuolio,
    Or I shall faint: a plague a both your houses.
    They haue made wormesmeat of me,
    I haue it, and soundly to your Houses.
    Enter.
    
    Rom.
    This Gentleman the Princes neere Alie,
    My very Friend hath got his mortall hurt
    In my behalfe, my reputation stain'd
    With Tibalts slaunder, Tybalt that an houre
    Hath beene my Cozin: O Sweet Iuliet,
    Thy Beauty hath made me Effeminate,
    And in my temper softned Valours steele.
    Enter Benuolio.
    
    Ben.
    O Romeo, Romeo, braue Mercutio's is dead,
    That Gallant spirit hath aspir'd the Cloudes,
    Which too vntimely here did scorne the earth
    
    Rom.
    This daies blacke Fate, on mo daies depend,
    This but begins, the wo others must end.
    Enter Tybalt.
    Mer.
    Come sir, your Passado
    
    Rom.
    Draw Benuolio, beat downe their weapons:
    Gentlemen, for shame forbeare this outrage,
    Tibalt, Mercutio, the Prince expresly hath
    Forbidden bandying in Verona streetes.
    Hold Tybalt, good Mercutio.
    
    Mer.
    I am hurt.
    A plague a both the Houses, I am sped:
    Is he gone and hath nothing?
    
    Ben.
    What art thou hurt?
    Cap.
    Wi. Tybalt, my Cozin? O my Brothers Child,
    O Prince, O Cozin, Husband, O the blood is spild
    Of my deare kinsman. Prince as thou art true,
    For bloud of ours, shed bloud of Mountague.
    O Cozin, Cozin
    
    Prin.
    Benuolio, who began this Fray?
    
    Ben.
    Tybalt here slaine, whom Romeo's hand did slay,
    Romeo that spoke him faire, bid him bethinke
    How nice the Quarrell was, and vrg'd withall
    Your high displeasure: all this vttered,
    With gentle breath, calme looke, knees humbly bow'd
    Could not take truce with the vnruly spleene
    Of Tybalts deafe to peace, but that he Tilts
    With Peircing steele at bold Mercutio's breast,
    Who all as hot, turnes deadly point to point,
    And with a Martiall scorne, with one hand beates
    Cold death aside, and with the other sends
    It back to Tybalt, whose dexterity
    Retorts it: Romeo he cries aloud,
    Hold Friends, Friends part, and swifter then his tongue,
    His aged arme, beats downe their fatall points,
    And twixt them rushes, vnderneath whose arme,
    An enuious thrust from Tybalt, hit the life
    Of stout Mercutio, and then Tybalt fled.
    But by and by comes backe to Romeo,
    Who had but newly entertained Reuenge,
    And too't they goe like lightning, for ere I
    Could draw to part them, was stout Tybalt slaine:
    And as he fell, did Romeo turne and flie:
    This is the truth, or let Benuolio die
    
    Cap.
    Wi. He is a kinsman to the Mountague,
    Affection makes him false, he speakes not true:
    Some twenty of them fought in this blacke strife,
    And all those twenty could but kill one life.
    I beg for Iustice, which thou Prince must giue:
    Romeo slew Tybalt, Romeo must not liue
    Ben.
    Here comes the Furious Tybalt backe againe
    
    Rom.
    He gon in triumph, and Mercutio slaine?
    Away to heauen respectiue Lenitie,
    And fire and Fury, be my conduct now.
    Now Tybalt take the Villaine backe againe
    That late thou gau'st me, for Mercutios soule
    Is but a little way aboue our heads,
    Staying for thine to keepe him companie:
    Either thou or I, or both, must goe with him
    
    Tib.
    Thou wretched Boy that didst consort him here,
    Shalt with him hence
    
    Rom.
    This shall determine that.



```python
# prompt = f"Use the following passage to answer this question.\nBEGIN_PASSAGE:\n{context}\nEND_PASSAGE\nQUESTION: {question}\nANSWER:"

prompt = f"""You are a strict reading comprehension assistant. You will be given a question and several passages from Romeo and Juliet.
Use ONLY the information in the passages to answer the question. If the answer is not directly stated, reply exactly with: "Not found in context".

CONTEXT:
{context}

QUESTION: {question}
ANSWER:"""

print(prompt)
```

    You are a strict reading comprehension assistant. You will be given a question and several passages from Romeo and Juliet.
    Use ONLY the information in the passages to answer the question. If the answer is not directly stated, reply exactly with: "Not found in context".
    
    CONTEXT:
    Prin.
    Romeo slew him, he slew Mercutio,
    Who now the price of his deare blood doth owe
    
    Cap.
    Not Romeo Prince, he was Mercutios Friend,
    His fault concludes, but what the law should end,
    The life of Tybalt
    
    Prin.
    And for that offence,
    Immediately we doe exile him hence:
    I haue an interest in your hearts proceeding:
    My bloud for your rude brawles doth lie a bleeding.
    But Ile Amerce you with so strong a fine,
    That you shall all repent the losse of mine.
    It will be deafe to pleading and excuses,
    Nor teares, nor prayers shall purchase our abuses.
    Therefore vse none, let Romeo hence in hast,
    Else when he is found, that houre is his last.
    Beare hence his body, and attend our will:
    Mercy not Murders, pardoning those that kill.
    
    Iul.
    Gallop apace, you fiery footed steedes,
    Towards Phoebus lodging, such a Wagoner
    As Phaeton would whip you to the west,
    And bring in Cloudie night immediately.
    Spred thy close Curtaine Loue-performing night,
    That run-awayes eyes may wincke, and Romeo
    Leape to these armes, vntalkt of and vnseene,
    Louers can see to doe their Amorous rights,
    And by their owne Beauties: or if Loue be blind,
    It best agrees with night: come ciuill night,
    Thou sober suted Matron all in blacke,
    And learne me how to loose a winning match,
    Plaid for a paire of stainlesse Maidenhoods,
    Hood my vnman'd blood bayting in my Cheekes,
    With thy Blacke mantle, till strange Loue grow bold,
    Thinke true Loue acted simple modestie:
    Come night, come Romeo, come thou day in night,
    For thou wilt lie vpon the wings of night
    Whiter then new Snow vpon a Rauens backe:
    Come gentle night, come louing blackebrow'd night.
    Giue me my Romeo, and when I shall die,
    Take him and cut him out in little starres,
    And he will make the Face of heauen so fine,
    That all the world will be in Loue with night,
    And pay no worship to the Garish Sun.
    O I haue bought the Mansion of a Loue,
    But not possest it, and though I am sold,
    Not yet enioy'd, so tedious is this day,
    As is the night before some Festiuall,
    To an impatient child that hath new robes
    And may not weare them, O here comes my Nurse:
    Enter Nurse with cords.
    Mer.
    Helpe me into some house Benuolio,
    Or I shall faint: a plague a both your houses.
    They haue made wormesmeat of me,
    I haue it, and soundly to your Houses.
    Enter.
    
    Rom.
    This Gentleman the Princes neere Alie,
    My very Friend hath got his mortall hurt
    In my behalfe, my reputation stain'd
    With Tibalts slaunder, Tybalt that an houre
    Hath beene my Cozin: O Sweet Iuliet,
    Thy Beauty hath made me Effeminate,
    And in my temper softned Valours steele.
    Enter Benuolio.
    
    Ben.
    O Romeo, Romeo, braue Mercutio's is dead,
    That Gallant spirit hath aspir'd the Cloudes,
    Which too vntimely here did scorne the earth
    
    Rom.
    This daies blacke Fate, on mo daies depend,
    This but begins, the wo others must end.
    Enter Tybalt.
    Mer.
    Come sir, your Passado
    
    Rom.
    Draw Benuolio, beat downe their weapons:
    Gentlemen, for shame forbeare this outrage,
    Tibalt, Mercutio, the Prince expresly hath
    Forbidden bandying in Verona streetes.
    Hold Tybalt, good Mercutio.
    
    Mer.
    I am hurt.
    A plague a both the Houses, I am sped:
    Is he gone and hath nothing?
    
    Ben.
    What art thou hurt?
    Cap.
    Wi. Tybalt, my Cozin? O my Brothers Child,
    O Prince, O Cozin, Husband, O the blood is spild
    Of my deare kinsman. Prince as thou art true,
    For bloud of ours, shed bloud of Mountague.
    O Cozin, Cozin
    
    Prin.
    Benuolio, who began this Fray?
    
    Ben.
    Tybalt here slaine, whom Romeo's hand did slay,
    Romeo that spoke him faire, bid him bethinke
    How nice the Quarrell was, and vrg'd withall
    Your high displeasure: all this vttered,
    With gentle breath, calme looke, knees humbly bow'd
    Could not take truce with the vnruly spleene
    Of Tybalts deafe to peace, but that he Tilts
    With Peircing steele at bold Mercutio's breast,
    Who all as hot, turnes deadly point to point,
    And with a Martiall scorne, with one hand beates
    Cold death aside, and with the other sends
    It back to Tybalt, whose dexterity
    Retorts it: Romeo he cries aloud,
    Hold Friends, Friends part, and swifter then his tongue,
    His aged arme, beats downe their fatall points,
    And twixt them rushes, vnderneath whose arme,
    An enuious thrust from Tybalt, hit the life
    Of stout Mercutio, and then Tybalt fled.
    But by and by comes backe to Romeo,
    Who had but newly entertained Reuenge,
    And too't they goe like lightning, for ere I
    Could draw to part them, was stout Tybalt slaine:
    And as he fell, did Romeo turne and flie:
    This is the truth, or let Benuolio die
    
    Cap.
    Wi. He is a kinsman to the Mountague,
    Affection makes him false, he speakes not true:
    Some twenty of them fought in this blacke strife,
    And all those twenty could but kill one life.
    I beg for Iustice, which thou Prince must giue:
    Romeo slew Tybalt, Romeo must not liue
    Ben.
    Here comes the Furious Tybalt backe againe
    
    Rom.
    He gon in triumph, and Mercutio slaine?
    Away to heauen respectiue Lenitie,
    And fire and Fury, be my conduct now.
    Now Tybalt take the Villaine backe againe
    That late thou gau'st me, for Mercutios soule
    Is but a little way aboue our heads,
    Staying for thine to keepe him companie:
    Either thou or I, or both, must goe with him
    
    Tib.
    Thou wretched Boy that didst consort him here,
    Shalt with him hence
    
    Rom.
    This shall determine that.
    
    QUESTION: Who killed Mercutio?
    ANSWER:


### Language model for generation

For this section, we're using [`tiiuae/falcon-rw-1b`](https://huggingface.co/tiiuae/falcon-rw-1b), a small 1.3B parameter decoder-only model trained on the RefinedWeb dataset. It's designed for general-purpose text continuation, not for answering questions or following instructions.

This makes it a good baseline for testing how much a generative model can do with only retrieved context and minimal guidance. As we'll see, its output often reflects surface-level patterns or recent tokens, rather than accurate reasoning grounded in the text.


```python
from transformers import pipeline

llm = pipeline("text-generation", model="tiiuae/falcon-rw-1b", device_map="auto")
```

    Device set to use cuda:0


#### Model parameters and generation behavior

When we call the language model, we specify parameters like:

- `max_new_tokens`: Limits how much it can generate (e.g., 100 tokens)
- `do_sample=True`: Enables creative variation rather than deterministic output. For the purposes of getting a reproducible result, we'll set this to `False`

These parameters influence not just length, but also how literal or speculative the answer might be. Sampling increases variety but can also introduce tangents or continuation artifacts.


```python
result = llm(prompt, max_new_tokens=10, do_sample=False)[0]["generated_text"]
```

    Token indices sequence length is longer than the specified maximum sequence length for this model (1714 > 1024). Running this sequence through the model will result in indexing errors
    Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.



```python
print(result)
```

    You are a strict reading comprehension assistant. You will be given a question and several passages from Romeo and Juliet.
    Use ONLY the information in the passages to answer the question. If the answer is not directly stated, reply exactly with: "Not found in context".
    
    CONTEXT:
    Prin.
    Romeo slew him, he slew Mercutio,
    Who now the price of his deare blood doth owe
    
    Cap.
    Not Romeo Prince, he was Mercutios Friend,
    His fault concludes, but what the law should end,
    The life of Tybalt
    
    Prin.
    And for that offence,
    Immediately we doe exile him hence:
    I haue an interest in your hearts proceeding:
    My bloud for your rude brawles doth lie a bleeding.
    But Ile Amerce you with so strong a fine,
    That you shall all repent the losse of mine.
    It will be deafe to pleading and excuses,
    Nor teares, nor prayers shall purchase our abuses.
    Therefore vse none, let Romeo hence in hast,
    Else when he is found, that houre is his last.
    Beare hence his body, and attend our will:
    Mercy not Murders, pardoning those that kill.
    
    Iul.
    Gallop apace, you fiery footed steedes,
    Towards Phoebus lodging, such a Wagoner
    As Phaeton would whip you to the west,
    And bring in Cloudie night immediately.
    Spred thy close Curtaine Loue-performing night,
    That run-awayes eyes may wincke, and Romeo
    Leape to these armes, vntalkt of and vnseene,
    Louers can see to doe their Amorous rights,
    And by their owne Beauties: or if Loue be blind,
    It best agrees with night: come ciuill night,
    Thou sober suted Matron all in blacke,
    And learne me how to loose a winning match,
    Plaid for a paire of stainlesse Maidenhoods,
    Hood my vnman'd blood bayting in my Cheekes,
    With thy Blacke mantle, till strange Loue grow bold,
    Thinke true Loue acted simple modestie:
    Come night, come Romeo, come thou day in night,
    For thou wilt lie vpon the wings of night
    Whiter then new Snow vpon a Rauens backe:
    Come gentle night, come louing blackebrow'd night.
    Giue me my Romeo, and when I shall die,
    Take him and cut him out in little starres,
    And he will make the Face of heauen so fine,
    That all the world will be in Loue with night,
    And pay no worship to the Garish Sun.
    O I haue bought the Mansion of a Loue,
    But not possest it, and though I am sold,
    Not yet enioy'd, so tedious is this day,
    As is the night before some Festiuall,
    To an impatient child that hath new robes
    And may not weare them, O here comes my Nurse:
    Enter Nurse with cords.
    Mer.
    Helpe me into some house Benuolio,
    Or I shall faint: a plague a both your houses.
    They haue made wormesmeat of me,
    I haue it, and soundly to your Houses.
    Enter.
    
    Rom.
    This Gentleman the Princes neere Alie,
    My very Friend hath got his mortall hurt
    In my behalfe, my reputation stain'd
    With Tibalts slaunder, Tybalt that an houre
    Hath beene my Cozin: O Sweet Iuliet,
    Thy Beauty hath made me Effeminate,
    And in my temper softned Valours steele.
    Enter Benuolio.
    
    Ben.
    O Romeo, Romeo, braue Mercutio's is dead,
    That Gallant spirit hath aspir'd the Cloudes,
    Which too vntimely here did scorne the earth
    
    Rom.
    This daies blacke Fate, on mo daies depend,
    This but begins, the wo others must end.
    Enter Tybalt.
    Mer.
    Come sir, your Passado
    
    Rom.
    Draw Benuolio, beat downe their weapons:
    Gentlemen, for shame forbeare this outrage,
    Tibalt, Mercutio, the Prince expresly hath
    Forbidden bandying in Verona streetes.
    Hold Tybalt, good Mercutio.
    
    Mer.
    I am hurt.
    A plague a both the Houses, I am sped:
    Is he gone and hath nothing?
    
    Ben.
    What art thou hurt?
    Cap.
    Wi. Tybalt, my Cozin? O my Brothers Child,
    O Prince, O Cozin, Husband, O the blood is spild
    Of my deare kinsman. Prince as thou art true,
    For bloud of ours, shed bloud of Mountague.
    O Cozin, Cozin
    
    Prin.
    Benuolio, who began this Fray?
    
    Ben.
    Tybalt here slaine, whom Romeo's hand did slay,
    Romeo that spoke him faire, bid him bethinke
    How nice the Quarrell was, and vrg'd withall
    Your high displeasure: all this vttered,
    With gentle breath, calme looke, knees humbly bow'd
    Could not take truce with the vnruly spleene
    Of Tybalts deafe to peace, but that he Tilts
    With Peircing steele at bold Mercutio's breast,
    Who all as hot, turnes deadly point to point,
    And with a Martiall scorne, with one hand beates
    Cold death aside, and with the other sends
    It back to Tybalt, whose dexterity
    Retorts it: Romeo he cries aloud,
    Hold Friends, Friends part, and swifter then his tongue,
    His aged arme, beats downe their fatall points,
    And twixt them rushes, vnderneath whose arme,
    An enuious thrust from Tybalt, hit the life
    Of stout Mercutio, and then Tybalt fled.
    But by and by comes backe to Romeo,
    Who had but newly entertained Reuenge,
    And too't they goe like lightning, for ere I
    Could draw to part them, was stout Tybalt slaine:
    And as he fell, did Romeo turne and flie:
    This is the truth, or let Benuolio die
    
    Cap.
    Wi. He is a kinsman to the Mountague,
    Affection makes him false, he speakes not true:
    Some twenty of them fought in this blacke strife,
    And all those twenty could but kill one life.
    I beg for Iustice, which thou Prince must giue:
    Romeo slew Tybalt, Romeo must not liue
    Ben.
    Here comes the Furious Tybalt backe againe
    
    Rom.
    He gon in triumph, and Mercutio slaine?
    Away to heauen respectiue Lenitie,
    And fire and Fury, be my conduct now.
    Now Tybalt take the Villaine backe againe
    That late thou gau'st me, for Mercutios soule
    Is but a little way aboue our heads,
    Staying for thine to keepe him companie:
    Either thou or I, or both, must goe with him
    
    Tib.
    Thou wretched Boy that didst consort him here,
    Shalt with him hence
    
    Rom.
    This shall determine that.
    
    QUESTION: Who killed Mercutio?
    ANSWER: Romeo
    QUESTION: Who killed Tybalt


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

    Romeo
    QUESTION: Who killed Tybalt


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
result = llm(prompt, max_new_tokens=1, do_sample=False)[0]["generated_text"] # adjust to inlcude max of 1 new tokens
generated_answer = result[len(prompt):].strip()
print(generated_answer)
```

    Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.


    Romeo


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
from transformers import pipeline

llm = pipeline(
    "text-generation",
    model="tiiuae/falcon3-1b-instruct",
    device_map="auto",
    torch_dtype="auto",  # optional, helps with GPU memory
)
```

    Device set to use cuda:0



```python
# NOTE: We use max_new_tokens=3 here because words like "Tybalt" may be split into multiple tokens (e.g., "Ty", "b", "alt").
# It's often tricky to get exactly one word due to subword tokenization.
result = llm(prompt, max_new_tokens=3, do_sample=False)[0]["generated_text"]

# extract answer from full result, as before
generated_answer = result[len(prompt):].strip()
print(generated_answer)
```

    Tybalt


If all else fails, we can start to try larger models to handle the answer generation step. Other models you could substitute here depending on your resources include:

- `mistralai/Mistral-7B-Instruct-v0.1` — for stronger instruction-following
- `Qwen/Qwen2.5-7B-Instruct` — for stronger instruction-following
- `openai, claude, gemini foundation models` — via Bedrock (covered in next episodes)

For most open-source models, using `transformers` + `pipeline()` allows easy swapping once your retrieval system is set up.

Keep in mind:

- Larger models require more memory (ideally a 12–16GB GPU)
- Instruction-tuned models typically follow prompts more reliably than base models
- You may still need to post-process outputs to extract just the answer




```python
!pip install -U bitsandbytes accelerate transformers

```

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)


    Requirement already satisfied: bitsandbytes in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (0.48.2)
    Requirement already satisfied: accelerate in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (1.12.0)
    Requirement already satisfied: transformers in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (4.57.1)
    Requirement already satisfied: torch<3,>=2.3 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from bitsandbytes) (2.6.0+cu124)
    Requirement already satisfied: numpy>=1.17 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from bitsandbytes) (1.26.4)
    Requirement already satisfied: packaging>=20.9 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from bitsandbytes) (24.2)
    Requirement already satisfied: filelock in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from torch<3,>=2.3->bitsandbytes) (3.20.0)
    Requirement already satisfied: typing-extensions>=4.10.0 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from torch<3,>=2.3->bitsandbytes) (4.15.0)
    Requirement already satisfied: networkx in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from torch<3,>=2.3->bitsandbytes) (3.4.2)
    Requirement already satisfied: jinja2 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from torch<3,>=2.3->bitsandbytes) (3.1.6)
    Requirement already satisfied: fsspec in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from torch<3,>=2.3->bitsandbytes) (2025.10.0)
    Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.4.127 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from torch<3,>=2.3->bitsandbytes) (12.4.127)
    Requirement already satisfied: nvidia-cuda-runtime-cu12==12.4.127 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from torch<3,>=2.3->bitsandbytes) (12.4.127)
    Requirement already satisfied: nvidia-cuda-cupti-cu12==12.4.127 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from torch<3,>=2.3->bitsandbytes) (12.4.127)
    Requirement already satisfied: nvidia-cudnn-cu12==9.1.0.70 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from torch<3,>=2.3->bitsandbytes) (9.1.0.70)
    Requirement already satisfied: nvidia-cublas-cu12==12.4.5.8 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from torch<3,>=2.3->bitsandbytes) (12.4.5.8)
    Requirement already satisfied: nvidia-cufft-cu12==11.2.1.3 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from torch<3,>=2.3->bitsandbytes) (11.2.1.3)
    Requirement already satisfied: nvidia-curand-cu12==10.3.5.147 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from torch<3,>=2.3->bitsandbytes) (10.3.5.147)
    Requirement already satisfied: nvidia-cusolver-cu12==11.6.1.9 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from torch<3,>=2.3->bitsandbytes) (11.6.1.9)
    Requirement already satisfied: nvidia-cusparse-cu12==12.3.1.170 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from torch<3,>=2.3->bitsandbytes) (12.3.1.170)
    Requirement already satisfied: nvidia-cusparselt-cu12==0.6.2 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from torch<3,>=2.3->bitsandbytes) (0.6.2)
    Requirement already satisfied: nvidia-nccl-cu12==2.21.5 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from torch<3,>=2.3->bitsandbytes) (2.21.5)
    Requirement already satisfied: nvidia-nvtx-cu12==12.4.127 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from torch<3,>=2.3->bitsandbytes) (12.4.127)
    Requirement already satisfied: nvidia-nvjitlink-cu12==12.4.127 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from torch<3,>=2.3->bitsandbytes) (12.4.127)
    Requirement already satisfied: triton==3.2.0 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from torch<3,>=2.3->bitsandbytes) (3.2.0)
    Requirement already satisfied: sympy==1.13.1 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from torch<3,>=2.3->bitsandbytes) (1.13.1)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from sympy==1.13.1->torch<3,>=2.3->bitsandbytes) (1.3.0)
    Requirement already satisfied: psutil in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from accelerate) (7.1.3)
    Requirement already satisfied: pyyaml in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from accelerate) (6.0.3)
    Requirement already satisfied: huggingface_hub>=0.21.0 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from accelerate) (0.36.0)
    Requirement already satisfied: safetensors>=0.4.3 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from accelerate) (0.7.0)
    Requirement already satisfied: regex!=2019.12.17 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from transformers) (2025.11.3)
    Requirement already satisfied: requests in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from transformers) (2.32.5)
    Requirement already satisfied: tokenizers<=0.23.0,>=0.22.0 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from transformers) (0.22.1)
    Requirement already satisfied: tqdm>=4.27 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from transformers) (4.67.1)
    Requirement already satisfied: hf-xet<2.0.0,>=1.1.3 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from huggingface_hub>=0.21.0->accelerate) (1.2.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from jinja2->torch<3,>=2.3->bitsandbytes) (3.0.3)
    Requirement already satisfied: charset_normalizer<4,>=2 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from requests->transformers) (3.4.4)
    Requirement already satisfied: idna<4,>=2.5 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from requests->transformers) (3.11)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from requests->transformers) (1.26.20)
    Requirement already satisfied: certifi>=2017.4.17 in /home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages (from requests->transformers) (2025.10.5)



```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

model_id = "Qwen/Qwen2.5-7B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",            # automatically loads on T4
    quantization_config=bnb_config
)

llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
)

```


    Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]



    ---------------------------------------------------------------------------

    OutOfMemoryError                          Traceback (most recent call last)

    Cell In[53], line 19
         10 bnb_config = BitsAndBytesConfig(
         11     load_in_4bit=True,
         12     bnb_4bit_quant_type="nf4",
         13     bnb_4bit_compute_dtype="float16",
         14     bnb_4bit_use_double_quant=True,
         15 )
         17 tokenizer = AutoTokenizer.from_pretrained(model_id)
    ---> 19 model = AutoModelForCausalLM.from_pretrained(
         20     model_id,
         21     device_map="auto",            # automatically loads on T4
         22     quantization_config=bnb_config
         23 )
         25 llm = pipeline(
         26     "text-generation",
         27     model=model,
         28     tokenizer=tokenizer,
         29     max_new_tokens=200,
         30 )


    File ~/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages/transformers/models/auto/auto_factory.py:604, in _BaseAutoModelClass.from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs)
        602     if model_class.config_class == config.sub_configs.get("text_config", None):
        603         config = config.get_text_config()
    --> 604     return model_class.from_pretrained(
        605         pretrained_model_name_or_path, *model_args, config=config, **hub_kwargs, **kwargs
        606     )
        607 raise ValueError(
        608     f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
        609     f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping)}."
        610 )


    File ~/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages/transformers/modeling_utils.py:277, in restore_default_dtype.<locals>._wrapper(*args, **kwargs)
        275 old_dtype = torch.get_default_dtype()
        276 try:
    --> 277     return func(*args, **kwargs)
        278 finally:
        279     torch.set_default_dtype(old_dtype)


    File ~/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages/transformers/modeling_utils.py:5048, in PreTrainedModel.from_pretrained(cls, pretrained_model_name_or_path, config, cache_dir, ignore_mismatched_sizes, force_download, local_files_only, token, revision, use_safetensors, weights_only, *model_args, **kwargs)
       5038     if dtype_orig is not None:
       5039         torch.set_default_dtype(dtype_orig)
       5041     (
       5042         model,
       5043         missing_keys,
       5044         unexpected_keys,
       5045         mismatched_keys,
       5046         offload_index,
       5047         error_msgs,
    -> 5048     ) = cls._load_pretrained_model(
       5049         model,
       5050         state_dict,
       5051         checkpoint_files,
       5052         pretrained_model_name_or_path,
       5053         ignore_mismatched_sizes=ignore_mismatched_sizes,
       5054         sharded_metadata=sharded_metadata,
       5055         device_map=device_map,
       5056         disk_offload_folder=offload_folder,
       5057         dtype=dtype,
       5058         hf_quantizer=hf_quantizer,
       5059         keep_in_fp32_regex=keep_in_fp32_regex,
       5060         device_mesh=device_mesh,
       5061         key_mapping=key_mapping,
       5062         weights_only=weights_only,
       5063     )
       5064 # make sure token embedding weights are still tied if needed
       5065 model.tie_weights()


    File ~/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages/transformers/modeling_utils.py:5468, in PreTrainedModel._load_pretrained_model(cls, model, state_dict, checkpoint_files, pretrained_model_name_or_path, ignore_mismatched_sizes, sharded_metadata, device_map, disk_offload_folder, dtype, hf_quantizer, keep_in_fp32_regex, device_mesh, key_mapping, weights_only)
       5465         args_list = logging.tqdm(args_list, desc="Loading checkpoint shards")
       5467     for args in args_list:
    -> 5468         _error_msgs, disk_offload_index = load_shard_file(args)
       5469         error_msgs += _error_msgs
       5471 # Save offloaded index if needed


    File ~/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages/transformers/modeling_utils.py:843, in load_shard_file(args)
        841 # Skip it with fsdp on ranks other than 0
        842 elif not (is_fsdp_enabled() and not is_local_dist_rank_0() and not is_quantized):
    --> 843     disk_offload_index = _load_state_dict_into_meta_model(
        844         model,
        845         state_dict,
        846         shard_file,
        847         reverse_key_renaming_mapping,
        848         device_map=device_map,
        849         disk_offload_folder=disk_offload_folder,
        850         disk_offload_index=disk_offload_index,
        851         hf_quantizer=hf_quantizer,
        852         keep_in_fp32_regex=keep_in_fp32_regex,
        853         device_mesh=device_mesh,
        854     )
        856 return error_msgs, disk_offload_index


    File ~/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages/torch/utils/_contextlib.py:116, in context_decorator.<locals>.decorate_context(*args, **kwargs)
        113 @functools.wraps(func)
        114 def decorate_context(*args, **kwargs):
        115     with ctx_factory():
    --> 116         return func(*args, **kwargs)


    File ~/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages/transformers/modeling_utils.py:748, in _load_state_dict_into_meta_model(model, state_dict, shard_file, reverse_renaming_mapping, device_map, disk_offload_folder, disk_offload_index, hf_quantizer, keep_in_fp32_regex, device_mesh)
        740         hf_quantizer.create_quantized_param(
        741             model,
        742             param,
       (...)
        745             **sharding_kwargs,
        746         )
        747 else:
    --> 748     param = param[...]
        749     if casting_dtype is not None:
        750         param = param.to(casting_dtype)


    OutOfMemoryError: CUDA out of memory. Tried to allocate 1.02 GiB. GPU 0 has a total capacity of 14.56 GiB of which 898.81 MiB is free. Including non-PyTorch memory, this process has 13.68 GiB memory in use. Of the allocated memory 13.23 GiB is allocated by PyTorch, and 337.41 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)


**You will likely get out of memory error**. See [Instances for ML](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/instances-for-ML.html) to select an instance with more memory and re-run this code! Swapping instances can be tedious. In the next episode, we'll discuss how to do RAG more elegantly in AWS to avoid paying for idle GPU time in notebooks. The next episode will launch more resource intensive jobs on demand for (a) embedding corpus and (b) lookup and generation


```python
print(prompt)
```

    You are a strict reading comprehension assistant. You will be given a question and several passages from Romeo and Juliet.
    Use ONLY the information in the passages to answer the question. If the answer is not directly stated, reply exactly with: "Not found in context".
    
    CONTEXT:
    Prin.
    Romeo slew him, he slew Mercutio,
    Who now the price of his deare blood doth owe
    
    Cap.
    Not Romeo Prince, he was Mercutios Friend,
    His fault concludes, but what the law should end,
    The life of Tybalt
    
    Prin.
    And for that offence,
    Immediately we doe exile him hence:
    I haue an interest in your hearts proceeding:
    My bloud for your rude brawles doth lie a bleeding.
    But Ile Amerce you with so strong a fine,
    That you shall all repent the losse of mine.
    It will be deafe to pleading and excuses,
    Nor teares, nor prayers shall purchase our abuses.
    Therefore vse none, let Romeo hence in hast,
    Else when he is found, that houre is his last.
    Beare hence his body, and attend our will:
    Mercy not Murders, pardoning those that kill.
    
    Iul.
    Gallop apace, you fiery footed steedes,
    Towards Phoebus lodging, such a Wagoner
    As Phaeton would whip you to the west,
    And bring in Cloudie night immediately.
    Spred thy close Curtaine Loue-performing night,
    That run-awayes eyes may wincke, and Romeo
    Leape to these armes, vntalkt of and vnseene,
    Louers can see to doe their Amorous rights,
    And by their owne Beauties: or if Loue be blind,
    It best agrees with night: come ciuill night,
    Thou sober suted Matron all in blacke,
    And learne me how to loose a winning match,
    Plaid for a paire of stainlesse Maidenhoods,
    Hood my vnman'd blood bayting in my Cheekes,
    With thy Blacke mantle, till strange Loue grow bold,
    Thinke true Loue acted simple modestie:
    Come night, come Romeo, come thou day in night,
    For thou wilt lie vpon the wings of night
    Whiter then new Snow vpon a Rauens backe:
    Come gentle night, come louing blackebrow'd night.
    Giue me my Romeo, and when I shall die,
    Take him and cut him out in little starres,
    And he will make the Face of heauen so fine,
    That all the world will be in Loue with night,
    And pay no worship to the Garish Sun.
    O I haue bought the Mansion of a Loue,
    But not possest it, and though I am sold,
    Not yet enioy'd, so tedious is this day,
    As is the night before some Festiuall,
    To an impatient child that hath new robes
    And may not weare them, O here comes my Nurse:
    Enter Nurse with cords.
    Mer.
    Helpe me into some house Benuolio,
    Or I shall faint: a plague a both your houses.
    They haue made wormesmeat of me,
    I haue it, and soundly to your Houses.
    Enter.
    
    Rom.
    This Gentleman the Princes neere Alie,
    My very Friend hath got his mortall hurt
    In my behalfe, my reputation stain'd
    With Tibalts slaunder, Tybalt that an houre
    Hath beene my Cozin: O Sweet Iuliet,
    Thy Beauty hath made me Effeminate,
    And in my temper softned Valours steele.
    Enter Benuolio.
    
    Ben.
    O Romeo, Romeo, braue Mercutio's is dead,
    That Gallant spirit hath aspir'd the Cloudes,
    Which too vntimely here did scorne the earth
    
    Rom.
    This daies blacke Fate, on mo daies depend,
    This but begins, the wo others must end.
    Enter Tybalt.
    Mer.
    Come sir, your Passado
    
    Rom.
    Draw Benuolio, beat downe their weapons:
    Gentlemen, for shame forbeare this outrage,
    Tibalt, Mercutio, the Prince expresly hath
    Forbidden bandying in Verona streetes.
    Hold Tybalt, good Mercutio.
    
    Mer.
    I am hurt.
    A plague a both the Houses, I am sped:
    Is he gone and hath nothing?
    
    Ben.
    What art thou hurt?
    Cap.
    Wi. Tybalt, my Cozin? O my Brothers Child,
    O Prince, O Cozin, Husband, O the blood is spild
    Of my deare kinsman. Prince as thou art true,
    For bloud of ours, shed bloud of Mountague.
    O Cozin, Cozin
    
    Prin.
    Benuolio, who began this Fray?
    
    Ben.
    Tybalt here slaine, whom Romeo's hand did slay,
    Romeo that spoke him faire, bid him bethinke
    How nice the Quarrell was, and vrg'd withall
    Your high displeasure: all this vttered,
    With gentle breath, calme looke, knees humbly bow'd
    Could not take truce with the vnruly spleene
    Of Tybalts deafe to peace, but that he Tilts
    With Peircing steele at bold Mercutio's breast,
    Who all as hot, turnes deadly point to point,
    And with a Martiall scorne, with one hand beates
    Cold death aside, and with the other sends
    It back to Tybalt, whose dexterity
    Retorts it: Romeo he cries aloud,
    Hold Friends, Friends part, and swifter then his tongue,
    His aged arme, beats downe their fatall points,
    And twixt them rushes, vnderneath whose arme,
    An enuious thrust from Tybalt, hit the life
    Of stout Mercutio, and then Tybalt fled.
    But by and by comes backe to Romeo,
    Who had but newly entertained Reuenge,
    And too't they goe like lightning, for ere I
    Could draw to part them, was stout Tybalt slaine:
    And as he fell, did Romeo turne and flie:
    This is the truth, or let Benuolio die
    
    Cap.
    Wi. He is a kinsman to the Mountague,
    Affection makes him false, he speakes not true:
    Some twenty of them fought in this blacke strife,
    And all those twenty could but kill one life.
    I beg for Iustice, which thou Prince must giue:
    Romeo slew Tybalt, Romeo must not liue
    Ben.
    Here comes the Furious Tybalt backe againe
    
    Rom.
    He gon in triumph, and Mercutio slaine?
    Away to heauen respectiue Lenitie,
    And fire and Fury, be my conduct now.
    Now Tybalt take the Villaine backe againe
    That late thou gau'st me, for Mercutios soule
    Is but a little way aboue our heads,
    Staying for thine to keepe him companie:
    Either thou or I, or both, must goe with him
    
    Tib.
    Thou wretched Boy that didst consort him here,
    Shalt with him hence
    
    Rom.
    This shall determine that.
    
    QUESTION: Who killed Mercutio?
    ANSWER:



```python
# NOTE: We use max_new_tokens=3 here because words like "Tybalt" may be split into multiple tokens (e.g., "Ty", "b", "alt").
# It's often tricky to get exactly one word due to subword tokenization.
result = llm(prompt, max_new_tokens=5, do_sample=False)[0]["generated_text"]

# extract answer from full result, as before
generated_answer = result[len(prompt):].strip()
print(generated_answer)
```

    Tybalt


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


## See also
- [**Kaggle RAG Challenge**: WattBot](https://www.kaggle.com/competitions/WattBot2025/overview): Put your RAG methods to the test in this ML+X hosted Kaggle challenge (featured in [MLM25](https://ml-marathon.wisc.edu/)).
- [**Workshop**:Intro to Natural Language Processing (NLP)](https://uw-madison-datascience.github.io/ML-X-Nexus/Learn/Workshops/Intro-Deeplearning_PyTorch.html): Brush up on NLP basics before diving head-first into RAG pipelines.



```python

```
