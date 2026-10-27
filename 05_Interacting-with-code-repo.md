---
title: "(Optional) Pushing Your Notebooks to GitHub from SageMaker"
teaching: 25
exercises: 10
---

:::::::::::::::::::::::::::::::::::::: questions 

- How can I put the code I write on a SageMaker notebook instance under version control on GitHub?
- How do I authenticate to GitHub from a notebook instance without leaving a password or key behind?
- How can I convert notebooks to `.py` files and keep `.ipynb` files out of version control?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Create a GitHub repository for your own SageMaker work and connect a folder on the notebook instance to it.
- Configure Git on a notebook instance to push over HTTPS with a GitHub Personal Access Token (PAT), entered with `getpass`.
- Convert `.ipynb` files to `.py` files for cleaner version control.

::::::::::::::::::::::::::::::::::::::::::::::::

## Open prefilled .ipynb notebook
Open the notebook from: `/ML_with_AWS_SageMaker/notebooks/05_Interacting-with-code-repo.ipynb`. Select the **`conda_python3`** kernel.

This episode is **optional self-study**: it is not needed to run the rest of the lesson, but it is the workflow you will want once you start writing code of your own in SageMaker. It also doubles as a short git refresher. You need a GitHub account for it; nothing else in the lesson does.

## Step 0: What goes under version control
So far every notebook you have run came from the lesson repository, which you cloned read-only in the notebooks-as-controllers episode. You cannot push to that repository. This episode sets up the usual situation instead: **your own work in your own repository**. We will create an empty repository on GitHub, make a matching folder on the notebook instance, copy a notebook into it, and push.

1. On GitHub, click **New repository** (the **+** in the top-right corner). Name it `sagemaker-work` (any name is fine; the cells below use this one). Leave it **Public** or make it **Private**, as you prefer. Tick **Add a README file** so the repository is not empty, which gives Step 6 a pull with something in it. Click **Create repository**.
2. Back in JupyterLab, make the folder on the instance and turn it into a git repository. We put it under `/home/ec2-user/SageMaker/`, the one directory that survives stopping the instance.

```python
%cd /home/ec2-user/SageMaker/
!mkdir -p sagemaker-work
%cd sagemaker-work
!git init -b main   # start an empty git repository here, with "main" as its first branch
```

Connect it to the repository you just created, replacing `YOUR_GITHUB_USERNAME`:

```python
# A remote is a copy of the repository elsewhere; "origin" is the conventional name for the main one (here, GitHub)
!git remote add origin https://github.com/YOUR_GITHUB_USERNAME/sagemaker-work.git
!git remote -v   # list the remotes and their URLs, to check the address
```

## Step 1: A GitHub personal access token (PAT)
A notebook instance is a machine that other people may be able to reach (an administrator, or whoever uses it after you) and that you may delete next month. Neither a saved password nor an SSH key belongs on it. GitHub's answer is a **Personal Access Token**: a credential you create with only the permissions you need and an expiry date, use over HTTPS, and revoke in one click. We will enter it with `getpass` each session so that it never appears in a cell or a saved notebook.


#### Generate the token on GitHub
1. Go to **Settings** by clicking on your profile picture in the upper-right corner of GitHub.
2. Click **Developer settings** at the very bottom of the left sidebar.
3. Select **Personal access tokens**, then **Tokens (classic)**.
4. Click **Generate new token (classic)**.
5. Give it a descriptive name (e.g., "SageMaker notebook instance") and an expiry (30 days is plenty for a workshop).
6. **Select the minimum permissions needed**:
   - **For a public repository**: only **`public_repo`**.
   - **For a private repository**: **`repo`**.
7. Click **Generate token** and **copy it immediately**: you cannot see it again once you leave the page.

Treat the token like a password: store it in a password manager, not in a file on the instance, and revoke it (same page, **Delete**) when the workshop is over; the cleanup episode's checklist reminds you.

#### Enter it with `getpass`
`input()` and `getpass.getpass()` prompt you in the notebook; the token is held in a Python variable for this kernel session and is not written into the notebook file.

```python
import getpass   # prompts without echoing what you type, so the token never shows on screen or in the saved notebook

username = input("GitHub Username: ")
token = getpass.getpass("GitHub Personal Access Token (PAT): ")  # kept in this variable for the kernel session only
```

When you re-run the whole notebook later, this cell prompts again, because nothing is saved.

## Step 2: Configure Git settings
Git records an author on every commit. Set yours in this repository: without `--global` the setting goes into the repository's own `.git/config`, under `SageMaker/`, which survives a stop. (`--global` would write to your home directory, which is reset when the instance stops.)

```python
!git config user.name "Your name"             # appears as the author in the commit history
!git config user.email your_email@example.edu  # use the email on your GitHub account so commits link to your profile
```

## Step 3: Convert `.ipynb` to `.py`
We want our notebook under version control, but a `.ipynb` file is JSON with outputs and metadata baked in, which makes diffs unreadable and repositories large. The usual practice is to version a plain `.py` copy of the code and ignore the notebooks.

#### Benefits of converting to `.py` before committing
- **Cleaner version control**: readable diffs, reviewable pull requests, sane merges.
- **Script compatibility**: a `.py` file runs from the command line and imports as a module.
- **Smaller repository**: no outputs, no images, no cell metadata.

1. Install Jupytext, which converts in both directions.

```python
!pip install jupytext   # converts between .ipynb and .py in both directions
```

2. Copy the notebook from the Accessing S3 episode into our repository folder (we are still in `/home/ec2-user/SageMaker/sagemaker-work`) and convert it there. Copying keeps the lesson clone untouched and puts the `.py` file where git will see it.

```python
# Copy the notebook into the repository folder ("." is the current directory); the lesson clone stays untouched
!cp /home/ec2-user/SageMaker/ML_with_AWS_SageMaker/notebooks/04_Accessing-S3-via-SageMaker-notebooks.ipynb .
!jupytext --to py 04_Accessing-S3-via-SageMaker-notebooks.ipynb   # writes a .py next to it: code and markdown, no outputs
```

```text
[jupytext] Reading 04_Accessing-S3-via-SageMaker-notebooks.ipynb in format ipynb
[jupytext] Writing 04_Accessing-S3-via-SageMaker-notebooks.py
```

3. With several notebooks, `helpers.convert_files()` in the lesson's `scripts/helpers.py` converts them all. It converts every `.ipynb` in the *current directory*, so run it from inside the folder you mean to convert:

```python
import sys
# Put the lesson's scripts/ folder on Python's import path, as in the Accessing S3 episode
sys.path.insert(0, "/home/ec2-user/SageMaker/ML_with_AWS_SageMaker/scripts")
import helpers
helpers.convert_files(direction="notebook_to_python")
```

## Step 4: Add and commit the `.py` file

1. Check the status of the repository.

```python
%cd /home/ec2-user/SageMaker/sagemaker-work
!git status   # shows the untracked files (the .ipynb and the .py); nothing is staged yet
```

2. Add and commit. We add the `.py` by name rather than `git add .`, so the `.ipynb` copy is never tracked (Step 5 ignores it for good).

```python
!git add 04_Accessing-S3-via-SageMaker-notebooks.py   # stage: mark this file to go into the next commit
!git commit -m "Add Accessing-S3 notebook as a script"   # a commit is a snapshot of the staged files; the message says what changed, specifically
```

3. Check the status again.

```python
!git status   # the .py is committed; only the untracked .ipynb remains, which Step 5 ignores
```

## Step 5: Ignore notebooks and clutter with `.gitignore`
A `.gitignore` file lists patterns git should never track. Ignoring `*.ipynb` across the repository means a stray `git add .` cannot commit a notebook by accident.

1. Create the file (hidden in the file browser because it starts with a dot; `ls -a` shows it) and add the notebook pattern.

```python
with open(".gitignore", "a") as gitignore:   # "a" appends, so re-running the cell keeps earlier rules
    gitignore.write("# Jupyter notebooks: we version the .py copies instead\n*.ipynb\n")  # one pattern per line; * matches any name
!cat .gitignore
```

2. While we are here, ignore the usual clutter too: caches and temporary files. (The NumPy archives the training episodes write are the challenge at the end.)

```python
with open(".gitignore", "a") as gitignore:
    gitignore.write("\n# Cache and temp files\n__pycache__/\n*.tmp\n*.log\n")
!cat .gitignore
```

3. Add and commit `.gitignore` so the rule travels with the repository.

```python
!git add .gitignore   # .gitignore is itself tracked, so everyone who clones gets the same rules
!git commit -m "Ignore notebooks, caches and temp files"
```

## Step 6: Pull, then push
Before pushing, pull: the remote already has the README that GitHub created, and in a shared repository it may have your collaborators' commits. Pulling first means your push is a simple addition rather than a rejected one.

#### Which pull strategy
Git has to decide how to combine your local commits with remote ones. The three options:

* **Merge** (`pull.rebase false`): the remote changes come in as a merge commit. Simplest, shows where you integrated. The default and the right choice for most collaborative work.
* **Rebase** (`pull.rebase true`): your commits are replayed on top of the remote ones, giving a linear history. Cleaner, but conflicts have to be resolved commit by commit.
* **Fast-forward only** (`pull.ff only`): refuses to pull unless nothing diverged. Useful as a safety check, not as a daily setting.

We use merge. Because our folder and the GitHub repository started separately (one from `git init`, one from **Create repository**), git also needs `--allow-unrelated-histories` for this very first pull; never again after that.

```python
!git config pull.rebase false   # pull strategy: merge (remote commits come in through a merge commit)
# pull = fetch origin's main and merge it into ours; the flag is needed only this once, as the two histories share no commit
!git pull https://{username}:{token}@github.com/{username}/sagemaker-work.git main --allow-unrelated-histories --no-edit   # the token authenticates the pull too, which a private repository needs
```

If git reports a merge conflict, each conflicting file has to be resolved one way or the other and then staged and committed. You will not get one here (the README and your script are different files), but this is what it looks like:

```python
# Example only.
# Keep YOUR version of a file:
# !git checkout --ours 04_Accessing-S3-via-SageMaker-notebooks.py
# Keep the REMOTE version of a file:
# !git checkout --theirs README.md
# Then stage and commit the resolution:
# !git add 04_Accessing-S3-via-SageMaker-notebooks.py README.md
# !git commit -m "Resolve merge conflicts"
```

#### Push with the PAT
The token goes into the URL for this one command, built from the variables `getpass` filled. It is not stored by git and does not appear in the notebook.

```python
github_url = f"github.com/{username}/sagemaker-work.git"
# IPython substitutes the Python variables in {braces}; user:token@ in an HTTPS URL authenticates this one push and is not stored
!git push https://{username}:{token}@{github_url} main
```

Open `https://github.com/YOUR_GITHUB_USERNAME/sagemaker-work` in a browser: the README, the `.py` file and `.gitignore` should all be there, with four commits in the history: GitHub's initial commit, your two, and the merge the pull made.

## Step 7: Pulling `.py` files and converting back to notebooks
Suppose a collaborator edits the `.py` file on GitHub, or you edit it from another machine. Pull as usual:

```python
!git pull https://{username}:{token}@github.com/{username}/sagemaker-work.git main   # a routine pull: no extra flag now that the histories are joined
```

Then convert the script back to a notebook with Jupytext. The `.ipynb` it writes is ignored by git (Step 5), so the round trip leaves the repository clean.

```python
# --to notebook converts the other way; the .ipynb it writes matches the *.ipynb rule in .gitignore
!jupytext --to notebook 04_Accessing-S3-via-SageMaker-notebooks.py --output 04_Accessing-S3-via-SageMaker-notebooks.ipynb
```

To convert every `.py` in the folder back to notebooks, `helpers.convert_files(direction="python_to_notebook")` does the loop; it converts *every* `.py` in the current directory, so keep real modules out of that folder.

::::::::::::::::::::::::::::::::::::: challenge

### Challenge: ignore the big files too

The PyTorch episode saves NumPy arrays (`.npz`) in its working folder, and your own notebooks will do the same in this one. Those are binary and can be large, which makes them a poor fit for git. Add `*.npz` to your repository's `.gitignore`, commit it, push with your PAT, and verify the change on GitHub.

:::::::::::::::::::::::::::::::::::: solution

Still inside `/home/ec2-user/SageMaker/sagemaker-work`:

```python
with open(".gitignore", "a") as gitignore:
    gitignore.write("\n# NumPy archives saved by the training episodes\n*.npz\n")
!cat .gitignore
!git add .gitignore
!git commit -m "Ignore .npz arrays"
!git push https://{username}:{token}@github.com/{username}/sagemaker-work.git main
```

Then open `https://github.com/YOUR_GITHUB_USERNAME/sagemaker-work/blob/main/.gitignore` in a browser: the `*.npz` line should be there, and the commit list should show "Ignore .npz arrays" as the latest commit. If the push is rejected with "authentication failed", the PAT has expired or lacks the `public_repo` (or `repo`) scope; generate a new one and re-run the `getpass` cell.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::


:::::::::::::::::::::::::::::::::::::: keypoints 

- Your own work belongs in your own repository: create one on GitHub, `git init` a folder under `/home/ec2-user/SageMaker/`, and connect the two with `git remote add`.
- Authenticate over HTTPS with a scoped, expiring Personal Access Token entered through `getpass`; nothing is saved on the instance, and the token is revoked when you are done.
- Version `.py` copies of notebooks (Jupytext converts both ways) and ignore `*.ipynb`, caches and data archives in `.gitignore`.
- Pull before you push; the first pull into a freshly initialised folder needs `--allow-unrelated-histories`.

::::::::::::::::::::::::::::::::::::::::::::::::
