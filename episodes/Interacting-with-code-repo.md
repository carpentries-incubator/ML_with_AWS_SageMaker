---
title: "Using a GitHub Personal Access Token (PAT) to Push/Pull from a SageMaker Notebook"
teaching: 25
exercises: 10
---

:::::::::::::::::::::::::::::::::::::: questions 

- How can I securely push/pull code to and from GitHub within a SageMaker notebook?
- What steps are necessary to set up a GitHub PAT for authentication in SageMaker?
- How can I convert notebooks to `.py` files and ignore `.ipynb` files in version control?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Configure Git in a SageMaker notebook to use a GitHub Personal Access Token (PAT) for HTTPS-based authentication.
- Securely handle credentials in a notebook environment using `getpass`.
- Convert `.ipynb` files to `.py` files for better version control practices in collaborative projects.

::::::::::::::::::::::::::::::::::::::::::::::::

## Open prefilled .ipynb notebook
Open the notebook from: `/ML_with_AWS_SageMaker/notebooks/Interacting-with-code-repo.ipynb.ipynb`.

## Step 0: Initial setup
In this episode, we'll demonstrate how to push code to GitHub from a SageMaker Jupyter Notebook.

To begin, we will clone the fork we had you creat during the [workshop setup](https://uw-madison-datascience.github.io/ML_with_Amazon_SageMaker/#workshop-repository-setup).

Let's make sure we're starting at the same directory. Cd to the root directory of this instance before going further.

```python
%cd /home/ec2-user/SageMaker/
```

Then, clone the fork. Replace "USERNAME" below with your GitHub username.

```python
!git clone https://github.com/USERNAME/AWS_helpers.git # replace username with your GitHub username
```
    
## Step 1: Using a GitHub personal access token (PAT) to push/pull from a SageMaker notebook
When working in SageMaker notebooks, you may often need to push code updates to GitHub repositories. However, SageMaker notebooks are typically launched with temporary instances that don't persist configurations, including SSH keys, across sessions. This makes HTTPS-based authentication, secured with a GitHub Personal Access Token (PAT), a practical solution. PATs provide flexibility for authentication and enable seamless interaction with both public and private repositories directly from your notebook. 

> **Important Note**: Personal access tokens are powerful credentials that grant specific permissions to your GitHub account. To ensure security, only select the minimum necessary permissions and handle the token carefully.

#### Generate a personal access token (PAT) on GitHub
1. Go to **Settings** by clicking on your profile picture in the upper-right corner of GitHub.
2. Click **Developer settings** at the very bottom of the left sidebar.
3. Select **Personal access tokens**, then click **Tokens (classic)**.
4. Click **Generate new token (classic)**.
5. Give your token a descriptive name (e.g., "SageMaker Access Token") and set an expiration date if desired for added security.
6. **Select the minimum permissions needed**:
   - **For public repositories**: Choose only **`public_repo`**.
   - **For private repositories**: Choose **`repo`** (full control of private repositories).
   - Optional permissions, if needed:
     - **`repo:status`**: Access commit status (if checking status checks).
     - **`workflow`**: Update GitHub Actions workflows (only if working with GitHub Actions).
7. Click **Generate token** and **copy it immediately**—you won't be able to see it again once you leave the page.


> **Caution**: Treat your PAT like a password. Avoid sharing it or exposing it in your code. Store it securely (e.g., via a password manager like LastPass) and consider rotating it regularly.

#### Use `getpass` to prompt for username and PAT
The `getpass` library allows you to input your GitHub username and PAT without exposing them in the notebook. This approach ensures you're not hardcoding sensitive information.

```python
import getpass

# Prompt for GitHub username and PAT securely
username = input("GitHub Username: ")
token = getpass.getpass("GitHub Personal Access Token (PAT): ")
```

**Note**: After running, you may want to comment out the above code so that you don't have to enter in your login every time you run your whole notebook

## Step 2: Configure Git settings
In your SageMaker or Jupyter notebook environment, run the following commands to set up your Git user information.

Setting this globally (`--global`) will ensure the configuration persists across all repositories in the environment. If you're working in a temporary environment, you may need to re-run this configuration after a restart.

```python
!git config --global user.name "Your name" # This is your GitHub username (or just your name), which will appear in the commit history as the author of the changes.
!git config --global user.email your_email@wisc.edu # This should match the email associated with your GitHub account so that commits are properly linked to your profile.
```

## Step 3: Convert json .ipynb files to .py
We'd like to track our notebook files within our AWS_helpers fork. However, to avoid tracking ipynb files directly, which are formatted as json, we may want to convert our notebook to .py first (plain text). Converting notebooks to `.py` files helps maintain code (and version-control) readability and minimizes potential issues with notebook-specific metadata in Git history. 

#### Benefits of converting to `.py` before Committing
- **Cleaner version control**: `.py` files have cleaner diffs and are easier to review and merge in Git.
- **Script compatibility**: Python files are more compatible with other environments and can run easily from the command line.
- **Reduced repository size**: `.py` files are generally lighter than `.ipynb` files since they don't store outputs or metadata.

Here's how to convert `.ipynb` files to `.py` in SageMaker without needing to export or download files.

1. First, install Jupytext.
```python
!pip install jupytext
```

2. Then, run the following command in a notebook cell to convert both of our notebooks to `.py` files
```python
# Adjust filename(s) if you used something different
!jupytext --to py Interacting-with-S3.ipynb
```
    [jupytext] Reading Interacting-with-S3.ipynb in format ipynb
    [jupytext] Writing Interacting-with-S3.py

3. If you have multiple notebooks to convert, you can automate the conversion process by running this code, which converts all `.ipynb` files in the current directory to `.py` files:

```python
import subprocess
import os

# List all .ipynb files in the directory
notebooks = [f for f in os.listdir() if f.endswith('.ipynb')]

# Convert each notebook to .py using jupytext
for notebook in notebooks:
    output_file = notebook.replace('.ipynb', '.py')
    subprocess.run(["jupytext", "--to", "py", notebook, "--output", output_file])
    print(f"Converted {notebook} to {output_file}")

```

For convenience, we have placed this code inside a `convert_files()` function in `helpers.py`.

```python
import AWS_helpers.helpers as helpers
helpers.convert_files(direction="notebook_to_python")
```

**Once converted, move our new .py file to the AWS_helpers folder using the file explorer panel in Jupyter Lab.**

## Step 4. Add and commit .py files

1. Check status of repo. Make sure you're in the repo folder before running the next step.
```python
%cd /home/ec2-user/SageMaker/AWS_helpers
```

```python
!git status
```

2. Add and commit changes

```python
!git add . # you may also add files one at a time, for further specificity over the associated commit message
!git commit -m "Updates from Jupyter notebooks" # in general, your commit message should be more specific!

```

3. Check status

```python
!git status
```

## Step 5. Adding .ipynb to gitigore

Adding `.ipynb` files to `.gitignore` is a good practice if you plan to only commit `.py` scripts. This will prevent accidental commits of Jupyter Notebook files across all subfolders in the repository.

Here's how to add `.ipynb` files to `.gitignore` to ignore them project-wide:

1. **Check working directory**: First make sure we're in the repo folder
    
```python
!pwd
#%cd AWS_helpers
```

2. **Create the `.gitignore` file**: This file will be hidden in Jupyter (since it starts with "."), but you can verify it exists using `ls`.
```python
!touch .gitignore
!ls -a
```

3. **Add `.ipynb` files to `.gitignore`**: You can add this line using a command within your notebook:

```python
with open(".gitignore", "a") as gitignore:
	gitignore.write("\n# Ignore all Jupyter Notebook files\n*.ipynb\n")
```

View file contents
```python
!cat .gitignore
```

4. **Ignore other common temp files**
While we're at it, let's ignore other common files that can clutter repos, such as cache folders and temporary files.

```python
with open(".gitignore", "a") as gitignore:
	gitignore.write("\n# Ignore cache and temp files\n__pycache__/\n*.tmp\n*.log\n")
```

View file contents

```python
!cat .gitignore
```
    
5. **Add and commit the `.gitignore` file**:

Add and commit the updated `.gitignore` file to ensure it's applied across the repository.

```python
!git add .gitignore
!git commit -m "Add .ipynb files to .gitignore to ignore notebooks"
```

This setup will:

- Prevent all `.ipynb` files from being tracked by Git.
- Keep your repository cleaner, containing only `.py` scripts for easier version control and reduced repository size. 

## Step 6. Merging local changes with remote/GitHub
Our local changes have now been committed, and we can begin the process of mergining with the remote main branch. Before we try to push our changes, it's good practice to first to a pull. This is critical when working on a collaborate repo with multiple users, so that you don't miss any updates from other team members.


### 1. Pull the latest changes from the main branch
There are a few different options for pulling the remote code into your local version. The best pull strategy depends on your workflow and the history structure you want to maintain. Here's a breakdown to help you decide:

* Merge (pull.rebase false): Combines the remote changes into your local branch as a merge commit.
   - **Use if**: You're okay with having merge commits in your history, which indicate where you pulled in remote changes. This is the default and is usually the easiest for team collaborations, especially if conflicts arise.

* Rebase (pull.rebase true): Replays your local changes on top of the updated main branch, resulting in a linear history.
    - **Use if**: You prefer a clean, linear history without merge commits. Rebase is useful if you like to keep your branch history as if all changes happened sequentially.

* Fast-forward only (pull.ff only): Only pulls if the local branch can fast-forward to the remote without diverging (no new commits locally).
    - **Use if**: You only want to pull updates if no additional commits have been made locally. This can be helpful to avoid unintended merges when your branch hasn't diverged.

#### Recommended for Most Users
If you're collaborating and want simplicity, **merge (pull.rebase false)** is often the most practical option. This will ensure you get remote changes with a merge commit that captures the history of integration points. For those who prefer a more streamlined history and are comfortable with Git, **rebase (pull.rebase true)** can be ideal but may require more careful conflict handling.

```python
!git config pull.rebase false # Combines the remote changes into your local branch as a merge commit.
!git pull origin main
```

If you get merge conflicts, be sure to resolve those before moving forward (e.g., use git checkout -> add -> commit). You can skip the below code if you don't have any conflicts. 

```python
# Keep your local changes in one conflicting file
# !git checkout --ours Interacting-with-git.py

# Keep remote version for the other conflicting file
# !git checkout --theirs Interacting-with-git.py

# # Stage the files to mark the conflicts as resolved
# !git add Interacting-with-git.py

# # Commit the merge result
# !git commit -m "Resolved merge conflicts by keeping local changes"
```

### 2. Push changes using PAT creditials 

```python
# Push with embedded credentials from getpass (avoids interactive prompt)
github_url = f'github.com/{username}/AWS_helpers.git' # The full address for your fork can be found under Code -> Clone -> HTTPS (remote the https:// before the rest of the address)
!git push https://{username}:{token}@{github_url} main
```
   
After pushing, you can navigate back to your fork on GitHub to verify everything worked (e.g., https://github.com/username/AWS_helpers/tree/main)

## Step 7: Pulling .py files and converting back to notebook format

Let's assume you've taken a short break from your work, and others on your team have made updates to your .py files on the remote main branch. If you'd like to work with notebook files again, you can again use jupytext to convert your `.py` files back to `.ipynb`. 

1. First, pull any updates from the remote main branch.

```python
!git config pull.rebase false # Combines the remote changes into your local branch as a merge commit.
!git pull origin main
```

2. We can then use jupytext again to convert in the other direction (.py to .ipynb).
This command will create `Interacting-with-S3.ipynb` in the current directory, converting the Python script to a Jupyter Notebook format. Jupytext handles the conversion gracefully without expecting the `.py` file to be in JSON format.

```python
!jupytext --to notebook Interacting-with-S3.py --output Interacting-with-S3.ipynb
```

### Applying to all .py files
To convert all of your .py files to notebooks, you can use our helper function as follows

```python
helpers.convert_files(direction="python_to_notebook")
```


:::::::::::::::::::::::::::::::::::::: keypoints 

- Use a GitHub PAT for HTTPS-based authentication in temporary SageMaker notebook instances.
- Securely enter sensitive information in notebooks using `getpass`.
- Converting `.ipynb` files to `.py` files helps with cleaner version control and easier review of changes.
- Adding `.ipynb` files to `.gitignore` keeps your repository organized and reduces storage.

::::::::::::::::::::::::::::::::::::::::::::::::
