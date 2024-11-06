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

# Using a GitHub Personal Access Token (PAT) to Push/Pull from a SageMaker Notebook
When working in SageMaker notebooks, you may often need to push code updates to GitHub repositories. However, SageMaker notebooks are typically launched with temporary instances that don’t persist configurations, including SSH keys, across sessions. This makes HTTPS-based authentication, secured with a GitHub Personal Access Token (PAT), a practical solution. PATs provide flexibility for authentication and enable seamless interaction with both public and private repositories directly from your notebook. 

> **Important Note**: Personal access tokens are powerful credentials that grant specific permissions to your GitHub account. To ensure security, only select the minimum necessary permissions and handle the token carefully.

In this episode, we'll see how to push our code to the fork we created during the [workshop setup](https://uw-madison-datascience.github.io/ML_with_Amazon_SageMaker/#workshop-repository-setup).

## Step 1: Generate a Personal Access Token (PAT) on GitHub

1. Go to **Settings > Developer settings > Personal access tokens** on GitHub.
2. Click **Generate new token**, select **Classic**.
3. Give your token a descriptive name (e.g., "SageMaker Access Token") and set an expiration date if desired for added security.
4. **Select the minimum permissions needed**:
   - **For public repositories**: Choose only **`public_repo`**.
   - **For private repositories**: Choose **`repo`** (full control of private repositories).
   - Optional permissions, if needed:
     - **`repo:status`**: Access commit status (if checking status checks).
     - **`workflow`**: Update GitHub Actions workflows (only if working with GitHub Actions).
5. Generate the token and **copy it** (you won’t be able to see it again).

> **Caution**: Treat your PAT like a password. Avoid sharing it or exposing it in your code. Store it securely (e.g., via a password manager like LastPass) and consider rotating it regularly.


## Step 2: Configure Git `user.name` and `user.email`
In your SageMaker or Jupyter notebook environment, run the following commands to set up your Git user information


#### Directory setup
Let's make sure we're starting at the same directory. Cd to the root directory of this instance before going further.


```python
%cd /home/ec2-user/SageMaker/
```

    /home/ec2-user/SageMaker

```python

!git config --global user.name "Your name"
!git config --global user.email your_email@wisc.edu

```

### Explanation

- **`user.name`**: This is your GitHub username, which will appear in the commit history as the author of the changes.
- **`user.email`**: This should match the email associated with your GitHub account so that commits are properly linked to your profile.

Setting this globally (`--global`) will ensure the configuration persists across all repositories in the environment. If you’re working in a temporary environment, you may need to re-run this configuration after a restart.

## Step 3: Use `getpass` to Prompt for Username and PAT

The `getpass` library allows you to input your GitHub username and PAT without exposing them in the notebook. This approach ensures you’re not hardcoding sensitive information.

```python
import getpass

# Prompt for GitHub username and PAT securely
username = input("GitHub Username: ")
token = getpass.getpass("GitHub Personal Access Token (PAT): ")
```

**Note**: After running, you may want to comment out the above code so that you don't have to enter in your login every time you run your whole notebook


### Explanation

- **`input("GitHub Username: ")`**: Prompts you to enter your GitHub username.
- **`getpass.getpass("GitHub Personal Access Token (PAT): ")`**: Prompts you to securely enter the PAT, keeping it hidden on the screen.



## Step 4: Add, Commit, and Push Changes with Manual Authentication
### 1. Navigate to the Repository Directory (adjust the path if needed):



```python
!pwd
%cd test_AWS
```

    /home/ec2-user/SageMaker
    /home/ec2-user/SageMaker/test_AWS


### 2. Preview changes: You may see elaborate changes if you are tracking ipynb files directly.


```python
!git diff 
```

### 3. Convert json ipynb files to .py

To avoid tracking ipynb files directly, which are formatted as json, we may want to convert our notebook to .py first (plain text). This will make it easier to see our code edits across commits. Otherwise, each small edit will have massive changes associated with it.

#### Benefits of converting to `.py` before Committing

- **Cleaner Version Control**: `.py` files have cleaner diffs and are easier to review and merge in Git.
- **Script Compatibility**: Python files are more compatible with other environments and can run easily from the command line.
- **Reduced Repository Size**: `.py` files are generally lighter than `.ipynb` files since they don’t store outputs or metadata.

Converting notebooks to `.py` files helps streamline the workflow for both collaborative projects and deployments. This approach also maintains code readability and minimizes potential issues with notebook-specific metadata in Git history. Here’s how to convert `.ipynb` files to `.py` in SageMaker without needing to export or download files:

#### Method 1: Using JupyText

1. **Install Jupytext** (if you haven’t already):


```python
!pip install jupytext

```


1. **Run the following command** in a notebook cell to convert the current notebook to a `.py` file:

This command will create a `.py` file in the same directory as the notebook.


```python
# Replace 'your_notebook.ipynb' with your actual notebook filename
!jupytext --to py Data-storage-and-access-via-buckets.ipynb
```

    [jupytext] Reading 03_Data-storage-and-access-via-buckets.ipynb in format ipynb
    [jupytext] Updating the timestamp of 03_Data-storage-and-access-via-buckets.py


#### Method 2: Automated Script for Converting All Notebooks in a Directory

If you have multiple notebooks to convert, you can automate the conversion process by running this script, which converts all `.ipynb` files in the current directory to `.py` files:


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

### 4. Adding .ipynb to gitigore

Adding `.ipynb` files to `.gitignore` is a good practice if you plan to only commit `.py` scripts. This will prevent accidental commits of Jupyter Notebook files across all subfolders in the repository.

Here’s how to add `.ipynb` files to `.gitignore` to ignore them project-wide:

1. **Open or Create the `.gitignore` File**:

    ```python
    !ls -a # check for existing .gitignore file
    ```
    
   - If you don’t already have a `.gitignore` file in the repository root (use '!ls -a' to check, you can create one by running:
   
     ```python
     !touch .gitignore
     ```


2. **Add `.ipynb` Files to `.gitignore`**:

   - Append the following line to your `.gitignore` file to ignore all `.ipynb` files in all folders:

     ```plaintext
     *.ipynb # Ignore all Jupyter Notebook files
     ```

   - You can add this line using a command within your notebook:
   
     ```python
     with open(".gitignore", "a") as gitignore:
         gitignore.write("\n# Ignore all Jupyter Notebook files\n*.ipynb\n")
     ```



3. **Verify and Commit the `.gitignore` File**:

   - Add and commit the updated `.gitignore` file to ensure it’s applied across the repository.

     ```python
     !git add .gitignore
     !git commit -m "Add .ipynb files to .gitignore to ignore notebooks"
     !git push origin main
     ```

This setup will:
- Prevent all `.ipynb` files from being tracked by Git.
- Keep your repository cleaner, containing only `.py` scripts for easier version control and reduced repository size. 

Now any new or existing notebooks won’t show up as untracked files in Git, ensuring your commits stay focused on the converted `.py` files.


2. **Add and Commit Changes**:




```python
!git add . # you may also add files one at a time, for further specificity over the associated commit message
!git commit -m "Updates from Jupyter notebooks" # in general, your commit message should be more specific!

```

3. **Pull the Latest Changes from the Main Branch**: Pull the latest changes from the remote main branch to ensure your local branch is up-to-date.

    Recommended: Set the Pull Strategy for this Repository (Merge by Default)

    All options:

    * Merge (pull.rebase false): Combines the remote changes into your local branch as a merge commit.
    * Rebase (pull.rebase true): Replays your local changes on top of the updated main branch, resulting in a linear history.
    * Fast-forward only (pull.ff only): Only pulls if the local branch can fast-forward to the remote without diverging (no new commits locally).


```python
!git config pull.rebase false # Combines the remote changes into your local branch as a merge commit.

!git pull origin main

```

If you get merge conflicts, be sure to resolve those before moving forward (e.g., use git checkout -> add -> commit). You can skip the below code if you don't have any conflicts. 


```python
# Keep your local changes in one conflicting file
# !git checkout --ours train_nn.py

# Keep remote version for the other conflicting file
# !git checkout --theirs train_xgboost.py

# # Stage the files to mark the conflicts as resolved
# !git add train_nn.py
# !git add train_xgboost.py

# # Commit the merge result
# !git commit -m "Resolved merge conflicts by keeping local changes"
```

4. **Push Changes and Enter Credentials**:


```python
# Push with embedded credentials from getpass (avoids interactive prompt)
github_url = 'github.com/username/ML_with_Amazon_SageMaker.git' # replace username with your own. THe full address for your fork can be found under Code -> Clone -> HTTPS (remote the https:// before the rest of the address)
!git push https://{username}:{token}@{github_url} main
```

    fatal: unable to access 'https://{github_url}/': URL rejected: Bad hostname


## Step 5: Pulling .py files and converting back to notebook format

Let's assume you've taken a short break from your work, and you would like to start again by pulling in your code repo. If you'd like to work with notebook files again, you can again use jupytext to convert your `.py` files back to `.ipynb`

This command will create `03_Data-storage-and-access-via-buckets-test.ipynb` in the current directory, converting the Python script to a Jupyter Notebook format. Jupytext handles the conversion gracefully without expecting the `.py` file to be in JSON format.


```python
# Replace 'your_script.py' with your actual filename
!jupytext --to notebook Data-storage-and-access-via-buckets.py --output Data-storage-and-access-via-buckets-test.ipynb

```

### Applying to all .py files
To convert all of your .py files to notebooks, you can use the following code:


```python
import subprocess
import os

# List all .py files in the directory
scripts = [f for f in os.listdir() if f.endswith('.py')]

# Convert each .py file to .ipynb using jupytext
for script in scripts:
    output_file = script.replace('.py', '.ipynb')
    subprocess.run(["jupytext", "--to", "notebook", script, "--output", output_file])
    print(f"Converted {script} to {output_file}")

```

:::::::::::::::::::::::::::::::::::::: keypoints 

- Use a GitHub PAT for HTTPS-based authentication in temporary SageMaker notebook instances.
- Securely enter sensitive information in notebooks using `getpass`.
- Converting `.ipynb` files to `.py` files helps with cleaner version control and easier review of changes.
- Adding `.ipynb` files to `.gitignore` keeps your repository organized and reduces storage.

::::::::::::::::::::::::::::::::::::::::::::::::
