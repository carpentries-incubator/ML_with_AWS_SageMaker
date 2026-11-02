import hashlib
import os
import re

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

# Paths
episodes_dir = "episodes"
notebooks_dir = "notebooks"

# List of Markdown files to ignore (no conversion needed)
ignore_list = [
    "01_SageMaker-overview.md",
    "02_Data-storage-setting-up-S3.md",
    "03_SageMaker-notebooks-as-controllers.md",
    "14_CLI-equivalents.md"
]

# Ensure notebooks directory exists
os.makedirs(notebooks_dir, exist_ok=True)

# Regular expression to detect code blocks (matches ```language\n...\n```)
code_block_pattern = re.compile(r"```(\w+)?\n(.*?)\n```", re.DOTALL)


def stable_cell_id(source, seen_ids):
    """Return a cell id derived from the cell's content.

    nbformat assigns a fresh random id to every new cell, which makes each
    run of this script rewrite every notebook even when nothing changed.
    Hashing the source instead gives the same id on every run; a counter is
    appended when two cells in the same notebook have identical content,
    because nbformat requires ids to be unique within a notebook.
    """
    base = hashlib.sha1(source.encode("utf-8")).hexdigest()[:8]
    cell_id = base
    n = 1
    while cell_id in seen_ids:
        n += 1
        cell_id = f"{base}-{n}"
    seen_ids.add(cell_id)
    return cell_id


# Figures live in episodes/fig/ and the episodes reference them as fig/<name>.
# The generated notebooks sit in notebooks/, where that relative path does not
# exist, so point them at the published lesson site instead, which serves the
# same files with the right content types. The Workbench {alt="..."} attribute
# block is dropped because Jupyter would print it literally.
SITE_FIG_URL = "https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/fig/"
figure_pattern = re.compile(r"!\[([^\]]*)\]\(fig/([^)\s]+)\)(\{[^}]*\})?")


def rewrite_figure_links(md_content):
    return figure_pattern.sub(lambda m: f"![{m.group(1)}]({SITE_FIG_URL}{m.group(2)})", md_content)


def split_markdown(md_content):
    """Splits Markdown content into separate Markdown and Code cells."""
    cells = []
    seen_ids = set()
    position = 0
    md_content = rewrite_figure_links(md_content)

    for match in code_block_pattern.finditer(md_content):
        # Extract text before the code block as Markdown
        before_code = md_content[position:match.start()].strip()
        if before_code:
            cells.append(new_markdown_cell(before_code, id=stable_cell_id(before_code, seen_ids)))

        # Only ```python fences become code cells. Any other fence (bash, json,
        # text, yaml, an untagged block) is explanatory and stays in Markdown,
        # so the generated notebook never contains a cell that cannot run.
        lang = (match.group(1) or "").lower()
        code_content = match.group(2).strip()
        if lang == "python":
            if code_content:
                cells.append(new_code_cell(code_content, id=stable_cell_id(code_content, seen_ids)))
        else:
            block_md = match.group(0).strip()
            cells.append(new_markdown_cell(block_md, id=stable_cell_id(block_md, seen_ids)))

        position = match.end()

    # Add any remaining Markdown content after the last code block
    remaining_md = md_content[position:].strip()
    if remaining_md:
        cells.append(new_markdown_cell(remaining_md, id=stable_cell_id(remaining_md, seen_ids)))

    return cells


# Convert each Markdown file in episodes/
for filename in sorted(os.listdir(episodes_dir)):
    if filename.endswith(".md") and filename not in ignore_list:
        md_path = os.path.join(episodes_dir, filename)
        ipynb_path = os.path.join(notebooks_dir, filename.replace(".md", ".ipynb"))

        # Read Markdown content
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        # Split into Markdown and Code cells
        notebook_cells = split_markdown(md_content)

        # Create Jupyter notebook
        nb = new_notebook(cells=notebook_cells)

        # Save as .ipynb
        with open(ipynb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

print("Conversion complete! Excluded:", ", ".join(ignore_list))
