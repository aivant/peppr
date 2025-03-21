import sys
from pathlib import Path
import peppr

DOC_PATH = Path(__file__).parent

# Include documentation in PYTHONPATH
# in order to import modules for API doc generation etc.
sys.path.insert(0, str(DOC_PATH))
import viewcode  # noqa: E402

#### Source code link ###

# linkcode_resolve = viewcode.linkcode_resolve

#### General ####

extensions = [
    "jupyter_sphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx.ext.linkcode",
    "sphinx.ext.todo",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "sphinx_copybutton",
    "numpydoc",
]

numpydoc_show_class_members = False
linkcode_resolve = viewcode.linkcode_resolve

intersphinx_mapping = {
    "biotite": ("https://www.biotite-python.org/latest/", None),
    "rdkit": ("https://www.rdkit.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}
intersphinx_timeout = 60

templates_path = ["templates"]
master_doc = "index"

project = "pepp'r"
copyright = """2025, The pepp'r contributors"""
author = """The pepp'r contributors"""
version = peppr.__version__
release = peppr.__version__

pygments_style = "sphinx"

#### HTML ####

html_theme = "pydata_sphinx_theme"

html_static_path = ["static"]
html_css_files = [
    "peppr.css",
    # Get fonts from Google Fonts CDN
    "https://fonts.googleapis.com/css2"
    "?family=Geologica:wght@100..900"
    "&family=Chelsea+Market"
    "&display=swap",
]
html_title = "pepp'r"
html_logo = "static/assets/general/logo.svg"
html_favicon = "static/assets/general/icon.png"
# html_baseurl = "https://peppr.github.io"
html_theme_options = {
    "header_links_before_dropdown": 7,
    "pygment_light_style": "friendly",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/aivant/peppr",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/peppr/",
            "icon": "fa-solid fa-box-open",
            "type": "fontawesome",
        },
    ],
    "use_edit_page_button": True,
    "show_prev_next": False,
    "show_toc_level": 2,
}
html_sidebars = {
    # No primary sidebar for these pages
    "tutorial/index": [],
    "contribution": [],
}
html_context = {
    "github_user": "aivant",
    "github_repo": "peppr",
    "github_version": "main",
    "doc_path": "docs",
}
html_scaled_image_link = False
