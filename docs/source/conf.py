# Configuration file for the Sphinx documentation builder.
import os
import sys
# Need this so sphinx can find lumache.py. Change is .py files are elsewhere than root.
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../bird'))

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, '..', '..',  "bird", "version.py"), encoding="utf-8") as f:
    version = f.read()
version = version.split("=")[-1].strip().strip('"').strip("'")

# -- Project information

project = "Bio Reactor Design (BiRD)"
copyright = '2024'
author = 'NREL'

release = version
version = version

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
#    'autoapi.extension',
#    'sphinxcontrib.apidoc',
]

#autoapi_type = 'python'
#autoapi_dirs = ['../../bird']

#apidoc_module_dir = '../../src'
#apidoc_output_dir = '.'
#apidoc_excluded_paths = ['tests']
#apidoc_separate_modules = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

html_context = {
    "display_github": True, # Integrate GitHub
    "github_repo": "NREL/BioReactorDesign", # Repo name
    "github_version": "main", # Version
    "conf_py_path": "docs/source/", # Path in the checkout to the docs root
}

# -- Options for HTML output -------------------------------------------------

html_short_title = "bird-doc"
html_show_sourcelink = False
html_show_sphinx = True
html_show_copyright = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
repository_url = f"https://github.com/NREL/BioReactorDesign"
html_context = {
    "menu_links": [
        (
            '<i class="fa fa-github fa-fw"></i> Source Code',
            repository_url,
        ),
        (
            '<i class="fa fa-book fa-fw"></i> License',
            f"{repository_url}/blob/main/LICENSE",
        ),
    ],
}
