# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'SPICE'
copyright = '2024-2026, Maja Jablonska'
author = 'Maja Jablonska'

# Pull the version from the installed package metadata so the docs don't drift
# from pyproject.toml. Fall back to a sensible default if the package can't be
# imported during the build.
try:
    from importlib.metadata import version as _pkg_version
    release = _pkg_version('stellar-spice')
except Exception:
    release = '0.0.0'
version = release

html_logo = "../img/spice_pink.svg"
html_favicon = '../img/spice_pink-cropped.png'

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#ff7ce4",
        "color-brand-content": "#ff7ce4",
    },
    "dark_css_variables": {
        "color-brand-primary": "#ff7ce4",
        "color-brand-content": "#ff7ce4",
    }
}

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx'
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

# html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

html_theme = 'furo'

# -- Post process ------------------------------------------------------------
import collections
def remove_namedtuple_attrib_docstring(app, what, name, obj, skip, options):
    if type(obj) is collections._tuplegetter:
        return True
    return skip


def setup(app):
    app.connect('autodoc-skip-member', remove_namedtuple_attrib_docstring)