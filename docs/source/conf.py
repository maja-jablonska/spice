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
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx'
]

# The API reference is generated from docstrings; optional heavy dependencies
# are mocked so the docs build needs only the core install.
autodoc_mock_imports = ['phoebe', 'h5py', 'astro_emulators_toolkit', 'zarr']
autodoc_member_order = 'bysource'
napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

html_static_path = ['_static']
html_css_files = ['custom.css']

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


def resolve_public_alias(app, env, node, contnode):
    """Resolve cross-references written with a public path (``spice.spectrum.simulate_observed_flux``)
    to the full module path under which autodoc documents the object
    (``spice.spectrum.spectrum.simulate_observed_flux``). The public names are lazy re-exports, so
    the object's own ``__module__`` gives the documented location."""
    import importlib
    target = node.get('reftarget', '')
    if node.get('refdomain') != 'py' or not target.startswith('spice.'):
        return None
    module_name, _, attr = target.rpartition('.')
    try:
        obj = getattr(importlib.import_module(module_name), attr)
    except Exception:
        return None
    full = f"{getattr(obj, '__module__', '')}.{getattr(obj, '__qualname__', attr)}"
    if full == target:
        return None
    return env.get_domain('py').resolve_xref(env, node['refdoc'], app.builder, node['reftype'], full, node, contnode)


def setup(app):
    app.connect('autodoc-skip-member', remove_namedtuple_attrib_docstring)
    app.connect('missing-reference', resolve_public_alias)