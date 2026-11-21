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
    'numpy': ('https://numpy.org/doc/stable/', None),
    'jax': ('https://docs.jax.dev/en/latest/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}
intersphinx_disabled_domains = ['std']

# Shorthands used in docstring types, rewritten to documented targets before
# intersphinx resolves them (see ``rewrite_type_alias`` below).
TYPE_ALIASES = {
    'np.ndarray': 'numpy.ndarray',
    'ndarray': 'numpy.ndarray',
    'jnp.ndarray': 'jax.Array',
    'plt.axes': 'matplotlib.axes.Axes',
    'plt.Axes': 'matplotlib.axes.Axes',
    'plt.figure': 'matplotlib.figure.Figure',
    'plt.Figure': 'matplotlib.figure.Figure',
    'matplotlib.pyplot.axes': 'matplotlib.axes.Axes',
    'matplotlib.pyplot.figure': 'matplotlib.figure.Figure',
    'FuncAnimation': 'matplotlib.animation.FuncAnimation',
    'MeshModel': 'spice.models.mesh_model.MeshModel',
    'Model': 'spice.models.model.Model',
    'Binary': 'spice.models.binary.Binary',
    'PhoebeBinary': 'spice.models.binary.PhoebeBinary',
    'PhoebeConfig': 'spice.models.phoebe_utils.PhoebeConfig',
    'Filter': 'spice.spectrum.filter.Filter',
}

# Cross-reference targets with nothing to link to (checked by ``sphinx-build -n``):
# jaxtyping annotations and their shape strings, JAX/NumPy private typing
# internals that autodoc expands from signatures, the SpectrumEmulator TypeVar,
# and the plain-English parts of Google-style type fields.
nitpick_ignore_regex = [
    ('py:class', r'(jaxtyping\.)?(Float|Int|Bool|Array|Shaped)(\[.*\])?'),
    ('py:class', r'(jax\.)?jaxlib\..*'),
    ('py:class', r'numpy\._typing\..*'),
    ('py:class', r'numpy\.(number|bool|dtype|float64|int64)'),
    ('py:class', r'"[^"]*"'),
    ('py:class', r'(spice\.spectrum\.spectrum_emulator\.)?T'),
    ('py:class', r'(optional|default.*|sequence|shape.*|N|\(N,\)|callable|array-like|scalar|n_wavelengths)'),
    # Typing expressions written out in docstring type fields (Napoleon passes
    # them through as one unresolvable name).
    ('py:class', r'.*(Optional|Union|Tuple|List|Dict|Callable)\[.*'),
    ('py:class', r'.*\]+'),
    ('py:class', r'(int|float|str|bool)\]'),
    # numpy.typing.ArrayLike is a type alias (py:data), not a linkable class.
    ('py:class', r'(numpy\.typing\.)?(ArrayLike|array_like)'),
    # Literal values and choice sets in numpy-style type specs
    # ("{'a', 'b'}, default 5", "(elev, azim)").
    ('py:class', r'[\d.]+'),
    ('py:class', r'.*[{}].*'),
    ('py:class', r"'[^']*'"),
    ('py:class', r'(elev|azim)'),
    # Optional dependencies mocked in the docs build.
    ('py:class', r'(phoebe|astro_emulators_toolkit|polars)\..*'),
]

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


def rewrite_type_alias(app, env, node, contnode):
    """Point shorthand type names (``np.ndarray``, ``plt.Axes``, ``MeshModel``)
    at their documented targets. Runs before intersphinx, which then resolves
    the rewritten target; local targets are resolved here."""
    target = node.get('reftarget', '')
    if node.get('refdomain') != 'py' or target not in TYPE_ALIASES:
        return None
    node['reftarget'] = TYPE_ALIASES[target]
    if node['reftarget'].startswith('spice.'):
        return env.get_domain('py').resolve_xref(env, node['refdoc'], app.builder, node['reftype'],
                                                 node['reftarget'], node, contnode)
    return None


def setup(app):
    app.connect('autodoc-skip-member', remove_namedtuple_attrib_docstring)
    app.connect('missing-reference', rewrite_type_alias, priority=400)
    app.connect('missing-reference', resolve_public_alias)