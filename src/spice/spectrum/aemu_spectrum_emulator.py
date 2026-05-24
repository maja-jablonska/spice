from typing import List, Dict, Any, Callable, Optional
from typing_extensions import override
from jaxtyping import ArrayLike
import numpy as np
import jax
import jax.numpy as jnp
import os
import warnings

from spice.spectrum.flux_limb_darkening import apply_flux_limb_darkening
from spice.spectrum.solar_parameters import SOLAR_PARAMETERS
from spice.spectrum.spectrum_emulator import SpectrumEmulator


def _lazy_import_astro_emulators_toolkit():
    try:
        import astro_emulators_toolkit
    except ImportError as e:
        raise ValueError(
            "astro_emulators_toolkit is required for this functionality but is "
            "not installed. Install via the optional extra declared in "
            "pyproject.toml: `pip install \"stellar-spice[aemu]\"`."
        ) from e
    try:
        from astro_emulators_toolkit.emulator_runtime import (
            apply_jax_runtime,
            make_frozen_apply_runtime,
        )
    except ImportError as e:
        raise ImportError(
            "astro_emulators_toolkit is installed but failed to import "
            "(commonly a flax/JAX version mismatch — newer flax.nnx requires "
            "jax._src.core.mutable_array, added in JAX 0.4.34). Original "
            f"error: {e}"
        ) from e
    return astro_emulators_toolkit, apply_jax_runtime, make_frozen_apply_runtime


aemu, _apply_jax_runtime, _make_frozen_apply_runtime = _lazy_import_astro_emulators_toolkit()


def _load_emulator(name: str) -> "aemu.Emulator":
    """Load an :class:`aemu.Emulator` from ``name``.

    Accepts either a local bundle directory (loaded with ``Emulator.from_bundle``)
    or a Hugging Face repo id (loaded with ``Emulator.from_pretrained``). The
    older code path only supported the latter, which broke when ``name`` was a
    path to a bundle produced by a local training run.
    """
    try:
        is_dir = os.path.isdir(name)
    except (TypeError, ValueError):
        is_dir = False
    if is_dir:
        return aemu.Emulator.from_bundle(name)
    return aemu.Emulator.from_pretrained(name)


def _affine_section_of_combined(combined: Any, side: str) -> Optional[Dict[str, Any]]:
    """Extract the input- or output-side affine block from a *combined*
    ``reference_scaling`` block whose ``min_tree`` / ``max_tree`` are nested by
    section (``{"inputs": {...}, "outputs": {...}}``). Returns ``None`` when the
    block is absent or not laid out this way."""
    if not isinstance(combined, dict):
        return None
    min_tree = combined.get("min_tree")
    max_tree = combined.get("max_tree")
    if not isinstance(min_tree, dict) or not isinstance(max_tree, dict):
        return None
    side_min = min_tree.get(side)
    side_max = max_tree.get(side)
    if not isinstance(side_min, dict) or not isinstance(side_max, dict):
        return None
    block: Dict[str, Any] = {"min_tree": side_min, "max_tree": side_max}
    for key in ("kind", "parametrization"):
        value = combined.get(key)
        if value is not None:
            block[key] = value
    return block


def _affine_reference_scaling(emulator: "aemu.Emulator", side: str) -> Optional[Dict[str, Any]]:
    """Return ``{"min_tree": ..., "max_tree": ...}`` affine reference scaling for
    ``side`` (``"inputs"`` / ``"outputs"``), tolerating the toolkit versions we
    support:

    * older bundles expose split top-level ``reference_scaling_inputs`` /
      ``reference_scaling_outputs`` blocks (and the ``Emulator`` surfaces them
      as attributes);
    * newer bundles keep a single ``reference_scaling`` block whose
      ``min_tree`` / ``max_tree`` are nested by section.

    Falls back to reading the raw ``emulator.spec`` if the convenience
    attributes are missing entirely. Returns ``None`` when nothing usable is
    found (callers already treat ``None`` as "bundle declares no scaling").
    """
    block = getattr(emulator, f"reference_scaling_{side}", None)
    if isinstance(block, dict) and isinstance(block.get("min_tree"), dict) and isinstance(block.get("max_tree"), dict):
        return block

    section = _affine_section_of_combined(getattr(emulator, "reference_scaling", None), side)
    if section is not None:
        return section

    spec = getattr(emulator, "spec", None)
    if isinstance(spec, dict):
        block = spec.get(f"reference_scaling_{side}")
        if isinstance(block, dict) and isinstance(block.get("min_tree"), dict) and isinstance(block.get("max_tree"), dict):
            return block
        section = _affine_section_of_combined(spec.get("reference_scaling"), side)
        if section is not None:
            return section
    return None


def _input_domain_param_bounds(emulator: "aemu.Emulator", which: str) -> Optional[ArrayLike]:
    """Return the ``parameters`` leaf of ``input_domain`` ``{which}_tree``
    (``which`` is ``"min"`` / ``"max"``), tolerating both the flat
    (``{"parameters": ...}``) and section-nested (``{"inputs": {"parameters": ...}}``)
    layouts. Returns ``None`` if the bundle declares no such bounds."""
    domain = getattr(emulator, "input_domain", None)
    if not isinstance(domain, dict):
        return None
    tree = domain.get(f"{which}_tree")
    if not isinstance(tree, dict):
        return None
    if "parameters" in tree:
        return tree["parameters"]
    inputs = tree.get("inputs")
    if isinstance(inputs, dict) and "parameters" in inputs:
        return inputs["parameters"]
    return None


def _np_parameter_helper(interpolator, parameter_values: Dict[str, Any] = None) -> ArrayLike:
    # Use ``is None`` / dict-empty check explicitly: ``if not parameter_values``
    # raises on multi-element numpy arrays ("truth value is ambiguous").
    if parameter_values is None or (isinstance(parameter_values, dict) and not parameter_values):
        parameter_values = SOLAR_PARAMETERS

    if isinstance(parameter_values, dict):
        parameters = np.array([parameter_values.get(name, SOLAR_PARAMETERS.get(name, 0.0)) for name in interpolator.stellar_parameter_names])
        for i, name in enumerate(interpolator.stellar_parameter_names):
            if name in parameter_values:
                parameters[i] = parameter_values[name]
    else:
        parameters = np.array(parameter_values)

    if not (np.all(parameters >= interpolator.min_stellar_parameters) and np.all(parameters <= interpolator.max_stellar_parameters)):
        warnings.warn("Possible exceeding parameter bounds - extrapolating.")

    return parameters


class AemuSpectrumEmulator(SpectrumEmulator[ArrayLike]):
    def __init__(self, emulator: aemu.Emulator):
        self._emulator = emulator
        # Input-side affine reference scaling. Newer toolkit builds no longer
        # expose ``Emulator.reference_scaling_inputs`` directly (they keep a
        # single ``reference_scaling`` block nested by section), so resolve it
        # tolerantly. ``None`` means the bundle declares no input scaling.
        self.ref = _affine_reference_scaling(emulator, "inputs")
        self._apply: Callable[..., Any] = self._build_frozen_apply(emulator)

    @staticmethod
    def _build_frozen_apply(emulator: "aemu.Emulator") -> Callable[..., Any]:
        # Close over the current params/model_state so the inner apply can run
        # under jax.jit / vmap. Going through Emulator.predict would force a
        # device->numpy roundtrip that is incompatible with tracing.
        task = emulator._ensure_task()
        post_fn = (
            task.postprocess_pred
            if (task is not None and hasattr(task, "postprocess_pred"))
            else None
        )
        return _make_frozen_apply_runtime(
            graphdef=emulator.graphdef,
            params=emulator.params,
            model_state=emulator.model_state,
            post_fn=post_fn,
            jit=False,
        )

    @property
    def emulator(self) -> aemu.Emulator:
        return self._emulator

    def scale_parameters(self, parameters: ArrayLike) -> Dict[str, ArrayLike]:
        # Return a ``{"parameters": ...}`` tree even when the bundle does not
        # advertise reference_scaling_inputs, so callers (notably
        # ``to_parameters``) can uniformly index ``["parameters"]``.
        if self.ref is None:
            return {"parameters": parameters}

        return aemu.normalize_tree(
            {"parameters": parameters},
            self.ref["min_tree"],
            self.ref["max_tree"],
        )

    def _bundle_parameter_names(self) -> List[str]:
        """The full ordered list of input parameter channel names recorded
        on the bundle, including ``mu`` when the bundle is an intensity
        emulator."""
        return list(self.emulator.input_spec['channel_names_tree']['parameters'])

    def _mu_index(self) -> Optional[int]:
        names = self._bundle_parameter_names()
        return names.index('mu') if 'mu' in names else None

    @override
    @property
    def stellar_parameter_names(self) -> List[str]:
        return [p for p in self._bundle_parameter_names() if p != 'mu']

    @override
    @property
    def min_stellar_parameters(self) -> ArrayLike:
        bounds = _input_domain_param_bounds(self.emulator, "min")
        if bounds is None:
            raise ValueError("Bundle does not declare input_domain parameter bounds.")
        bounds = np.asarray(bounds)
        idx = self._mu_index()
        return bounds if idx is None else np.delete(bounds, idx)

    @override
    @property
    def max_stellar_parameters(self) -> ArrayLike:
        bounds = _input_domain_param_bounds(self.emulator, "max")
        if bounds is None:
            raise ValueError("Bundle does not declare input_domain parameter bounds.")
        bounds = np.asarray(bounds)
        idx = self._mu_index()
        return bounds if idx is None else np.delete(bounds, idx)

    def to_parameters(self, parameters: ArrayLike = None) -> ArrayLike:
        # Convert a (possibly partial) parameter dict into an ordered array of
        # raw stellar values matching ``stellar_parameter_names``. Scaling to
        # the bundle's training space happens inside ``flux``/``intensity`` via
        # ``scale_parameters``, so callers (and ``MeshModel.parameters``) can
        # always carry the natural-units representation.
        return _np_parameter_helper(self, parameters)

    def flux(self, log_wavelengths: ArrayLike, parameters: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def intensity(self, log_wavelengths: ArrayLike, mu: float, parameters: ArrayLike) -> ArrayLike:
        """Calculate the intensity for given wavelengths and mus

        Args:
            log_wavelengths (ArrayLike): [log(angstrom)]
            mu (float): cosine of the angle between the star's radius and the line of sight
            parameters (ArrayLike): an array of predefined stellar parameters

        Returns:
            ArrayLike: intensities corresponding to passed wavelengths [erg/cm2/s/angstrom]
        """
        raise NotImplementedError



class PretrainedAemuSpectrumEmulator(AemuSpectrumEmulator):
    def __init__(self, name: str):
        # ``name`` may be a Hugging Face repo id or a local bundle directory.
        emulator = _load_emulator(name)
        super().__init__(emulator)



def _infer_native_wavelengths(emulator: "aemu.Emulator") -> Optional[np.ndarray]:
    """Try to read the emulator's native flux wavelength grid (Angstroms).

    Bundles published with this toolkit usually stash the grid under
    ``bundle_extras["wavelength_angstrom"]``. Fall back to parsing
    ``output_spec.channel_names_tree['flux']`` when channel names are numeric.
    Returns None when nothing usable is found."""
    extras = emulator.bundle_extras or {}
    wave = np.asarray(extras.get("wavelength_angstrom", []), dtype=np.float32)
    if wave.size:
        return wave

    output_spec = emulator.output_spec or {}
    channel_names_tree = output_spec.get('channel_names_tree') or {}
    channels = channel_names_tree.get('flux')
    if not channels:
        return None
    try:
        return np.asarray([float(c) for c in channels], dtype=np.float64)
    except (TypeError, ValueError):
        return None


class FluxPretrainedAemuSpectrumEmulator(PretrainedAemuSpectrumEmulator):
    """Fixed-grid flux bundle: a single ``flux`` output channel sampled on a
    pre-baked wavelength grid stored in ``bundle_extras['wavelength_angstrom']``
    (or, as a fallback, parsed from numeric channel names). Use this for
    bundles like ``RozanskiT/example_bundle``.
    """

    def __init__(
        self,
        name: str,
        native_wavelengths: Optional[ArrayLike] = None,
    ):
        super().__init__(name)
        if native_wavelengths is None:
            native_wavelengths = _infer_native_wavelengths(self.emulator)
        if native_wavelengths is None:
            raise ValueError(
                "Could not infer the emulator's native wavelength grid from its "
                "output_spec.channel_names_tree['flux']. Pass native_wavelengths "
                "(Angstroms) explicitly to FluxPretrainedAemuSpectrumEmulator(...)."
            )
        self._native_wavelengths = jnp.asarray(native_wavelengths)

    def _disc_flux(self, log_wavelengths: ArrayLike, parameters: ArrayLike) -> ArrayLike:
        # The bundle's reference_scaling_inputs are applied here so the
        # caller (e.g. ``simulate_observed_flux``) can pass raw stellar
        # parameters per mesh element. ``scale_parameters`` is a no-op when
        # the bundle does not advertise scaling.
        scaled = self.scale_parameters(parameters)["parameters"]
        emu_flux = self._apply({"parameters": jnp.atleast_2d(scaled)})['flux']
        # (1, n_native) -> (n_native,)
        emu_flux = emu_flux.reshape(-1)
        target_wavelengths = jnp.power(10.0, log_wavelengths)
        return jnp.interp(target_wavelengths, self._native_wavelengths, emu_flux)

    @override
    def flux(self, log_wavelengths: ArrayLike, parameters: ArrayLike) -> ArrayLike:
        # Disc-integrated flux. Bundles in this family expose a single flux
        # channel; spice's contract is (n_wavelengths, [flux, continuum]) so
        # duplicate the curve as the continuum proxy.
        resampled = self._disc_flux(log_wavelengths, parameters)
        return jnp.stack([resampled, resampled], axis=-1)

    @override
    def intensity(
        self,
        log_wavelengths: ArrayLike,
        mu: float,
        parameters: ArrayLike,
        ld_law: str = "linear",
        ld_coeffs: Optional[ArrayLike] = None,
    ) -> ArrayLike:
        """Specific intensity at angle ``mu`` derived from the disc-integrated
        flux via flux-conservation limb darkening.

        Args:
            log_wavelengths: log10(Angstrom) sample points.
            mu: cosine of the angle from disc centre.
            parameters: stellar parameters in training order.
            ld_law: name of the limb-darkening law (``linear``, ``quadratic``,
                ``nonlinear_4``).
            ld_coeffs: coefficients for the chosen law. Defaults to
                ``[0.6, 0.6, 0, 0]`` (typical solar-type linear LD).
        """
        resampled = self._disc_flux(log_wavelengths, parameters)
        mu_value = jnp.squeeze(mu)
        intensity = apply_flux_limb_darkening(resampled, mu_value, ld_law, ld_coeffs)
        return jnp.stack([intensity, intensity], axis=-1)


class IntensityPretrainedAemuSpectrumEmulator(PretrainedAemuSpectrumEmulator):
    """Wavelength-conditioned, two-channel, log10-output intensity bundle.

    Matches bundles like ``RozanskiT/TPayne-spice-small-random`` (and the
    locally-trained "new FE" Tpayne intensity bundles):
        * Inputs:  ``{"parameters": (B, n_p), "wavelengths": (B, n_w)}`` —
          ``wavelengths`` are **log10(Angstrom)**. The full input parameter
          channel list includes ``mu`` (the bundle is a true intensity
          emulator); spice keeps ``mu`` separate from the rest of the
          stellar parameters, so this class splices it back in at call time.
        * Outputs: ``{"flux": (B, n_w, 2)}`` with channels
          ``[log10 line intensity, log10 continuum intensity]``.
        * Both ``inputs`` and ``outputs`` are min-max normalised against the
          bundle's affine reference scaling — exposed either as split
          ``reference_scaling_inputs`` / ``reference_scaling_outputs`` blocks
          (older bundles) or as a single section-nested ``reference_scaling``
          block (newer bundles); both layouts are handled here.

    The forward pass mirrors the reference quickstart shipped with the bundle:
    normalise → apply → denormalise → ``10**y``.
    """

    def __init__(self, name: str):
        super().__init__(name)
        ref_outputs = _affine_reference_scaling(self.emulator, "outputs")
        if self.ref is None or ref_outputs is None:
            raise ValueError(
                "IntensityPretrainedAemuSpectrumEmulator requires the bundle to "
                "declare both input and output reference scaling (got "
                f"reference_scaling_inputs={'present' if self.ref is not None else 'missing'}, "
                f"reference_scaling_outputs={'present' if ref_outputs is not None else 'missing'})."
            )
        mu_idx = self._mu_index()
        if mu_idx is None:
            raise ValueError(
                "IntensityPretrainedAemuSpectrumEmulator expects 'mu' in the "
                f"bundle's input parameter channel names, got "
                f"{self._bundle_parameter_names()!r}."
            )
        self._ref_outputs = ref_outputs
        self._mu_pos = mu_idx
        # Output leaf name in the bundle's output tree. The training script
        # writes the two intensity channels under "flux".
        self._output_leaf = "flux"

    def _splice_mu(self, parameters: ArrayLike, mu: ArrayLike) -> ArrayLike:
        """Return a (n_p+1,) array with ``mu`` inserted at ``self._mu_pos``,
        matching the order of the bundle's input parameter channels."""
        params = jnp.asarray(parameters).reshape(-1)
        mu_scalar = jnp.squeeze(jnp.asarray(mu))
        return jnp.concatenate([
            params[: self._mu_pos],
            mu_scalar[None],
            params[self._mu_pos :],
        ])

    def _intensity_at_mu(
        self,
        log_wavelengths: ArrayLike,
        parameters_with_mu: ArrayLike,
    ) -> ArrayLike:
        """Run the bundle once; return ``(n_w, 2)`` linear intensity."""
        x_physical = {
            "parameters": jnp.atleast_2d(parameters_with_mu),
            "wavelengths": jnp.atleast_2d(log_wavelengths),
        }
        x_scaled = aemu.normalize_tree(
            x_physical,
            self.ref["min_tree"],
            self.ref["max_tree"],
        )
        y_scaled = self._apply(x_scaled)
        y_log10 = aemu.denormalize_tree(
            y_scaled,
            self._ref_outputs["min_tree"],
            self._ref_outputs["max_tree"],
        )
        # (1, n_w, 2) -> (n_w, 2). Bundle stores log10; convert back to linear.
        return jnp.power(10.0, y_log10[self._output_leaf][0])

    @override
    def intensity(
        self,
        log_wavelengths: ArrayLike,
        mu: float,
        parameters: ArrayLike,
    ) -> ArrayLike:
        """Specific intensity at angle ``mu``. ``mu`` is an explicit input
        parameter of the bundle, so the network already encodes the full
        I(μ, λ, θ) dependence — no external limb-darkening law is needed."""
        full_parameters = self._splice_mu(parameters, mu)
        return self._intensity_at_mu(log_wavelengths, full_parameters)

    @override
    def flux(
        self,
        log_wavelengths: ArrayLike,
        parameters: ArrayLike,
        mus_number: int = 20,
    ) -> ArrayLike:
        """Disc-integrated flux: ``F = 2π ∫₀¹ I(μ) μ dμ`` evaluated by
        Gauss–Legendre quadrature over the [0, 1] μ range, mirroring the
        approach in :class:`IntensityLazyZarrInterpolator`."""
        roots, weights = np.polynomial.legendre.leggauss(mus_number)
        roots = (roots + 1) / 2  # map [-1, 1] -> [0, 1]
        weights = weights / 2
        roots_j = jnp.asarray(roots)
        weights_j = jnp.asarray(weights)
        # vmap intensity over mu; broadcast μ·w against the (n_w, 2) channels.
        per_mu = jax.vmap(self.intensity, in_axes=(None, 0, None))(
            log_wavelengths, roots_j, parameters
        )
        return 2 * jnp.pi * jnp.sum(
            per_mu * roots_j[:, None, None] * weights_j[:, None, None],
            axis=0,
        )
