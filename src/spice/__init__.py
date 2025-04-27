def _patch_phoebe_anim_to_html():
    """
    Ensure PHOEBE uses a Python-3-safe anim_to_html and that the
    cached Animation._repr_html_ points to it.
    """
    import sys, importlib, base64, tempfile, warnings

    # 1) Provide a clean autofig module for PHOEBE
    sys.modules["phoebe.dependencies.autofig"] = importlib.import_module("autofig")

    # 2) Import PHOEBE *after* the redirect
    import phoebe
    from phoebe.dependencies.autofig import mpl_animate as _a
    import matplotlib.animation as _mpl_anim

    # 3) Patch only if the legacy line is still present
    if "video.encode(\"base64\")" in _a.anim_to_html.__code__.co_consts:
        def _anim_to_html_py3(anim, fps=20, codec="libx264"):
            if not hasattr(anim, "_encoded_video"):
                with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
                    anim.save(tmp.name, fps=fps,
                              extra_args=["-vcodec", codec])
                    anim._encoded_video = (
                        base64.b64encode(tmp.read()).decode("ascii")
                    )
            return _a.VIDEO_TAG.format(anim._encoded_video)

        # Replace the function **in the module**
        _a.anim_to_html = _anim_to_html_py3
        # Replace the cached reference used by Jupyter
        _mpl_anim.Animation._repr_html_ = _a.anim_to_html

        warnings.warn("Patched PHOEBE anim_to_html for Python 3")

# run the patch exactly once when spice is imported
_patch_phoebe_anim_to_html()
