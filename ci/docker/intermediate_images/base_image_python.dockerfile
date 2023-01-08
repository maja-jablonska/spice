ARG BASE_IMAGE

FROM ${BASE_IMAGE} as Image_python

    RUN apt-get -qq update; apt-get install -qqy graphviz python3-pygraphviz;
    RUN python3 -m pip install --upgrade pip setuptools
    RUN python3 -m pip install --no-cache-dir astropy scikit-learn pyYAML matplotlib \
                                              pillow scipy numpy neptune-client optuna \
                                              neptune-tensorflow-keras psutil \
                                              neptune-optuna numba pydot tensorboard-plugin-profile
    RUN python3 -m pip install --upgrade jax[cuda11_cudnn805] -f https://storage.googleapis.com/jax-releases/jax_releases.html
    RUN python3 -m pip install --upgrade git+https://github.com/google/flax.git
    RUN python3 -m pip install --no-cache-dir optax
    RUN python3 -m pip install --no-cache-dir phoebe==2.4
