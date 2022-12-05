ARG BASE_IMAGE

FROM ${BASE_IMAGE} as user_image

    ARG UID
    ARG GID

    ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu"

    RUN groupadd --gid ${GID} developer
    RUN useradd --create-home -u ${UID} -g ${GID} developer && echo "developer:1234" | chpasswd && adduser developer sudo

    WORKDIR /home/developer/workdir
    RUN chown -R ${UID}:${GID} /tf /usr/local/share/ ${HOME} /usr/local/cuda/	

    EXPOSE 8888

    USER developer
    
    RUN python3 -m ipykernel.kernelspec
    CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/home/developer/workdir --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.max_buffer_size=10000000000"]
