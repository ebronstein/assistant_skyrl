Custom Installation for CUDA 12.2 and GLIBC 2.3.1 Systems
===========================================================

This guide provides installation instructions for skyrl-train on systems with CUDA 12.2 and GLIBC 2.3.1, where the standard installation instructions do not work.

Steps
-----

1. Create and activate a virtual environment:

.. code-block:: shell

   uv venv --python 3.12 /path/to/venv
   source activate /path/to/venv

2. Remove ``flash-attn`` from ``pyproject.toml`` and run:

.. code-block:: shell

   uv lock
   uv sync --active --extra vllm

3. Build ``flash-attn`` from source:

.. code-block:: shell

   export FLASH_ATTENTION_FORCE_BUILD=TRUE
   export MAX_JOBS=64 # Decrease depending on CPUs available

   # Build flash-attn from source. Can remove -v flag if verbose output not needed.
   uv pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.8.0.post2 --no-build-isolation -v

4. Verify ``flash-attn`` was installed (should return without error):

.. code-block:: shell

   python -c "from flash_attn import flash_attn_interface"

5. Run the quickstart example, but modify ``examples/gsm8k/run_gsm8k.sh`` to use ``uv run --active --no-sync`` instead of ``uv run --isolated`` to prevent ``uv`` from syncing the virtual environment. Also, disable the FlashInfer attention backend because it causes errors (cannot find some standard C libraries). Use flash-attn instead:

.. code-block:: shell

   VLLM_USE_FLASHINFER_SAMPLER=0 bash examples/gsm8k/run_gsm8k.sh

Important Notes
---------------

- Make sure your temp directory is not set or is set to ``/tmp``. For some reason, setting it to a location in your NAS directory will cause the same errors related to not finding standard C libraries.
- This installation method was tested on a system with CUDA 12.2 and GLIBC 2.3.1, where the standard installation fails.
- After installation, you may need to be careful when using `uv` because it may try to sync the environment with the `uv.lock`` file, which means it will uninstall flash-attn. In scripts that use `uv run --isolated`, use `uv run --active --no-sync` instead in order to use the active virtual environment and not sync it with the lock file. When using `uv pip install`, you can pass the `--dry-run` flag to make sure flash-attn would not be uninstalled.
