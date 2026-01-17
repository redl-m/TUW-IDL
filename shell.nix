{
  pkgs ? import <nixpkgs> {
    config = {
      cudaSupport = true;
      allowUnfree = true;
    };
  },
}:
pkgs.mkShell {
  buildInputs = with pkgs; [
    python313
    cudatoolkit
    libsndfile
    zlib
  ];

  shellHook = ''
    if [ ! -d .venv ]; then
      python -m venv .venv
    fi

    source .venv/bin/activate

    export CUDA_PATH=${pkgs.cudatoolkit}
    export LD_LIBRARY_PATH="/run/opengl-driver/lib:${pkgs.cudatoolkit}/lib:${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:$LD_LIBRARY_PATH"

    if [ -f requirements.txt ]; then
        pip install --upgrade pip
        pip install -r requirements.txt
        pip install notebook jupyterlab ipython ipywidgets
    fi
  '';
}