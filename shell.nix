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
  ];

  shellHook = ''
    if [ ! -d .venv ]; then
      python -m venv .venv
    fi

    source .venv/bin/activate

    export CUDA_PATH=${pkgs.cudatoolkit}

    if [ -f requirements.txt ]; then
        pip install --upgrade pip
        pip install -r requirements.txt
    fi
  '';
}