{
  pkgs ?
    import <nixpkgs> {
      config = {
        cudaSupport = true;
        allowUnfree = true;
      };
    },
}: let
  python313Packages = pkgs.python313.withPackages (ps:
    with ps; [
      torchWithCuda
      torchaudio
      torchcodec
      transformers
      accelerate
      librosa
      soundfile
      scipy
      pandas
      numpy
      scikit-learn
      matplotlib
      tqdm
      kagglehub
      seaborn
      datasets
    ]);
in
  pkgs.mkShell {
    buildInputs = [
      python313Packages
      pkgs.cudatoolkit
    ];

    shellHook = ''
      ln -snf ${python313Packages} ./.venv
      export CUDA_PATH=${pkgs.cudatoolkit}
    '';
  }
