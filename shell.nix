{
  pkgs ?
    import <nixpkgs> {
      config = {
        cudaSupport = true;
        allowUnfree = true;
      };
    },
}: let
  # Define package audiomentations missing in nixpkgs
  audiomentations = pkgs.python313Packages.buildPythonPackage rec {
    pname = "audiomentations";
    version = "0.43.1";
    format = "setuptools";

    # Fetch source from PyPI
    src = pkgs.python313Packages.fetchPypi {
      inherit pname version;
      hash = "sha256-3jCJA8bZxD2D/hzuswm++vsbfOQWSp5xb1AAnEjltzw=";
    };

    doCheck = false; # Skip tests to save build time/fix minor errors
  };

  pythonEnv = pkgs.python313.withPackages (ps:
    with ps; [
      audiomentations
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
      pythonEnv
      pkgs.cudatoolkit
    ];

    shellHook = ''
      ln -snf ${pythonEnv} ./.venv
      export CUDA_PATH=${pkgs.cudatoolkit}
    '';
  }