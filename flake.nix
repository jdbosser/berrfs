{
  description = "A very basic flake";

    
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.05";
    flakeutils.url = "github:numtide/flake-utils";
    filtc.url = "github:jdbosser/filtc";
    berpf.url = "github:jdbosser/berpf";
    mcsim.url = "github:jdbosser/mcsim";
    anim.url = "github:jdbosser/render-anim";
    crane = {
      url = "github:ipetkov/crane";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    filtc.inputs.nixpkgs.follows = "nixpkgs";
    berpf.inputs.nixpkgs.follows = "nixpkgs";
    mcsim.inputs.nixpkgs.follows = "nixpkgs";

  };


  outputs = { self, nixpkgs, flakeutils, filtc, berpf, mcsim, anim, crane}: 
    flakeutils.lib.eachDefaultSystem (system:
        let pkgs = nixpkgs.legacyPackages.${system}; 

        
        python = pkgs.python310;


        pfiltc = filtc.buildPythonPackage python;
        
        customRustToolchain = pkgs.rust-bin.stable."1.70.0".default;
        craneLib =
          (crane.mkLib pkgs).overrideToolchain customRustToolchain;
        projectName =
          (craneLib.crateNameFromCargoToml { cargoToml = ./Cargo.toml; }).pname;

        projectVersion = (craneLib.crateNameFromCargoToml {
          cargoToml = ./Cargo.toml;
        }).version;

        crateCfg = {
          src = craneLib.cleanCargoSource (craneLib.path ./.);
          nativeBuildInputs = [ python ];
        };

        wheelTail =
          "cp310-cp310-manylinux_2_34_x86_64"; # Change if pythonVersion changes
        wheelName = "${projectName}-${projectVersion}-${wheelTail}.whl";

        # Build the library, then re-use the target dir to generate the wheel file with maturin
        crateWheel = (craneLib.buildPackage (crateCfg // {
          pname = projectName;
          version = projectVersion;
          # cargoArtifacts = crateArtifacts;
        })).overrideAttrs (old: {
          nativeBuildInputs = old.nativeBuildInputs ++ [ pkgs.maturin ];
          buildPhase = old.buildPhase + ''
            maturin build --offline --target-dir ./target
          '';
          installPhase = old.installPhase + ''
            cp target/wheels/${wheelName} $out/
          '';
        });
        pythonPackage = ps:
            ps.buildPythonPackage rec {
              pname = projectName;
              format = "wheel";
              version = projectVersion;
              src = "${crateWheel}/${wheelName}";
              doCheck = false;
              pythonImportsCheck = [ projectName ];
            };
        in
        
        # Some python packages have dependencies that 
        # are broken on 32-bit systems. Hence, 
        # we have this if case here. We have no results
        # in this flake for such systems. 
        if !(pkgs.lib.hasInfix "i686" system) then {
            devShells.default =  pkgs.mkShell {
                  buildInputs = [
                    (python.withPackages (p: [
                        p.numpy 
                        # (p.matplotlib.override {
                        #     enableTk = true;
                        #     enableQt = false;
                        # })
                        p.matplotlib
                        p.setuptools
                        p.scipy 
                        p.mypy
                        p.tqdm
                        p.tkinter
                        p.joblib
                        p.numba
                        pfiltc 
			pythonPackage


                        # p.pyqt5
                        # p.ipython
                        # p.jupyter

                        # For documentation
                        # p.myst-parser
                        # p.sphinx


                    ]))
                    # pkgs.poetry
		    pkgs.maturin
                    pkgs.pyright


                  ];
                QT_QPA_PLATFORM_PLUGIN_PATH="${pkgs.qt5.qtbase.bin}/lib/qt-${pkgs.qt5.qtbase.version}/plugins";
            };

        } else {}
    );
}
