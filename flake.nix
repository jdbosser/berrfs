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
        
        # customRustToolchain = pkgs.rust-bin.stable."1.70.0".default;
        craneLib = (crane.mkLib pkgs); #.overrideToolchain customRustToolchain;
        projectName =
          (craneLib.crateNameFromCargoToml { cargoToml = ./Cargo.toml; }).pname;

        projectVersion = (craneLib.crateNameFromCargoToml {
          cargoToml = ./Cargo.toml;
        }).version;

        crateCfg = {
          src = craneLib.cleanCargoSource (craneLib.path ./.);
          nativeBuildInputs = [ python pkgs.rustc ];
        };

	  lib-path = with pkgs; lib.makeLibraryPath [
	    libffi
	    openssl
	    # glibc
	    stdenv.cc
	  ];

	  createShellHook = python: ''
		    SOURCE_DATE_EPOCH=$(date +%s)
		    export "LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${lib-path}"
		    VENV=.venv

		    if test ! -d $VENV; then
		      python3.10 -m venv $VENV
		    fi
		    source ./$VENV/bin/activate
		    export PYTHONPATH=`pwd`/$VENV/${python.sitePackages}/:$PYTHONPATH
		    ln -sf ${python}/lib/python3.10/site-packages/* ./.venv/lib/python3.10/site-packages
		    #pip install -r requirements.txt
		  '';

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
          #cp target/wheels/${wheelName} $out/
          installPhase = old.installPhase + ''
	  	cp -r target $out/
          '';
        });
        pythonPackage = ps:
            ps.buildPythonPackage rec {
              pname = projectName;
              format = "pyproject";
              version = projectVersion;
	      cargoDeps = pkgs.rustPlatform.importCargoLock {
    		lockFile = ./Cargo.lock;
  		};
	      # build-system = [pgs.maturin];
	      propagetedBuildInputs = [pkgs.maturin];
		 nativeBuildInputs = with pkgs.rustPlatform; [
		    cargoSetupHook
		    maturinBuildHook
		  ];
              #src = "${crateWheel}/${wheelName}";
              src = ./.;
              doCheck = false;
              pythonImportsCheck = [ projectName ];
            };
        pythonApp = ps:
            ps.buildPythonApplication rec {
              pname = projectName;
              format = "wheel";
              version = projectVersion;
              src = "${crateWheel}/${wheelName}";
              doCheck = false;
              pythonImportsCheck = [ projectName ];
            };

	    
          pythonEnv = (python.withPackages
            (ps: [ (pythonPackage ps) ] ++ (with ps; [ numpy virtualenv venvShellHook ])));
        in
        
        # Some python packages have dependencies that 
        # are broken on 32-bit systems. Hence, 
        # we have this if case here. We have no results
        # in this flake for such systems. 
        if !(pkgs.lib.hasInfix "i686" system) then {
            devShells.old =  pkgs.mkShell {
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
		
		devShells.default = let 
			pythonEnv = python.withPackages (p: [p.numpy p.matplotlib]); 
			shell_hook = createShellHook pythonEnv; 
		in 
		pkgs.mkShell {
			buildInputs = [pkgs.maturin python pkgs.glibc];
			shellHook = shell_hook; 
		};

	    devShells.python = pkgs.mkShell {
		buildInputs = [pythonEnv pkgs.maturin];
		  shellHook = ''
		    SOURCE_DATE_EPOCH=$(date +%s)
		    export "LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${lib-path}"
		    VENV=.venv

		    if test ! -d $VENV; then
		      python3.10 -m venv $VENV
		    fi
		    source ./$VENV/bin/activate
		    export PYTHONPATH=`pwd`/$VENV/${python.sitePackages}/:$PYTHONPATH
		    ln -sf ${pythonEnv}/lib/python3.10/site-packages/* ./.venv/lib/python3.10/site-packages
		    #pip install -r requirements.txt
		  '';
	    };

	    packages.default = (pkgs.python3Packages.callPackage ./pack.nix {}); 
	    buildPythonPackage = (python: python.pkgs.callPackage ./pack.nix {});
	
	    apps = {
		default = {
		    type = "app";
		    program = "${pythonEnv}/bin/python";
		};
	    };


        } else {}
    );
}
