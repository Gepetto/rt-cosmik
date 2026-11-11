{
  description = "RT-COSMIK documentation: an mdBook, with an API reference generated from the docstrings";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    systems.url = "github:nix-systems/default";
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = import inputs.systems;
      perSystem =
        { lib, pkgs, ... }:
        let
          # The API reference reads the sources with griffe, without importing
          # them, so none of RT-COSMIK's own dependencies are needed here.
          python = pkgs.python3.withPackages (ps: [ ps.griffe ]);
          tools = [
            pkgs.mdbook
            pkgs.mdbook-katex
            python
          ];
        in
        {
          # `nix develop ./docs`, then `python3 docs/gen_api.py && mdbook serve docs`.
          devShells.default = pkgs.mkShell {
            name = "rt-cosmik-docs";
            packages = tools;
          };

          # `nix build ./docs`: the HTML book in ./result.
          packages.default = pkgs.stdenvNoCC.mkDerivation {
            name = "rt-cosmik-book";
            src = lib.fileset.toSource {
              root = ./..;
              fileset = lib.fileset.unions [
                ./.
                ../src
              ];
            };
            nativeBuildInputs = tools;
            buildPhase = ''
              runHook preBuild
              python3 docs/gen_api.py --src src --out docs/api
              mdbook build docs --dest-dir $out
              # mdBook copies every file of docs/ next to the pages: drop the
              # ones that build the book rather than belong to it.
              rm -f $out/{book.toml,flake.nix,flake.lock,gen_api.py}
              runHook postBuild
            '';
            dontInstall = true;
          };
        };
    };
}
