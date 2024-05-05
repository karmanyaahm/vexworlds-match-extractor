{
  description = "Thing";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs?ref=nixpkgs-unstable";
  };
  outputs = { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      pyopencv4 = pkgs.python3Packages.opencv4.override {
        enableGtk2 = true;
        gtk2 = pkgs.gtk2;
        #enableFfmpeg = true; #here is how to add ffmpeg and other compilation flags
        #ffmpeg_3 = pkgs.ffmpeg;
        };

    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          ffmpeg
          pyopencv4
          python3Packages.pytesseract

	  python3Packages.yt-dlp-light

        ];
        shellHook = ''
        '';
      };
      formatter.${system} = pkgs.nixpkgs-fmt;
    };
}

