# The LaMAR Benchmark
# for Localization and Mapping in Augmented Reality

<img src="assets/logos.svg" width="40%"/>

This repository hosts the source code for our upcoming ECCV 2022 paper:

- LaMAR: Benchmarking Localization and Mapping for Augmented Reality
- Authors: Paul-Edouard Sarlin\*, Mihai Dusmanu\*, Johannes L. Schönberger, Pablo Speciale, Lukas Gruber, Viktor Larsson, Ondrej Miksik, and Marc Pollefeys

This pre-release contains the code required to load the data and run the evaluation. More details on the ground-truthing tools, data, and leaderboad will follow later.

## Usage

Requirements:
- Python >= 3.8
- [pycolmap](https://github.com/mihaidusmanu/pycolmap) installed from source (recommended) or via `pip install pycolmap`
- [hloc](https://github.com/cvg/Hierarchical-Localization) and its dependencies
- [raybender](https://github.com/cvg/raybender)
- [pyceres](https://github.com/cvg/pyceres)
- everything listed in `requirements.txt`, via `python -m pip install -r requirements.txt`

Running the single-frame evaluation:
```
python -m lamar_benchmark.run \
	--scene SCENE --ref_id map --query_id query_phone \
	--retrieval netvlad --feature sift --matcher mnn
```

By default, the script assumes that the data was placed in `./data/` and will write the intermediate dumps and final outputs to `./outputs/`.

## BibTex citation

Please consider citing our work if you use any code from this repo or ideas presented in the paper:

```
@inproceedings{sarlin2022lamar,
  author    = {Paul-Edouard Sarlin and
               Mihai Dusmanu and
               Johannes L. Schönberger and
               Pablo Speciale and
               Lukas Gruber and
               Viktor Larsson and
               Ondrej Miksik and
               Marc Pollefeys},
  title     = {{LaMAR: Benchmarking Localization and Mapping for Augmented Reality}},
  booktitle = {ECCV},
  year      = {2022},
}
```

## Legal Notices

Microsoft and any contributors grant you a license to the Microsoft documentation and other content
in this repository under the [Creative Commons Attribution 4.0 International Public License](https://creativecommons.org/licenses/by/4.0/legalcode),
see the [LICENSE](LICENSE) file, and grant you a license to any code in the repository under the [MIT License](https://opensource.org/licenses/MIT), see the
[LICENSE-CODE](LICENSE-CODE) file.

Microsoft, Windows, Microsoft Azure and/or other Microsoft products and services referenced in the documentation
may be either trademarks or registered trademarks of Microsoft in the United States and/or other countries.
The licenses for this project do not grant you rights to use any Microsoft names, logos, or trademarks.
Microsoft's general trademark guidelines can be found at http://go.microsoft.com/fwlink/?LinkID=254653.

Privacy information can be found at https://privacy.microsoft.com/en-us/

Microsoft and any contributors reserve all other rights, whether under their respective copyrights, patents,
or trademarks, whether by implication, estoppel or otherwise.
