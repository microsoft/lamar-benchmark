<p align="center">
  <h1 align="center"><img src="assets/lamar_white.svg" width="85"><br>LaMAR<br>Benchmarking Localization and Mapping<br>for Augmented Reality</h1>
  <p align="center">
    <a href="https://psarlin.com/">Paul-Edouard Sarlin*</a>
    ·
    <a href="https://dsmn.ml/">Mihai Dusmanu*</a>
    <br>
    <a href="https://demuc.de/">Johannes L. Schönberger</a>
    ·
    <a href="https://www.microsoft.com/en-us/research/people/paspecia/">Pablo Speciale</a>
    ·
    <a href="https://www.microsoft.com/en-us/research/people/lugruber/">Lukas Gruber</a>
    ·
    <a href="https://vlarsson.github.io/">Viktor Larsson</a>
    ·
    <a href="http://miksik.co.uk/">Ondrej Miksik</a>
    ·
    <a href="https://www.microsoft.com/en-us/research/people/mapoll/">Marc Pollefeys</a>
  </p>
<p align="center">
    <img src="assets/logos.svg" alt="Logo" height="40">
</p>
  <h2 align="center">ECCV 2022</h2>
  <h3 align="center"><a href="https://lamar.ethz.ch/">Project Page</a> | <a href="https://youtu.be/32XsRli2coo">Video</a></h3>
  <div align="center"></div>
</p>
<p align="center">
    <a href="https://lamar.ethz.ch/"><img src="assets/teaser.svg" alt="Logo" width="80%"></a>
    <br /><em>LaMAR includes multi-sensor streams recorded by AR devices along hundreds of unconstrained trajectories<br>captured over 2 years in 3 large indoor+outdoor locations.</em>
</p>

##

This repository hosts the source code for LaMAR, a new benchmark for localization and mapping with AR devices in realistic conditions. We are still in the process of fully releasing the benchmark. Here is the release plan:

- [x] Evaluation data: [apply here](https://lamar.ethz.ch/lamar/)
- [x] Evaluation, baselines, data format: see below
- [ ] Additional documentation
- [ ] Ground truthing pipeline: to be released soon
- [ ] Full raw data: to be released soon
- [ ] Leaderboard and evaluation server

## Running the evaluation

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
