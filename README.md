# LNP Cytotoxicity Dataset — SMILES Generation

Code used to build the machine-readable ionizable lipid structures behind the LNP cytotoxicity dataset.

Most source studies report their lipid libraries as a set of building blocks (head groups, tails, and in some cases linkers) rather than listing every full structure.
This repo contains the scripts that reconstruct the complete set of molecules for each of those studies and write out canonical SMILES.

## Repository structure

One folder per source dataset:

```
[Dataset Identifier]/
    [python files]     # Python scripts used to generate the data
    [csv files]        # CSV files containing data, including the individual components
```

`Miller_zwitter` and `Lee_unsat` did not require generation — structures for these studies were already available in machine-readable form through LNPDB, so those folders contain the extracted structures rather than generation code.

## How it works

For each study, the reported building blocks are combined according to the reaction chemistry described in the source paper. Structures are built and canonicalized with RDKit. 

## Requirements

- Python 3.11.13
- RDKit 2025.03.6
- Pandas 2.3.3


## Citation

If you use this code or the LNPCD dataset, please cite the dataset archive:

> Liu, N., Zhu, Q. & Lu, Z.-R. A cytotoxicity dataset for data-driven design of
> ionizable lipid nanoparticles. Zenodo https://doi.org/10.5281/zenodo.23428876 (2026).

## Related

* **Dataset archive (DOI):** https://doi.org/10.5281/zenodo.23428876
* **Interactive database:** https://lnpcd.pages.dev
* **Web app source:** https://github.com/Nathan6006/lnpcd

## License
The code in this repository is released under the MIT License (see [`LICENSE`](LICENSE)).
The LNPCD dataset itself is distributed separately under CC BY 4.0.
