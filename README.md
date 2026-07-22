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



## License
MIT
