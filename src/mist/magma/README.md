# MAGMa

MAGMa is an algorithm which takes as input a molecule and provides as output a list of fragment molecules of the parent.

In this project, MAGMa is used to label the fragment peaks of spectra datasets
with chemical formulae and corresponding smiles, to be used as an extra
training signal for models. The fragmentation code utilized is heavily inspired
by the [original source code](https://github.com/NLeSC/MAGMa).

`run_magma.py` can be run directly and requires the following arguments:

- **--spectra-dir**: The directory path containing the SIRIUS program outputs.
  To subset spectra, we use only peaks that have been preserved by SIRIUS as
  an initial cleaning step. The program can be adapted to use other spectra
  input sources.   
- **--output-dir**: The chosen output directory path to save the magma output files    
- **--lowest-penalty-filter**: If flag set, when selecting candidate chemical formulae and smiles to label spectra peaks, only candidates with the lowest penalty score (as assigned by the Magma fragmentation engine) will be selected    
- **--spec-labels**: TSV file containing all the smiles for the spectra being used.   
