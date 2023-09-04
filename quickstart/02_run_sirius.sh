SIRIUS=sirius/bin/sirius
CORES=32

# Assumes we know the correct chemical formula
echo "Processing quickstart" 
mkdir data/paired_spectra/quickstart
mkdir data/paired_spectra/quickstart/sirius_outputs

# Ignore formula
$SIRIUS --cores $CORES --output data/paired_spectra/quickstart/sirius_outputs/ --input quickstart/quickstart.mgf --ignore-formula formula --ppm-max-ms2 10
