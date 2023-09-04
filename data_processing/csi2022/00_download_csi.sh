# Note: Only accessible on local machine for dev
cp /data/mass_spec_data/csi2022_export_v2.tar data/paired_spectra/csi2022_export.tar
cd data/paired_spectra/
tar -xvf csi2022_export.tar

mv csi2022_export csi2022
rm -f csi2022_export.tar

# Mv to precomputed fp location
mv csi2022/precomputed_fp/cache_csi_csi2022.h5 ../../fingerprints/precomputed_fp/
