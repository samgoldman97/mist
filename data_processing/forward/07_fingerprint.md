# 07_fingerprint.md

Need to fingrprint the resulting smiles
Can do this by calling docker file and cloning ms novelist into another
directory. Requires more setup and only necessary for head-to-head with SIRIUS.



```
cp data/unpaired_mols/bio_mols/all_smis.txt ../MSNovelist/fp_data/all_smis_csi2022.txt

cd ../MSNovelist

python3 /fp_scripts/fingerprint_smiles.py --fp-map /fp_scripts/fp_map.p --smi-list /fp_data/all_smis_csi2022.txt --out-prefix cache_csi_csi2022 --workers 20 

mv ../MSNovelist/fp_data/fp_out/cache_csi_csi2022* fingerprints/precomputed_fp/
```
