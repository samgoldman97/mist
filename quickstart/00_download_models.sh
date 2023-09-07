contrast_model=https://www.dropbox.com/scl/fi/vzb5zumdaz254ly11e8j9/mist_contrastive_canopus_pretrain.ckpt?rlkey=f7e97yrafwcitcm43rm9x5vit
fp_model=https://www.dropbox.com/scl/fi/p9rz33w2bdmclgcsp7733/mist_fp_canopus_pretrain.ckpt?rlkey=w21ivhjd42jh0vi8j218cev9u

contrast_model=https://zenodo.org/record/8316682/files/mist_contrastive_canopus_pretrain.ckpt
fp_model=https://zenodo.org/record/8316682/files/mist_fp_canopus_pretrain.ckpt

wget -O pretrained_models/mist_fp_canopus_pretrain.ckpt  $fp_model
wget -O pretrained_models/mist_contrastive_canopus_pretrain.ckpt  $contrast_model
