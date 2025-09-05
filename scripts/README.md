Pour lancer le premier learning, il faut :

1. Récupérer la data sur partage dans results cosmik : https://partage.laas.fr/index.php/apps/files/files/14898673?dir=/results_cosmik 
Il faut au moins récupérer les 3d_keypoints_filtered.csv, les mks_data_gapfilled.csv et les mocap_downsampled_to_40hz.csv, q_cosmik_swika.csv, q_mocap_downsampled.csv pour les sujets et les trials pbmatiques (voir listes en haut de data_gestion.py)

2. Tout mettre dans un dossier par exemple "results_cosmik" avec l'arborescence suivante :
results_cosmik
├── cosmik
│   ├── cosmik_subject
│   │   ├── trial
│   │   │   ├── 3d_keypoints_filtered.csv
├── mocap
│   ├── mocap_subject
│   │   ├── trial
│   │   │   ├── mocap_downsampled_to_40hz.csv
│   │   │   ├── mks_data_gapfilled.csv
|   |   |   ├── q_cosmik_swika.csv
|   |   |   ├── q_mocap_downsampled.csv

3. Lancer le script data_gestion.py avec les arguments suivants :
python data_gestion.py /path/to/results_cosmik /path/to/dst_dataset/

4. Lancer le script get_jcp_from_mks_vicon.py avec les arguments suivants :
python get_jcp_from_mks_vicon.py /path/to/dst_dataset/

5. Lancer le script convert_csv_to_npz.py avec les arguments suivants :
python convert_csv_to_npz.py /path/to/dst_dataset/ /path/to/npz_dataset/

6. Lancer le script fine_tune_lstm_dynamic_only_HPE.py avec les arguments en-tête