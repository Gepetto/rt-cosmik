1. Allumer Vicon et les plateformes de force en branchant le câble débranché et en pressant sur les 2 boutons "Power"
2. Commande "$ xhost +local:" dans le terminal du PC rt-cosmik puis "$ code ." puis aller dans "Remote Explorer" à gauche et appuyer sur la flèche pour ouvrir le container de cosmik
3. Taper "$ source /deps_ws/devel/setup.bash" dans le terminal de cosmik
4. Placer les caméras à un endroit stratégique à l'aide du retour de rt-cosmik/scripts/run_pose_estimator_batched.py
5. Lancer le script "calib_runner.sh" dans le terminal de cosmik depuis le repo rt_cosmik directement (commande "$ bash calib_mocaptocam/calib_runner.sh")
6. Au milieu du script, avant d'appuyer sur P, calibrer le système Vicon
7. Une fois le script terminé, ouvrir un autre terminal sur lequel lancer gepetto viewer (commande "$ gepetto-gui") puis un autre pour lancer scripts/synch_vicon qui est sur la branche "main" de rt-cosmik

for trial in trials :
8. Mettre le nom du sujet et du trial dans "settings.py" et dans Nexus puis vérifier que le signal de l'Arduino est à zéro
9. Enfin sur le premier terminal, lancer run_pipeline.py lorsqu'on est prêt à démarrer un trial (après avoir lancé l'enregistrement sur Nexus) puis appuyer sur S pour commencer le trial
10. Presser Q puis puis Ctrl+C dans le terminal de run_pipeline puis stopper l'enrigistrement sur Nexus pour arrêter le trial et save les données
end

11. Tout éteindre quand les manips sont finies


N.B.:
gauche = bas
droite = haut