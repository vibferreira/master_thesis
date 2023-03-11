#!/bin/bash -l
# L'argument '-l' est indispensable pour bénéficier des directives de votre .bashrc
 
# On peut éventuellement placer ici les commentaires SBATCH permettant de définir les paramètres par défaut de lancement :
#SBATCH --chdir=/share/home/barbosaf/master_thesis
#SBATCH --gres gpu:1
##SBATCH --cpu-per-task=10
#SBATCH --mem 10G
#SBATCH --mail-type FAIL,END
#SBATCH --mail-user ferreiravitoriabarbosa@gmail.com
#SBATCH --partition longrun
##SBATCH --account barbosaf
# Si besoin, spécification de la version CUDA mise à disposition sur le cluster
# setcuda 11.7
 
# Exécution du script habituellement utilisé, on utilise la variable CUDA_VISIBLE_DEVICES qui contient la liste des GPU logiques actuellement réservés (toujours à partir de 0)
 
##./darknet detector train -map data/turtles.data cfg/yolov3-turtles_test_sn1.cfg darknet53.conv.74  -dont_show -gpus $CUDA_VISIBLE_DEVICES

conda activate help
python main.py
