sudo docker run -dit -v .data:/habitat-lab/data -v ./main.py:/habitat-lab/main.py --gpus all fairembodied/habitat-challenge:habitat_navigation_2023_base_docker
# Commands to run inside container:
# cd /habitat-lab
# . activate habitat
#python main.py
# torchrun --nproc_per_node=1 habitat-baselines/habitat_baselines/run.py --config-name=rule_based_train.yaml --config-dir=habitat-baselines/habitat_baselines/config/objectnav
