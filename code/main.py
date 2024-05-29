#!/usr/bin/env python
# coding=utf-8
import argparse
from ruamel.yaml import YAML
from module.local_crf_main import local_crf_main
from module.manifold_ptd import manifold_ptd_main
from module.manifold_mall_td import manifold_mall_td_main
from module.clip_mall_td import clip_mall_td_main
from module.gauss_mall_td import gauss_mall_td_main
from module.global_crf_main import global_crf_main
from module.fcluster_main import fcluster_main
from module.clip_mall_td import clip_mall_td_main

dict2main={'local_CRF':local_crf_main,'manifold_ptd':manifold_ptd_main,'gauss_mall_td':gauss_mall_td_main,'manifold_mall_td':manifold_mall_td_main,'global_CRF':global_crf_main,'fcluster':fcluster_main,'clip_mall_td':clip_mall_td_main}


def build_main(name):
    return dict2main[name]
    
    
yaml = YAML(typ='safe')
def load_yaml(p):
    with open(p) as infile:
        data=yaml.load(infile)
    return data
parser=argparse.ArgumentParser()
parser.add_argument("--yaml",help="echo the string")
parser.add_argument("--cluster_threshold",help="echo the string")
parser.add_argument("--alpha",help="echo the string")
args=parser.parse_args()

config=load_yaml(args.yaml)
print(config)
main=build_main(config['method'])
main(config)


