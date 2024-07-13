## Walk Wisely on Graph: Knowledge Graph Reasoning  with Dual Agents via Efficient Guidance-Exploration

![](https://pic.imgdb.cn/item/66824f87d9c307b7e974ec20.png)

### Introduction
This paper presents **LOCOCO**, an efficient guidance-exploration framework built on 
dual-agent KG reasoning methods to enhance the agent's long-distance reasoning 
ability on standard KG and sparse KG. The key insight behind our approach is balancing 
the self-exploration of DWARF with guidance from GIANT. Specifically, on the one hand, 
we leverage the attention mechanism to make DWARF pay attention to the neighbouring 
entities that are close to the query. On the other hand, we propose that dynamic 
path feedback enables GIANT to have better learning efficiency, thus providing 
DWARF with high-quality guidance, making the DWARF to have a favourable global vision 
while having excellent local reasoning ability.


### Installation
To run our code, you must first set up your environment. 
We recommend using virtualenv with pip to set up a virtual environment 
and dependency manager. For our experiments, we use `python3.10`; 
while other versions of python will probably work, they are untested 
and we cannot guarantee the same performance.

To install the various python dependencies (including pytorch).

`pip install -r requirements.txt`

Some of the most important packages

- scipy 1.7.2
- wandb 0.17.0
- torch 2.0.1
- tqdm 4.62.3
- numpy 1.22.4

Now, you should be ready to run experiments!

### Running Experiments

Here we provide a mapping from task name as used in the paper to 
task name used in the code.

- NELL-995: `nell.sh`
- WN18RR: `WN18RR.sh`
- FB15K-237: `fb15k-237.sh`
- PersonBornInLocation: `personborninlocation.sh`
- OrgHeadquarteredInCity: `orgheadquarteredincity.sh`
- AthletePlaysForTeam: `athleteplaysforteam.sh`
- AthletePlaysInLeague: `athleteplaysinleague.sh`
- AthletePlaysSports: `athleteplayssports.sh`
- TeamPlaysSports: `teamplayssports.sh`
- WorksFor: `worksfor.sh`

The hyperparam configs for each experiments are included in the configs directory. 
To start a particular experiment, just do
```
sh run.sh configs/${task}.sh
```

where the `${task}.sh` is the name of the config file. For example,

```
sh run.sh configs/athleteplaysforteam.sh
```

If you want to change the environment, please modify parameters in 
`./configs/${task}.sh`.

### Output
We use [weights and biases](https://wandb.ai) to log our results. If you do not have a 
weights and biases account, we recommend you get one! However, 
if you still do not want to use weights and biases, you can use the 
`--disable-wandb flag`. Then your results will be stored to a CSV file 
in `policies/<project_name>/<env>/<config>.csv`. The metrics used for evaluation 
are Hits@{1,3,5,10}, MRR, MAP, CSS and ESS. Along with this, the code also 
outputs the answers LOCOCO reached in a file.

### Baselines
1. **TransE** (Bordes et al. 2013) https://proceedings.neurips.cc/paper_files/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html
2. **DistMult** (Yang et al. 2015) https://arxiv.org/pdf/1412.6575v3
3. **ComplEx** (Trouillon et al. 2016) https://arxiv.org/pdf/1606.06357
4. **DeepPath** (Xiong, Hoang, and Wang 2017) https://arxiv.org/pdf/1707.06690v2
5. **MINERVA** (Das et al. 2018) https://arxiv.org/pdf/1711.05851
6. **M-Walk** (Shen et al. 2018) https://arxiv.org/pdf/1802.04394
7. **AttnPath** (Wang et al. 2019) https://aclanthology.org/D19-1264.pdf
8. **SQUIRE** (Bai et al. 2022) https://arxiv.org/pdf/2201.06206
9. **LMKE** (Wang et al. 2022) https://arxiv.org/pdf/2206.12617
10. **CURL** (Zhang et al. 2022) https://arxiv.org/pdf/2112.12876
