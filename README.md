### Tips ###
This code includes our proposed FULORA. However, we need some time to adjust the preprocessing files to show the ablation experiments in our appendix. It will be continuously updated. 

## Walk Wisely on Graph: Knowledge Graph Reasoning  with Dual Agents via Efficient Guidance-Exploration

![](https://pic.superbed.cc/item/66bed631fcada11d3758f03d.png)

### Introduction
This paper presents **FULORA**, an efficient guidance-exploration framework built on 
dual-agent KG reasoning methods to enhance the agent's long-distance reasoning 
ability on standard KG and sparse KG. The key insight behind our approach is balancing 
the self-exploration of DWARF with guidance from GIANT. Specifically, on the one hand, 
we leverage the attention mechanism to make DWARF pay attention to the neighbouring 
entities that are close to the query. On the other hand, we propose that dynamic 
path feedback enables GIANT to have better learning efficiency, thus providing 
DWARF with high-quality guidance, making the DWARF to have a favourable global vision 
while having excellent local reasoning ability.


### Installation
For our experiments, we use `python3.10`; 
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

### Experiments Setup

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

### Output
We use [weights and biases](https://wandb.ai) to log our results. If you do not have a 
weights and biases account, we recommend you get one! However, 
if you still do not want to use weights and biases, you can use the 
`--disable-wandb flag`. Then your results will be stored to a CSV file 
in `policies/<project_name>/<env>/<config>.csv`. The metrics used for evaluation 
are Hits@{1,3,5,10}, MRR, MAP, CSS and ESS. Along with this, the code also 
outputs the answers FULORA reached in a file.

### Acknowledgements
Thanks to the KG reasoning codes proposed in other papers, already cited in our forthcoming paper. We refer the agent training code of [DKGR](https://github.com/RutgersDM/DKGR)
