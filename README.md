<img align="center" src="DESP.png" width="350px" />

**Double-Ended Synthesis Planning with Goal-Constrained Bidirectional Search**\
_Kevin Yu, Jihye Roh, Ziang Li, Wenhao Gao, Runzhong Wang, Connor W. Coley_

This repo contains code for DESP (Double-Ended Synthesis Planning), which applies goal-constrained bidirectional search to computer-aided synthesis planning. DESP is designed to propose a synthesis plan towards a given target molecule under the user-specified constraint of using one or more specific starting materials.

## Quick Start

To reproduce our experimental results or to try DESP with our pretrained models, perform the following steps after cloning this repository.

#### 1. Environment Setup

DESP requires a GPU to run at a practical speed. Ensure that the `pytorch-cuda` dependency is compatible with the version of CUDA on your machine. To check, run the following command and look for the `CUDA Version`.
```bash
$ nvidia-smi
```

Now, create the `desp` conda environment from the project directory:
```bash
$ conda env create -f environment.yml
```

#### 2. Data and model installation

Download the pre-trained model weights [at this link](https://figshare.com/articles/preprint/25956076). Unzip the contents of `desp_data.zip` into `desp/data/`. 

#### 3. Run experiments

To reproduce the experiments, navigate to the directory `desp/experiments/` and run the evaluation script. The first argument refers the benchmark set to use, while the second argument refers to the method to use. The results of the experiments will be saved in `desp/experiments/<benchmark_set>/<method>.txt`, along with a corresponding `.pkl` file containing the full search graphs of each search.
```bash
$ sh evaluate.sh [pistachio_reachable|pistachio_hard|uspto_190] [f2e|f2f|retro|retro_sd|random|bfs]
```
A GPU is required for DESP-F2E or DESP-F2F. Specify the device in the evaluation script and ensure that your GPU has enough memory to load the building block index (around 3 GB). Additional memory is required for DESP-F2F due to batch inference of the synthetic distance predictions. The forward prediction module takes a few minutes to initialize as it loads the index into memory.

#### 4. Run DESP on your own targets and starting materials

To run DESP on your own specified targets and starting materials, navigate to the `desp/` directory. In a Python environment (IPython notebook, Python shell, Python script), initialize and invoke DESP with default parameters as follows:
```Python
from DESP import DESP

desp = DESP(strategy='f2e')                 # switch to 'f2f' if you want to try F2F
result, route = desp.search(
    'COC(=O)CC12CCC(c3ccc(Br)cc3)(CC1)CO2', # Target SMILES
    ['COC(=O)C1(c2ccc(Br)cc2)CCC(=O)CC1']   # List of starting materials SMILES
)
```
If DESP is able to find a route for the given inputs, the route can be visualized by running:
```Python
desp.visualize_route(route, 'route')
```
This will save a DOT file `desp/route` and image file `desp/route.png` to the directory which visualizes the solved route. To view the image directly in a IPython notebook, you can run, for example:
```Python
from IPython.display import Image

Image("route.png", width=300)
```

## Processing and Training from Scratch

See the guide at `/processing/README.md`.