# Data Processing

The following documentation will help anyone trying to train models from scratch, either from the USPTO reaction database or their own reaction datasets. The processing code and documentation is still a work-in-progress and could use a lot of refactoring to increase clarity and ease of use. Please raise issues (or pull requests) for any suggestions you have or issues you run into to improve this area of the codebase!

### 1. Processing and deduplicating raw reaction data

`01_process_uspto.py` can be used to clean and deduplicate raw reaction data. It is written to take the  [USPTO-Full dataset](https://figshare.com/articles/dataset/Chemical_reactions_from_US_patents_1976-Sep2016_/5104873) on input (but can be adapted to any reaction corpus with atom mapping) and output a text file containing each reaction in frozenset and SMILES form (tab delimited, one reaction per line). Create a folder `/processing/data/` and place `1976_Sep2016_USPTOgrants_smiles.rsmi` (for the USPTO dataset) in the folder before running
```Python
$ python 01_process_uspto.py
```

### 2. Extracting templates and training single-step retrosynthesis model

We recommend using the [ASKCOS template relevance module](https://gitlab.com/mlpds_mit/askcosv2/retro/template_relevance) to handle template extraction and single-step retrosynthesis model training. The codebase provides documentation and scripts to obtain train / test splits, extract templates, and train a model given a set of atom-mapped reactions (i.e. obtained from Step 1). Our [figshare](https://figshare.com/articles/preprint/25956076) also provides the exact filtered train and validation reaction splits we used for our paper along with the extracted templates in `.jsonl` files.

### 3. Filter reactions for forward model training

`02_test_templates_bi.py` and `03_get_fwd_rxns.py` filter multi-component reactions and reactions for which the reversed retro template cannot be applied to the reactants to yield the product, as the forward models are trained on only unimolecular or bimolecular reactions. In `/processing/data/` place `filtered_train.jsonl`, a `jsonl` file that contain elements of format `{"id": _, "rxn_smiles": _, "canon_reaction_smarts": _}` for each reaction in the training split obtained from ASKCOS. Running
```Python
$ python 02_test_templates_bi.py
$ python 03_get_fwd_rxns.py
```
will then yield a file `/processing/data/filtered_fwd_train.jsonl` of the same format as `filtered_train.jsonl`.

### 4. Extract training examples

Ensure the following are in `/processing/data/`:
- `filtered_train.jsonl` (training split retro reactions with templates)
- `filtered_fwd_train.jsonl` (filtered training split forward reaction with templates)
- `val_rxns_with_template.jsonl` (validation split retro reactions with templates)
- `building_blocks.pkl` (pickled dictionary indexed by building block SMILES. For our paper, we use a filtered version of the eMolecules dataset used by [Chen et al. 2020](https://www.dropbox.com/s/ar9cupb18hv96gj/retro_data.zip?e=1&dl=0). Our version can be found in `desp_data.zip` in our [figshare](https://figshare.com/articles/preprint/25956076))

```Python
$ python 04_extract_fwd_training.py
```
outputs the files needed to train the forward template and building block models. Namely:
- `fwd_train_fp.npz`: molecule + target concatenated fingerprints for template predictions
- `fwd_train_labels.npy`: one-hot encoded labels of templates for training forward model
- `fwd_val_fp.npz` + `fwd_val_labels.npy`: validation examples for above
- `fwd_train_fp_bb.npz`: molecule + target + template concatenated fingerprints for BB prediction
- `fwd_train_labels.npy`: 256-dimensional fingerprint of ground truth building blocks
- `fwd_val_fp_bb.npz` + `fwd_val_labels_bb.npy`: validation examples for above

```Python
$ python 05_extract_sd.py
```
outputs the files needed to train the synthetic distance model. Namely:
- `sd_train_fp.npz`: concatenated fingerprints of `m1`, `m2` pairs for learning synthetic distance 
- `sd_train_labels.npy`: ground truth synthetic distances
- `sd_val_fp.npz` + `sd_val_labels.npy`: validation examples for above

```Python
$ python 06_extract_values.py
```
outputs the files needed to train our variant of the Retro* value model (Chen et al. 2020). Namely:
- `ret_train_fp.npz`: fingerprints of molecules for training value model
- `ret_train_labels.npy`: ground truth `V_m` values
- `ret_val_fp.npz` + `ret_val_labels.npy`: validation examples for above

### 5. Train models

Code and scripts for training the forward template and building block models is found in `desp/inference/models/fwd_model/`. Code and scripts for training the synthetic distance and Retro* value models is found in `desp/inference/models/syn_value/`. More detailed instructions coming soon!