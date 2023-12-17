# KG-TREAT: Pre-training for Treatment Effect Estimation by Synergizing Patient Data with Knowledge Graphs

## Introduction
Code for paper "KG-TREAT: Pre-training for Treatment Effect Estimation by Synergizing Patient Data with Knowledge Graphs".

In this paper, we introduce a novel pre-training and fine-tuning framework, KG-TREAT, which synergizes large-scale observational patient data with biomedical knowledge graphs (KGs) to enhance TEE. Unlike previous approaches, KG-TREAT constructs *dual-focus* KGs and integrates a deep bi-level attention synergy method for in-depth information fusion, enabling distinct encoding of treatment-covariate and outcome-covariate relationships. KG-TREAT also incorporates two pre-training tasks to ensure a thorough grounding and contextualization of patient data and KGs.

<img src="framework.png" title="The overall framework of KG-TREAT.">

We obtain and preprocess 3M large-scale observational patient data from [MarketScan Research
Databases](https://www.ibm.com/products/marketscan-research-databases), and construct KGs from [Unified Medical Language System (UMLS)](https://www.nlm.nih.gov/research/umls/index.html) with 300K nodes and 1M edges as our pre-training data. We derive 4 downstream TEE tasks (10-20K patient samples) for evaluating the comparative treatment effectiveness in reducing stroke and myocardial infarction risk for patients with coronary artery disease (CAD).

## Requirements
Create a virtual environment and install the required dependencies by running the commands:
```bash
conda create -n kgtreat python=3.7
conda activate kgtreat
# check CUDA version via "nvcc --version"
pip install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==4.17.0 datasets==2.0.0 wandb
pip install torch-scatter torch-sparse torch-geometric torch-cluster -f https://pytorch-geometric.com/whl/torch-1.10.1+cu111.html
pip install scikit-learn==1.0.2 numpy==1.21.6 tqdm==4.64.1
```

## Pre-train KG-TREAT

```bash
python -m torch.distributed.launch --nproc_per_node=4 train.py 
    --fp16 
    --data_path data/cad/pretrain 
    --vocab_file data/cad/vocab.txt 
    --do_train 
    --num_train_epochs 2 
    --warmup_steps 20000 
    --learning_rate 1e-4 
    --overwrite_output_dir 
    --output_dir output/mcp_lp_v_g_bertbase 
    --use_kg 
    --mask_prediction 
    --link_prediction 
    --sep_graph 
    --per_device_train_batch_size 7 
    --validation_split_percentage 1 
    --logging_steps 100 
    --save_steps 40000 
    --max_seq_length 256 
    --max_node_num 200 
    --baseline_window 360 
    --cache_dir cache/ 
    --time_embedding 
    --ent_emb_paths data/umls/ent_emb_blbertL.npy 
```

## Fine-tune KG-TREAT for treatment effect estimation
```bash
python -m torch.distributed.launch --nproc_per_node=4 train.py
    --model_name_or_path output/mcp_lp_v_g_bertbase 
    --data_path data/cad/downstream 
    --target_drug Rivaroxaban.json 
    --compared_drug Aspirin.json 
    --vocab_file data/cad/vocab.txt 
    --do_train 
    --do_eval 
    --num_train_epoch 2 
    --learning_rate 5e-5 
    --overwrite_output_dir 
    --output_dir output/mcp_lp_v_g_bertbase_finetuned_outcome_prediction 
    --use_kg 
    --sep_graph
    --outcome_prediction 
    --per_device_train_batch_size 8 
    --validation_split_percentage 10 
    --cache_dir cache/ 
    --logging_steps 50 
    --max_seq_length 256 
    --max_node_num 200 
    --baseline_window 360 
    --overwrite_cache 
    --time_embedding 
```
