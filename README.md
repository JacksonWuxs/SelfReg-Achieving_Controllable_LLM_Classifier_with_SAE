# SelfReg: Regularizing LLM-based Classifiers on Unintended Features

### Introduction

This is the official implementation of the paper [Self-Regularization with Sparse Autoencoders for Controllable LLM-based Classification](https://arxiv.org/abs/2502.14133) (accepted by KDD 2025). In the paper, we introduced a novel regularization strategy to regularize the use of "unintended features" for LLM-based text classifiers, where the unintended features can be sensitive attributes for privacy/fairness purposes or shortcut patterns for generlizability. We evaluate our proposed method on three real world datasets, namely "ToxicChat",  "RewardBench", and "Dxy". We consider `Mistral-7B-inst-v0.2` as our backbone LLM and we pre-train our SAEs for it with 113 million tokens over 5 epochs. 

### Environement

We assume that you manage the environment with Conda library.

```shell
>>> conda create -n SelfReg python=3.9 -y
>>> conda activate SelfReg
>>> pip install -U requirements.txt
```

### Pre-training Sparse Autoencoders

Most of the materials to pre-train SAEs are located in the folder ``src/PretrainSAE``, in which you will see the TopK SAE implemented in ``autoencoders.py`` and the hooks in ``llm_surgery.py`` to mount/dismount SAEs on the LLMs. To pre-train SAEs, you shall first download the source dataset from https://drive.google.com/drive/folders/11uBH-2R03ctePervS9fdy3O7ZU5HDaLt?usp=sharing and then put it in the ``datasets/prompt_dataset_train.tsv``. To pre-train the SAEs, you shall run the following commands. 

```shell
>>> cd src/PretrainSAE

# Step-1: Collect hidden representations from LLMs on GPU0
>>> nohup python -u create_actvs.py 0 1 > logs/collect_actvs.log & 
# If you have multiple GPUs (e.g., 4 GPUs), you can collect representations with multiple GPUs as
>>> nohup python -u create_actvs.py 0 4 > logs/collect_actvs_gpu0.log & 
...
>>> nohup python -u create_actvs.py 3 4 > logs/collect_actvs_gpu1.log & 

# Step-2: Train SAE on a certain layer (e.g., 16th-layer) on GPU0
>>> nohup python -u train_SAEs_topK7.py 0 16 > logs/pretrain_layer16.log &
```

With 1 Nvidia-A6000 GPU, Step-1 requires around 8 hours, and Step-2 needs about 20 hours. The pre-trained model weight will be saved at ``outputs/TopK7_l16_h65k_epoch5.pth``. For your convinience, you may want to download the pre-trained weight from Huggingface: [wuxs/Mistral_TopK_SAE_l16](https://huggingface.co/wuxs/Mistral_TopK_SAE_l16).

### Fine-tuning Sparse Autoencoders

All the material to fine-tune SAEs is located in the folder ``src/FinetuneSAEs``. You can run with the following comands.

```bash
>>> cd src/FinetuneSAE

# Step-1: Collect hidden representations from LLMs for three datasets on GPU0:
>>> nohup python -u create_actvs_TDorRM.py 0 1 TD train > logs/collect_actvs_TD.log & 
>>> nohup python -u create_actvs_TDorRM.py 0 1 RM train > logs/collect_actvs_RM.log &  
>>> nohup python -u create_actvs_DD.py 0 1 DD train > logs/collect_actvs_DD.log & 

# Step-2: Fine-tune SAEs on three datasets on GPU0
>>> nohup python -u finetune_SAE.py 0 ../PretrainSAE/outputs/TopK7_l16_h65k_epoch5.pth TD 378 5 > logs/finetune_TD.log &
>>> nohup python -u finetune_SAE.py 0 ../PretrainSAE/outputs/TopK7_l16_h65k_epoch5.pth RM 149 5 > logs/finetune_RM.log &
>>> nohup python -u finetune_SAE.py 0 ../PretrainSAE/outputs/TopK7_l16_h65k_epoch5.pth DD 59 40 > logs/finetune_DD.log &
# Note: 378/149/59 are the numbers of tokens for the task templates. The templates are the same for entries from the same dataset.
```

The fine-tuned SAE weights will be save at ``outputs/TopK7_l16_h65k_FT_epochX_X.pth``. You may want to download our fine-tuned SAEs from Huggingface: [wuxs/Mistral_TopK_SAE_l16_FT_ToxicDetect](https://huggingface.co/wuxs/Mistral_TopK_SAE_l16_FT_ToxicDetect), [wuxs/Mistral_TopK_SAE_l16_FT_RewardModeling](https://huggingface.co/wuxs/Mistral_TopK_SAE_l16_FT_RewardModeling), and [wuxs/Mistral_TopK_SAE_l16_FT_DiseaseDiagnosis](https://huggingface.co/wuxs/Mistral_TopK_SAE_l16_FT_DiseaseDiagnosis), respectively.

### Self-Regulariztion Training 

For each dataset, we conduct experiments on independent folders, i.e., ``src/ToxicDetect/``, ``src/RewardModeling``, and ``src/DiseaseDiagnosis/``. Since the structures and pipelines are pretty similar across the datasets, we only provide 1 examplar code as following.

```bash
>>> cd src/ToxicDetect

# Step-1: Collect hidden representations from LLMs for TD dataset on GPU0
>>> nohup python -u create_actvs_TD.py 0 1 train > logs/collect_actvs_train.log &
>>> nohup python -u create_actvs_TD.py 0 1 valid > logs/collect_actvs_valid.log &
>>> nohup python -u create_actvs_TD.py 0 1 test > logs/collect_actvs_test.log &

# Step-2.1: Interpreting Fine-tuned SAE by collecting their Top-Activated features
>>> nohup python -u collect_activated_spans.py 0 ../FinetuneSAE/outputs/TopK7_l16_h65k_FT_epoch5_TD.pth 0 1 > logs/collect_textspans.log & # Note that, this script supports multi-processing running by set up GPU-idx 0 to others, and the total group 1 to the total number of GPUs you can use. 
>>> cp ./outputs/textspans_TopK7_l16_h65k_FT_epoch5_TD_TD/text_spans_group0.tsv ./outputs/textspans_TopK7_l16_h65k_FT_epoch5_TD_TD/full.tsv #You just need to concatenate all the subfiles to ``full.tsv`` will be fine.

# Step-2.2: Interpreting Fine-tuned SAE by summarizing their Top-Activated features
>>> cd annotations
>>> python -u groupby_textspans.py ../outputs/textspans_TopK7_l16_h65k_FT_epoch5_TD_TD/textspans_group0.tsv 
>>> vim annotate_explanations.py # setup your OpenAI key to use GPT-4o for fast explanation. It spends only a few dollars.
>>> python -u annotate_explanations.py ./TopAct_TopK7_l16_h65k_FT_epoch5_TD.tsv

# Step-2.3: Judging the learned features according to their summarizations.
>>> vim annotate_harmfulness.py # setup your OpenAI key to use GPT-4o for fast judging. It spends only a few dollars.
>>> python -u annotate_harmfulness.py ./TopAct_TopK7_l16_h65k_FT_epoch5_TD_explained.tsv
>>> cd ..

# Step-3: Training Self-Regularized classifiers!!!!
>>> sh run_train_clf_seeds.sh SelfReg  # it will automatically run in 6 random seeds, each of which will have their own logging file
>>> python report_results.py ./logs/train_SelfReg/  # this files will compute the average accuracy and F1 scores across the random seeds.The results from here is the numbers we reported in our paper
```

In our experiments, we call GPT4o-mini to scale up the automatic judgement process. It will totally cost less than 10 dollars for summarizing and judging the relevance to the task in 1 hour. In this repo, __we have shared our collected explanations for your reference__. That means, you can skip Step 2.1-2.3 to reproduce our results.

To reproduce results of __baselines__ (e.g., ``l2norm``), you can simply run ``sh run_train_clf_seeds.sh l2norm``, it will run the baseline you specified. You can check all 7 baselines and our SelfReg by running ``ls | grep 'train_*'``. All models share the same training configuration as outlined in ``pipelines.py``. 
