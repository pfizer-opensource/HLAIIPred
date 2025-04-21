# HLAIIPred
Cross-Attention Mechanism for Modeling the Interaction of HLA Class II Molecules with Peptides.
Preprint: https://www.biorxiv.org/content/10.1101/2024.10.01.616078v1

## Abstract 
We introduce HLAIIPred, a deep learning model to predict peptides presented by class II human leukocyte antigens (HLAII) on the surface of antigen presenting cells. HLAIIPred is trained using a Transformer-based neural network and a dataset comprising of HLAII-presented peptides identified by mass spectrometry. In addition to predicting peptide presentation, the model can also provide important insights into peptide-HLAII interactions by identifying core peptide residues that form such interactions. We evaluate the performance of HLAIIPred on three different tasks, peptide presentation in monoallelic samples, immunogenicity prediction of therapeutic antibodies, and neoantigen prioritization for cancer immunotherapy. Additionally, we created a new dataset of biotherapeutics HLAII peptides presented by human dendritic cells. This data is used to develop screening strategies to predict the unwanted immunogenic segments of therapeutic antibodies by HLAII presentation models. HLAIIPred demonstrates superior or equivalent performance when compared to the latest models across all evaluated benchmark datasets. We achieve a 16% increase in prediction of presented peptides compared to the second-best model on a set of unseen peptides presented by less frequent alleles. The model also improves the area under the precision-recall curve by 3% for distinguishing between immunogenic and non-immunogenic antibodies. We show that HLAIIPred can identify epitopes in therapeutic antibodies and prioritize neoantigens with high accuracy.



# Setup Instructions
Follow the steps below to install the necessary dependencies and set up the environment:

## Preferred Setup Method:

```bash
# Step 1: Create and activate a conda environment
conda create --name hlapred python=3.11
conda activate hlapred

# Step 2: Install PyTorch and related libraries
## please consult with the pytorch.org to select your preferences.
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Step 3: Install additional dependencies
mamba install -c conda-forge scipy numpy pandas tqdm yaml biopython
pip install pyyaml

# Step 4: Clone the repository and install the package (in the hlapred environment)
## Clone the repository
git clone <repository_url>
## Navigate to the repository
cd HLAIIPred
## Install the package
pip install -e .
```

## Alternative Setup Method:

```bash
# Step 1: Clone the repository and set up the environment
## Clone the repository
git clone <repository_url>
## Navigate to the repository
cd HLAIIPred
## Create a conda environment with all dependencies
conda env create -n hlapred -f requirements.yml

# Step 2: Install the package in the hlapred environment
conda activate hlapred
pip install -e .
```

# Example Usage

## Using a Python Script

Activate the `hlapred` environment and run the `predict_peptide_example.py` script:

```bash
python predict_peptide_example.py
```

## Command-Line Interface

Coming soon.


# Citation

If you use this tool in your research, please cite the following preprint:

[https://www.biorxiv.org/content/10.1101/2024.10.01.616078v1](https://www.biorxiv.org/content/10.1101/2024.10.01.616078v1)
