[![python >=3.8.0](https://img.shields.io/badge/python-3.8.0-brightgreen)](https://www.python.org/)      
# Spatialign: a batch alignment method for spatial transcriptomics via spatial embedding and unsupervised cross-domain adaptation contrastive learning                         
The integration of multiple spatially resolved transcriptomics (SRT) datasets can enhance the statistical power to investigate biological phenomena. However, batch effects can lead to irregular data distribution between sections, potentially compromising the reliability of downstream analyses. While various data integration methods have been developed, most are designed for scRNA-seq datasets without considering spatial context. Therefore, we propose Spatialign, an unsupervised cross-domain adaptation method that utilizes contrastive learning and spatial embedding to align latent representations and denoise gene expression profiles. We perform benchmarking analyses on four publicly available SRT datasets, demonstrating the superior performance of Spatialign compared to state-of-the-art methods. Furthermore, Spatialign is shown to be applicable to SRT datasets from diverse platforms. Overall, our results highlight the potential of Spatialign to improve the reliability of downstream analyses in spatially resolved transcriptomics studies.       
            
# Dependences       
[![anndata-0.8.0](https://img.shields.io/badge/anndata-0.8.0-red)](https://pypi.org/project/anndata/#history)
[![scanpy-1.8.2](https://img.shields.io/badge/scanpy-1.8.2-lightgrey)](https://pypi.org/project/scanpy/)
[![torch-1.10.0](https://img.shields.io/badge/torch-1.10.0-brightgreen)](https://pytorch.org/get-started/previous-versions/)
[![torch_geometric-2.0.2](https://img.shields.io/badge/torch_geometric-2.0.2-yellow)](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
[![torch_cluster-1.5.9](https://img.shields.io/badge/torch_cluster-1.5.9-green)](https://data.pyg.org/whl/torch-1.10.0%2Bcu113.html)
[![torch_scatter-2.0.9](https://img.shields.io/badge/torch_scatter-2.0.9-informational)](https://data.pyg.org/whl/torch-1.10.0%2Bcu113.html)
[![torch_sparse-0.6.12](https://img.shields.io/badge/torch_sparse-0.6.12-9cf)](https://data.pyg.org/whl/torch-1.10.0%2Bcu113.html)          
        
# Publicly available datasets            
- Stereo-seq Datasets: mouse olfactory bulb dataset has been deposited into CNGB Sequence Archive (CNSA) of China National GeneBank DataBase (CNGBdb) with accession number CNP001543, and the spatiotemporal dataset of mouse embryonic brain is available at https://db.cngb.org/stomics/mosta.          
- 10x Genomics Visium Dataset: (mouse olfactory bulb) https://www.10xgenomics.com/resources/datasets/adult-mouse-olfactory-bulb-1-standard. And (DLPFC datasets): https://zenodo.org/record/6925603#.YuM5WXZBwuU  
- Slide-seq Datasets: (mouse hippocampus datasets) https://singlecell.broadinstitute.org/single_cell/study/SCP815/highly-sensitive-spatial-transcriptomics-at-near-cellular-resolution-with-slide-seqv2#study-summary, https://singlecell.broadinstitute.org/single_cell/study/SCP354/slide-seq-study#study-summary, and https://singlecell.broadinstitute.org/single_cell/study/SCP948/robust-decomposition-of-cell-type-mixtures-in-spatial-transcriptomics#study-summary, respectively.

# Install     
- [Quick Start](https://spatialign-tutorials.readthedocs.io/en/latest/index.html)       
Downloading the package from https://github.com/zhangchao162/Spatialign/tree/release/wheel             
```python
pip install Spatialign
```     
or      
```git
git clone -b release https://github.com/zhangchao162/Spatialign.git   
cd spatialign
python setup.py install
```

        
# Usage         
- [Quick Start](https://spatialign-tutorials.readthedocs.io/en/latest/index.html)       
```python
from spatialign import Spatialign


data_lists = $DATA_PATH  # dataset list
model = Spatialign(*data_lists,
                   min_genes=20,
                   min_cells=20,
                   batch_key='batch',
                   is_norm_log=True,
                   is_scale=False,
                   is_hvg=False,
                   is_reduce=False,
                   n_pcs=100,
                   n_hvg=2000,
                   n_neigh=15,
                   is_undirected=True
                   latent_dims=100,
                   gpu=0,
                   save_path='./output')

model.train(tau1=0.05, tau2=0.01, tau3=0.1)  # training model
model.alignment()  # remove batch effects and align datasets distibution
```
#### ***Note: For more formal parameter descriptions, see the comments of corresponding functions.***           
        
# Disclaimer        
***This is not an official product.***       
        
         
         