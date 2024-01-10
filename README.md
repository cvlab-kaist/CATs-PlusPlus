[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cats-boosting-cost-aggregation-with/semantic-correspondence-on-pf-pascal)](https://paperswithcode.com/sota/semantic-correspondence-on-pf-pascal?p=cats-boosting-cost-aggregation-with)
<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cats-boosting-cost-aggregation-with/semantic-correspondence-on-spair-71k)](https://paperswithcode.com/sota/semantic-correspondence-on-spair-71k?p=cats-boosting-cost-aggregation-with)

## CATs++: Boosting Cost Aggregation with Convolutions and Transformers (TPAMI'22)
For more information, check out the paper on [[arXiv](https://arxiv.org/abs/2202.06817)]. Also check out project page here [[Project Page](https://ku-cvlab.github.io/CATs-PlusPlus-Project-Page/)]

# Network

Our model is illustrated below:

![Figure of Architecture](/images/ARCH1.png)
![Figure of Architecture](/images/ARCH2.png)

# Environment Settings
```
git clone https://github.com/KU-CVLAB/CATs-PlusPlus.git
cd CATs-PlusPlus

conda env create -f environment.yml
```

# Evaluation
- Download pre-trained weights on [Link](https://drive.google.com/drive/folders/18i16PWTSqeW7a3bBZa29nsFYmYyo80zz?usp=sharing)
- All datasets are automatically downloaded into directory specified by argument `datapath`

Result on SPair-71k:

      python test.py --pretrained "/path_to_pretrained_model/spair" --benchmark spair

Results on PF-PASCAL:

      python test.py --pretrained "/path_to_pretrained_model/pfpascal" --benchmark pfpascal

Results on PF-WILLOW:

      python test.py --pretrained "/path_to_pretrained_model/pfpascal" --benchmark pfwillow --thres {bbox|bbox-kp}

# Acknowledgement <a name="Acknowledgement"></a>

We borrow code from public projects (huge thanks to all the projects). We mainly borrow code from  [DHPF](https://github.com/juhongm999/dhpf), [GLU-Net](https://github.com/PruneTruong/GLU-Net), and [CATs](https://github.com/SunghwanHong/Cost-Aggregation-transformers). 
### BibTeX
If you find this research useful, please consider citing:
````BibTeX
@article{cho2022cats++,
  title={CATs++: Boosting Cost Aggregation with Convolutions and Transformers},
  author={Cho, Seokju and Hong, Sunghwan and Kim, Seungryong},
  journal={arXiv preprint arXiv:2202.06817},
  year={2022}
}
````
