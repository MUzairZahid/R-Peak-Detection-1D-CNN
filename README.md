# Robust R-Peak Detection in Low-Quality Holter ECGs Using 1D Convolutional Neural Network

This repository includes the implentation of R peak detection method in [Robust R-Peak Detection in Low-Quality Holter ECGs Using 1D Convolutional Neural Network](https://ieeexplore.ieee.org/abstract/document/9451595).

## Network Architecture
![The proposed systematic approach and network architecture](https://user-images.githubusercontent.com/43520052/162128277-43fef402-38c6-4bdd-b198-2ca117a7fae7.png)

## Verification Model
![image](https://user-images.githubusercontent.com/43520052/162134457-fe87131b-ea9c-461c-804c-c1c23c269cc1.png)



## Dataset

- [The China Physiological Signal Challenge 2020](http://2020.icbeb.org/CSPC2020), (CPSC-2020) dataset is used for training & testing.
- R peak annotations are already available in the data folder.


## Run

#### Train
- Download CPSC data from the link to the "data/" folder
- Data Preparation without augmentation
```http
  python prepare_data.py
```
- Data Preparation with augmentation
```http
  python prepare_data_augmentation.py
```
- Start patient wise training and evaluation.
```http
  python run_cnn.py
```



## Citation

If you use the provided method in this repository, please cite the following paper:

```
@article{zahid2021robust,
  title={Robust R-Peak Detection in Low-Quality Holter ECGs Using 1D Convolutional Neural Network},
  author={Zahid, Muhammad Uzair and Kiranyaz, Serkan and Ince, Turker and Devecioglu, Ozer Can and Chowdhury, Muhammad EH and Khandakar, Amith and Tahir, Anas and Gabbouj, Moncef},
  journal={IEEE Transactions on Biomedical Engineering},
  volume={69},
  number={1},
  pages={119--128},
  year={2021},
  publisher={IEEE}
}
```
