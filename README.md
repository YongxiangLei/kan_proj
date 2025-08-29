# KAN for climate temperature forecasting
This is the code repository for the paper entitled "**Tem2-KAN: Data-Driven Temporal Temperature Prediction via an Improved Kolmogorov-Arnold Network**"

## Dependenices
Before using this code, please install the following packages:
```
colorama          0.4.6
contourpy         1.3.1
cycler            0.12.1
fastdtw           0.3.4
filelock          3.13.1
fonttools         4.56.0
fsspec            2024.6.1
graphviz          0.20.3
Jinja2            3.1.4
joblib            1.4.2
kiwisolver        1.4.8
MarkupSafe        2.1.5
matplotlib        3.10.0
mpmath            1.3.0
networkx          3.3
numpy             2.1.2
packaging         24.2
pandas            2.2.3
pillow            11.0.0
pip               25.0.1
pykan             0.2.8
pyparsing         3.2.1
python-dateutil   2.9.0.post0
pytz              2025.1
PyYAML            6.0.2
scikit-learn      1.6.1
scipy             1.15.2
setuptools        70.2.0
six               1.17.0
sympy             1.13.1
threadpoolctl     3.5.0
torch             2.6.0+cu126
torchaudio        2.6.0+cu126
torchvision       0.21.0+cu126
torchviz          0.0.3
tqdm              4.67.1
typing_extensions 4.12.2
tzdata            2025.1
```

## Training Data
The training data is downloaded from online [UK Met Office Climate Portal](https://climatedataportal.metoffice.gov.uk/).

## Train and Plot
please see the specific training parameters in specific python files. The experimental results examples are:
![](predictions_0.png)
![](predictions_1.png)

## Citations

If you use this code in your research or publications, please cite:

Lei Y, Deng B, Wang Z. Tem2-KAN: Data-Driven Temporal Temperature Prediction via an Improved Kolmogorov-Arnold Network[J]. ISA transactions, 2025. (https://www.sciencedirect.com/science/article/abs/pii/S0019057825003635)

```
@article{lei2025tem2,
  title={Tem2-KAN: Data-Driven Temporal Temperature Prediction via an Improved Kolmogorov-Arnold Network},
  author={Lei, Yongxiang and Deng, Bin and Wang, Ziyang},
  journal={ISA transactions},
  year={2025},
  publisher={Elsevier}
}
```
