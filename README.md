# MobileNet

Based on: https://arxiv.org/abs/1704.04861

TODO
---

 - [x] Depthwise seperable convolution
 - [ ] Model given in paper
 - [ ] Misc
   - [x] Test on MNIST
   - [ ] Save MNIST model
   - [x] Test on CIFAR
   - [x] Save and deploy CIFAR model
   - [ ] Test with default SeperableConv2D layer

How to train
---
Example 'CIFAR 10'

```
python train_cifar.py
```
After training,
```
streamlit run cifar_streamlit.py
```