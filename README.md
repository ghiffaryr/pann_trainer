# Audio classification finetuned on pretrained audio neural networks (PANNs)

Audio classification is a task to classify audio clips into classes such as jazz, classical, etc.

**1. Requirements** 

python 3.8 + pytorch 1.0

**2. Then simply run:**

$ Run the bash script ./train.sh

Or run the commands in runme.sh line by line. The commands includes:

(1) Modify the paths of dataset and your workspace

(2) Extract features

(3) Train model

## Model
A 14-layer CNN of PANNs is fine-tuned. I use 2-fold holdout cross validation for classification. 

## Reference

[1] Kong, Qiuqiang, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D. Plumbley. "PANNs: Large-scale pretrained audio neural networks for audio pattern recognition." arXiv preprint arXiv:1912.10211 (2019).
