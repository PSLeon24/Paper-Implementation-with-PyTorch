# Paper-Implementation-with-PyTorch
Model Implementation with PyTorch in paper

## 1. SRCNN
- Paper title: Image Super-Resolution Using Deep Convolutional Networks
- Category: Image Super-Resolution
- Paper URL: https://arxiv.org/abs/1501.00092
- Abstract
  - This paper presents a method for single-image super-resolution (SR) using deep learning, specifically through Deep Convolutional Neural Networks (CNNs). The goal is to enhance the resolution of a low-resolution (LR) image to a high-resolution (HR) image, preserving the image's quality and details.
- Implemented code URL: https://github.com/PSLeon24/Paper-Implementation-with-PyTorch/blob/main/SRCNN/SRCNN.ipynb

## 2. AlexNet
- Paper title: ImageNet Classification with Deep Convolutional Neural Networks
- Category: Image Classification
- Paper URL: https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
- Abstract
  - This paper introduces AlexNet, a deep convolutional neural network that achieved breakthrough results in the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2012. The model significantly improved the accuracy of image classification tasks by employing a deeper and wider architecture compared to previous methods. Key innovations include the use of Rectified Linear Units (ReLU) for nonlinearity, dropout for regularization, and data augmentation techniques. AlexNet's success demonstrated the effectiveness of deep learning in large-scale image recognition tasks, paving the way for further advancements in the field.
- Implemented code URL: https://github.com/PSLeon24/Paper-Implementation-with-PyTorch/blob/main/AlexNet/CNN%20-%20AlexNet.ipynb
- Core keyword: fundamental CNN architecture, filter(=kernel), stride, padding, maxpooling, convolution layer, fully connected layer

## 3. ResNet
- Paper title: Deep Residual Learning for Image Recognition
- Category: Image Classification
- Paper URL: https://arxiv.org/pdf/1512.03385
- Abstract
  - This paper introduces ResNet (Residual Networks), a deep convolutional neural network that addresses the problem of training very deep networks by using a novel architecture called "residual learning." The key idea is to add shortcut connections (or skip connections) that bypass one or more layers, allowing the network to learn residual functions with reference to the layer inputs. This significantly eases the training of deep networks and allows the construction of networks with hundreds or even thousands of layers, achieving state-of-the-art performance on image classification tasks. ResNet won the 1st place in several tracks of the ILSVRC 2015 competition and has become a foundational model in the field of deep learning.
- Implemented code URL: https://github.com/PSLeon24/Paper-Implementation-with-PyTorch/blob/main/ResNet/ResNet.ipynb
- Core keyword: residual connection(skip connection)

## 4. ViT
- Paper title: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
- Category: Image Classification
- Paper URL: https://arxiv.org/pdf/2010.11929
- Abstract
  - This paper introduces the Vision Transformer (ViT), a model that applies the transformer architecture, originally designed for natural language processing tasks, to image classification. The core idea is to split an image into a sequence of fixed-size patches, treat these patches as tokens similar to words in NLP, and process them with a standard transformer encoder. The ViT model achieves state-of-the-art performance on multiple image recognition benchmarks when trained on large-scale datasets, demonstrating that transformers can be highly effective in computer vision tasks when provided with sufficient data. The paper highlights the scalability of the transformer architecture and its potential to replace convolutional neural networks (CNNs) in various vision applications.
- Implemented code URL: https://github.com/PSLeon24/Paper-Implementation-with-PyTorch/blob/main/ViT/ViT.ipynb
- Core keyword: patch
  
## 5. GoogLeNet
- Paper title: Going deeper with convolutions
- Category: Image Classification
- Paper URL: [https://](https://arxiv.org/pdf/1409.4842)
- Abstract
  - To be added.
- Implemented code URL: https://github.com/PSLeon24/Paper-Implementation-with-PyTorch/blob/main/GoogLeNet/GoogLeNet.ipynb
- Core keyword: inception module, 1x1 convolution(to reduce dimensions)

## 6. FGSM Attack
- Paper title: Explaining and Harnessing Adversarial Examples
- Category: Adversarial Example
- Paper URL: https://arxiv.org/pdf/1412.6572
- Abstract
  - This paper presents the Fast Gradient Sign Method (FGSM), a simple yet effective technique for creating adversarial examples that can fool neural networks. By slightly altering the input data in the direction that increases the model's error, FGSM highlights how even minor perturbations can cause significant misclassification, exposing vulnerabilities in the robustness of neural networks. The method provides key insights into the security challenges faced by machine learning models.
- Implemented code URL: To be added.
