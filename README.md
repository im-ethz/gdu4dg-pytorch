# Gated Domain Units (PyTorch)


[![DOI](https://img.shields.io/badge/DOI-2206.12444/arxiv.org/abs/2206.12444-red.svg)](https://arxiv.org/abs/2206.12444)
![GDU_logo](https://user-images.githubusercontent.com/73110207/185412013-40309db3-dc3d-4f32-bf65-6c00e87d28a7.png)

Gated Domain Units (GDU) aim to make your deep learning models robust against distribution shifts when applied in the real world. To make the GDus simple and easily applicable, we integrated our GDU into a modular layer. Hence, our GDUs can be applied to your deep learning models by simply replacing the task-specific head of your model with our layer. For example, you can use a pre-trained ResNet-50 as the feature extractor, and instead of a classification head, you attach our layer that performs the same classification task. TensorFlow? ➡️ [here](https://github.com/im-ethz/pub-gdu4dg).

🚀 User-friendly. Simple. Powerful.

In our [paper](https://arxiv.org/abs/2206.12444), we postulate that real-world distributions are composed of elementary distributions that remain invariant across different domains. We call this the **invariant elementary distribution (I.E.D.)** assumption. This invariance thus enables knowledge transfer to unseen domains. To exploit this assumption in domain generalization (DG), we developed a modular neural network layer (the DGLayer) that consists of **Gated Domain Units (GDUs)**. Because our layer is trained with backpropagation, it can be easily integrated into existing deep learning frameworks (see our example below).

## Introducing Gated Domain Units

Before we introduce our Gated Domain Units (GDUs), we will briefly present our theoretical idea **invariant elementary distribution (I.E.D.)** assumption from a practical point of view. 

### What are invariant elementary distributions?

We postulate the elementary domain bases are the invariant subspaces that allow us to generalize to unseen domains (see our [paper](https://arxiv.org/abs/2206.12444) for details). Consider the practical case of classifying the outcome of virus infections based on electronic health records collected from multiple sources such as patients, cohorts, and medical centers. Naturally, several factors determining the trajectory such as gender, pre-existing diseases, and virus mutations can change simultaneously across these sources. While, to a certain degree, these common factors remain invariant across individuals, the contribution of each of these factors may differ between individuals. In terms of the assumptions made in our work, we model each of these factors with a corresponding elementary distribution $\mathbb{P}_{i}$. For a previously unseen individual we can then determine the coefficients $\alpha_i^s$ and therewith quantify the contribution of each factor.

### How do we eploit the idea of elementary distributions in practice?

To exploit this I.E.D. assumption, we developed the Gatd Domain Units. Each GDU learns an embedding of an individual elementary domain that allows us to encode the domain similarities during the training. During inference, the GDUs compute similarities between an observation and each of the corresponding elementary distributions which are then used to form a weighted ensemble of learning machines (see Figure 1).

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/73110207/179177894-0528920c-1063-4834-ab3f-852a0ab2d156.png">
  <br>
    <em>Figure 1 Visualization of the DG layer (left panel). The DG layer layer consists of several GDUs that represent the elementary distributions. During training, these GDUs learn the elementary domain bases <MATH> V_{1} , ... , V_{M} </MATH> that approximate these distributions.</em>
</p>

### How can I transfer the I.E.D. assumption and GDUs to my problem setting?

### How can I explain and interpret what the GDUs have learned?

## Get started: Integrate Gated Domain Units in your deepl elarning models

### Install

Currently we do not support installation via pip or conda. To install and use our DGLayer in the meantime, please clone this repository.

### Example: Extend existing deep learning Models with Gated Domain Units

### Set the parameters

## References

This repository uses code from the following references. Please see their licence statements.

- [WILDS experiments](https://github.com/p-lambda/wilds) Koh PW, Sagawa S, Marklund H, et al. WILDS: A Benchmark of in-the-Wild Distribution Shifts. In: Meila M, Zhang T, eds. Proceedings of the 38th International Conference on Machine Learning. Vol 139. PMLR; 2021:5637--5664. https://proceedings.mlr.press/v139/koh21a.html

  MIT License
  
  Copyright (c) 2020 WILDS team
  
  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included   in all copies or substantial portions of the Software.

- [DOMAINBED experiments](https://github.com/facebookresearch/DomainBed) Gulrajani I, Lopez-Paz D. In Search of Lost Domain Generalization. arXiv:200701434 [cs, stat]. Published online July 2, 2020. http://arxiv.org/abs/2007.01434

  The MIT License
  
  Copyright (c) Facebook, Inc. and its affiliates.
  
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files   (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify,       merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is          furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  


# Citation
Original article: [Gated Domain Units for Multi-source Domain Generalization](https://doi.org/10.48550/arxiv.2206.12444)

```
@article{https://doi.org/10.48550/arxiv.2206.12444,
  doi = {10.48550/ARXIV.2206.12444},
  url = {https://arxiv.org/abs/2206.12444},
  author = {Föll, Simon and Dubatovka, Alina and Ernst, Eugen and Maritsch, Martin and Okanovic, Patrik and Thäter, Gudrun and Buhmann, Joachim M. and Wortmann, Felix and Muandet, Krikamol},
  title = {Gated Domain Units for Multi-source Domain Generalization},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
