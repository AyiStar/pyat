# PyAT: Python based Adaptive Testing Toolkit

This repo contains:

* An abstraction of the pipeline of Adaptive Testing, which is a standard educational test form that provides personalized items with a cognitive diagnosis model and an algorithmic item selection strategy.
* A bundle of cognitive diagnosis models and item selection strategies implemented in Python.

The repo also serves as released code of our work on Adaptive Testing, public on ICDM'20[1] and AAAI'23[2], where:
* In [1], we proposed an active learning based item selection strategy named MAAT (Model-Agnostic Adaptive Testing) in order to seperate item selection strategy design from specific cognitive diagnosis model details.
* In [2], we proposed a Bayesian meta-learning based cognitive diagnosis framework named BETA-CD (Bayesian mETA-learned Cognitive Diagnosis) to generally address the user cold-start problem for cognitive diagnosis models.

The docs, as well as some part of code (such as examples, tests, etc.), is being actively further completed.

[1] Haoyang Bi, Haiping Ma, Zhenya Huang, Yu Yin, Qi Liu, Enhong Chen, Yu Su, and Shijin Wang, Quality meets Diversity: A Model-Agnostic Framework for Computerized Adaptive Testing , The 20th IEEE International Conference on Data Mining (ICDM'2020) , Sorrento, Italy, November 17-20 2020.

[2] Haoyang Bi, Enhong Chen*, Weidong He, Han Wu, Weihao Zhao, Shijin Wang, Jinze Wu. BETA-CD: A Bayesian Meta-learned COgnitive Diagnosis Framework for Personalized Learning . The 37th AAAI Conference on Artificial Intelligence (AAAI'2023), accepted, 2023.
