# bert-distillation-backdoor

This repository includes Jupyter Notebooks to implement and optimize a backdoor attack on Large Language Models (BERT) inspired by ADBA for images. The goal is to train and optimize three loss functions simultaneously to identify an effective trigger in the original (teacher) model that will persist and remain functional in the distilled (student) version of the model after knowledge distillation. The method ensures the backdoor transferability while maintaining robustness and stealthiness.

This work is inspired by the paper **"Anti-distillation backdoor attacks: Backdoors can really survive in knowledge distillation"** by Ge et al. (2021), which demonstrated that backdoors can persist through the knowledge distillation process. Our approach builds upon their findings, which originally focused on backdoor attacks in images, but adapting such methods to text presents **unique challenges**. Unlike images, where **pixel perturbations** can be imperceptible, text-based backdoor attacks must modify **discrete tokens**, making stealthy trigger insertion significantly harder without altering fluency or meaning. 

To address this, we developed a **trigger optimization method specifically for text**, ensuring that the backdoor remains **hidden while preserving linguistic coherence** in the distilled (student) model.
### Citation
```bibtex
@inproceedings{ge2021anti,
  title={Anti-distillation backdoor attacks: Backdoors can really survive in knowledge distillation},
  author={Ge, Yunjie and Wang, Qian and Zheng, Baolin and Zhuang, Xinlu and Li, Qi and Shen, Chao and Wang, Cong},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={826--834},
  year={2021}
}
```

While working on this project, we discovered that a similar approach had already been explored in the paper below. Given this, we’ve decided to discontinue further development. However, we hope this repository can still serve as a useful resource for those interested in experimenting with distillation models and backdoor attacks on encoder-only LLMs.

```bibtex
@article{cheng2024transferring,
  title={Transferring backdoors between large language models by knowledge distillation},
  author={Cheng, Pengzhou and Wu, Zongru and Ju, Tianjie and Du, Wei and Liu, Zhuosheng Zhang Gongshen},
  journal={arXiv preprint arXiv:2408.09878},
  year={2024}
}
```
