
# SoundSpring

Pytorch implementation of the paper "SoundSpring: Loss-Resilient Audio Transceiver with Dual-Functional Masked Language Modeling" in IEEE JSAC.


[![ieee](https://img.shields.io/badge/IEEE-blue.svg)](https://ieeexplore.ieee.org/abstract/document/10845854)&nbsp;[![arXiv](https://img.shields.io/badge/arXiv%20paper-2501.12696-b31b1b.svg)](https://arxiv.org/abs/2501.12696)&nbsp;[![Demo](https://img.shields.io/badge/Demo-green.svg)](https://semcomm.github.io/ResilientAudio/)&nbsp;


## Introduction

In this paper, we propose “SoundSpring”, a cutting-edge error-resilient audio transceiver that marries the robustness benefits of joint source-channel coding (JSCC) while also being compatible with current digital communication systems. Unlike recent deep JSCC transceivers, which learn to directly map audio signals to analog channel-input symbols via neural networks, our SoundSpring adopts the layered architecture that delineates audio compression from digital coded transmission, but it sufficiently exploits the impressive in-context predictive capabilities of large language (foundation) models. Integrated with the casual-order mask learning strategy, our single model operates on the latent feature domain and serve dual-functionalities: as efficient audio compressors at the transmitter and as effective mechanisms for packet loss concealment at the receiver. By jointly optimizing towards both audio compression efficiency and transmission error resiliency, we show that mask-learned language models are indeed powerful contextual predictors, and our dual-functional compression and concealment framework offers fresh perspectives on the application of foundation language models in audio communication. Through extensive experimental evaluations, we establish that SoundSpring apparently outperforms contemporary audio transmission systems in terms of signal fidelity metrics and perceptual quality scores. These new findings not only advocate for the practical deployment of SoundSpring in learning-based audio communication systems but also inspire the development of future audio semantic transceivers.



## Acknowledgement

The implementation is based on [Encodec](https://github.com/facebookresearch/encodec/).

## Citation
If you find the code helpful in your research or work, please cite:
```
@ARTICLE{yao2025soundspring,
  author={Shengshi, Yao and Jincheng, Dai and Xiaoqi, Qin and Sixian, Wang and Siye, Wang and Kai, Niu and Ping, Zhang},
  journal={IEEE Journal on Selected Areas in Communications}, 
  title={SoundSpring: Loss-Resilient Audio Transceiver with Dual-Functional Masked Language Modeling}, 
  year={2025},
  doi={10.1109/JSAC.2025.3531406}
}
```
