
#### Velocity-Aware 3D Motion Diffusion Physics-Aware Prediction with Fuzzy Dynamics


> 3D human motion prediction is challenging due to the challenge of ensuring kinematic consistency. Existing diffusion-based methods typically treat skeletal data as high-dimensional Euclidean signals and perform blind noise fitting in the ambient space, overlooking the fact that human motion inherently resides on a low-dimensional biomechanical manifold. This limitation often leads to anatomically implausible poses and distorted kinematic structures. We claim that incorporating velocity priors is paramount for maintaining dynamic consistency during denoising. Therefore, we proposed the Dual-Objective Diffusion Prediction Network (DODP), which explicitly constrains the diffusion process along the motion manifold by jointly optimizing noise residual ($\epsilon$) and data velocity (v). Consequently, we design a Fuzzy-aware Dual-channel Module (FuD) that features: (1) a spatio-temporal decoupling mechanism to learn adaptive joint dependencies; and (2) a velocity-aware Type-1 fuzzy attention module to capture motion intensity semantics ("strong/moderate/weak") and modulate uncertainty-aware dynamics. Furthermore, DODP employs a velocity-field-driven skip connection across the diffusion trajectory to propagate motion evolution priors. Experiments demonstrate that our approach outperforms state-of-the-art methods on Human3.6M, AMASS, and HumanEva-I datasets. Specifically, DODP achieves a significant improvement of 12.74\%, 16.75\%, and 4.50\% in ADE on the three benchmarks respectively, validating its effectiveness in generating accurate and natural motion sequences.}
\keywords{Human Motion Prediction, Diffusion, Manifold, Fuzzy, Spatio-Temporal Modeling.


Python/Conda Environment

The following code is tested on Linux-64 in a cluster environment as well as on Windows 11. If you are using Linux-64 in a cluster environment, please use `source activate transfusion` instead of `conda activate transfusion`.

```
mkdir ./checkpoints
mkdir ./data
mkdir ./inference
mkdir ./results
conda create -n transfusion python=3.8
conda activate transfusion
python -m pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install -r requirement.txt
```



## 🌹 Acknowledgment
Project structure is borrowed from [HumanMAC](https://github.com/LinghaoChan/HumanMAC). We would like to thank the authors for making their code publicly available.

