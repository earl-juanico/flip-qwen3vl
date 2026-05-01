# Third-Party Notices

This project uses or builds upon the following third-party assets.  Each entry
lists the owner, version or release date where applicable, canonical URL, license,
and any additional terms.

---

## 1. Qwen3-VL (Qwen3-VL-4B-Instruct, Qwen3-VL-8B-Instruct)

**Owner:** Alibaba Cloud / Tongyi Lab (Qwen Team)

**Version:** Qwen3-VL, released May 2025
(model IDs: `Qwen/Qwen3-VL-4B-Instruct`, `Qwen/Qwen3-VL-8B-Instruct`)

**URLs:**
- Model: <https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct>
- Model: <https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct>
- Technical report: <https://arxiv.org/abs/2505.09838>
- GitHub: <https://github.com/QwenLM/Qwen3-VL>

**License:** Apache License 2.0
(<https://www.apache.org/licenses/LICENSE-2.0>)

**Citation:**
```bibtex
@misc{qwen3vl2025,
  title   = {Qwen3-VL Technical Report},
  author  = {Qwen Team},
  year    = {2025},
  url     = {https://arxiv.org/abs/2505.09838}
}
```

**Terms:** Model weights are redistributed under the Apache 2.0 licence.
This project applies in-process monkey-patching to the Qwen3-VL forward pass
for research purposes; no model weights are modified or redistributed.

---

## 2. Kimi-VL-A3B (Kimi-VL-A3B-Instruct)

**Owner:** Moonshot AI

**Version:** Kimi-VL-A3B, released March 2025
(model ID: `moonshotai/Kimi-VL-A3B-Instruct`)

**URLs:**
- Model: <https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct>
- Technical report: <https://arxiv.org/abs/2504.07491>
- GitHub: <https://github.com/MoonshotAI/Kimi-VL>

**License:** MIT License

**Citation:**
```bibtex
@misc{kimivl2025,
  title   = {Kimi-VL Technical Report},
  author  = {Kimi Team},
  year    = {2025},
  url     = {https://arxiv.org/abs/2504.07491}
}
```

**Terms:** Model weights are made available by Moonshot AI under the MIT
License.  Users should consult the licence file distributed with the model
weights and the Moonshot AI Terms of Service before commercial deployment.

---

## 3. MS COCO val2017

**Owner:** Microsoft Corporation and contributors (COCO Consortium)

**Version:** COCO 2017 validation split (5,000 images,
`instances_val2017.json`, `captions_val2017.json`)

**URLs:**
- Homepage: <https://cocodataset.org/>
- Download: <http://images.cocodataset.org/zips/val2017.zip>
- Annotations: <http://images.cocodataset.org/annotations/annotations_trainval2017.zip>
- Paper: <https://arxiv.org/abs/1405.0312>

**License:** Images — Creative Commons Attribution 4.0 International (CC BY 4.0).
Annotations — Creative Commons Attribution 4.0 International (CC BY 4.0).

**Citation:**
```bibtex
@inproceedings{lin2014microsoft,
  title     = {Microsoft {COCO}: Common Objects in Context},
  author    = {Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and
               Hays, James and Perona, Pietro and Ramanan, Deva and
               Doll{\'a}r, Piotr and Zitnick, C. Lawrence},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2014}
}
```

**Terms:** The COCO dataset is used **unmodified** for evaluation only
(no redistribution of images or annotations).  Users must comply with the
CC BY 4.0 terms and the COCO Terms of Use
(<https://cocodataset.org/#termsofuse>).  The dataset may not be used for
training models without separate authorisation from image rights-holders.

---

## 4. LMMS-Eval

**Owner:** LMMS-Lab (Bo Li, Yuanhan Zhang, and contributors)

**Version:** See project repository for pinned release
(<https://github.com/EvolvingLMMs-Lab/lmms-eval/releases>)

**URLs:**
- GitHub: <https://github.com/EvolvingLMMs-Lab/lmms-eval>
- Paper: <https://arxiv.org/abs/2407.12772>

**License:** Apache License 2.0
(<https://www.apache.org/licenses/LICENSE-2.0>)

**Citation:**
```bibtex
@inproceedings{zhang2024lmmseval,
  title     = {{LMMs-Eval}: Reality Check on the Evaluation of Large Multimodal Models},
  author    = {Zhang, Kaichen and Lyu, Bo and Ma, Haotian and Huang, Yanpeng and
               Li, Fanyi and Li, Yuhang and Xu, Jinxuan and Ye, Haoxuan and
               Wang, Rui and Zhang, Yuanhan and Li, Bo},
  booktitle = {NeurIPS 2024 Workshop on Multimodal Foundation Models},
  year      = {2024},
  url       = {https://arxiv.org/abs/2407.12772}
}
```

**Terms:** LMMS-Eval is used as an evaluation reference and framework;
any portions of its code incorporated into this project are reproduced under
the Apache 2.0 license.

---

*All other dependencies (PyTorch, vLLM, Transformers, NumPy, statsmodels, etc.)
are listed with their respective versions in `requirements.txt`.  Each carries
its own open-source licence; consult the individual package metadata for details.*
