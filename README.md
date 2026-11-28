<!-- PAI-SG-LINKS:BEGIN -->
## Project Links

- [Project Portal](https://statera-workshop.github.io/#repo-mm_contrastive_siglip)
- [Statera Workshop Portal](https://statera-workshop.github.io/)
<!-- PAI-SG-LINKS:END -->

# MM_Contrastive_SigLIP

**SigLIP sigmoid pre-training on Crop36-Text and COCO - convergence study (PAI-SG Multimodal Wave 1)**

Part of the *Physical AI* reference series of the **Statera Working Group**. One method of the image-text convergence comparison in [MM_Wave1B_Crop36COCO](https://github.com/Statera-Workshop/MM_Wave1B_Crop36COCO).

## Method

Pairwise sigmoid loss with learnable scale (init 10) and bias (init -10), as in open_clip SigLipLoss.

**Common setup** (identical for the contrastive repositories; full comparison in [MM_Wave1B_Crop36COCO](https://github.com/Statera-Workshop/MM_Wave1B_Crop36COCO)).
Dual encoder from scratch: image ResNet-18 (torchvision layout), text Transformer (4 layers, width 256, 4 heads, causal,
EOT pooling; OpenAI CLIP BPE tokenizer), both projected to a 256-d L2-normalised embedding. AdamW lr 5e-4, wd 0.1,
500 warm-up steps then constant (cosine = ablation), batch 256, no augmentation, fp32, seeds 777 / 778 / 779.
Data: **Crop36-Text** - the 26,716 / 6,699 Crop36 object crops (96 x 96, grayscale, 36 classes) of FashionLetters36 with
captions from the class name and the original box size (8 templates); **COCO subset** - 20,000 train2017 / 1,000
val2017 images (seed 777, 128 x 128), 5 captions each. Metrics: Crop36 zero-shot top-1 (3 size-free prompts, ensemble;
chance 0.028); COCO rsum = Recall@1+5+10 image->text + text->image on the 1,000 val images (max 6).

**BatchNorm finding.** With a constant learning rate the BatchNorm running statistics lag the moving weights, so the
plain eval-mode score shows isolated *dips* (a single evaluation falls, the next recovers) and, on COCO, a systematic
under-estimate. Re-estimating the BN statistics on 50 training batches before evaluation (the series' evaluation
practice) removes every dip and gives the *BN re-estimated* column; training is unchanged. With the cosine schedule the
two evaluations agree. *Dips* = evaluation-to-evaluation drops >= 0.1 (>= 0.6 rsum).

## Results (mean of seeds; range in brackets)

### Crop36-Text

| Loss | Init | Batch | Epochs | LR schedule | Seeds | Final zero-shot top-1 (range) | Final, BN re-estimated (range) | E90 | Dips (plain / BN re-est.) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SigLIP | scratch | 256 | 30 | constant | 3 | 0.904 (0.007) | 0.910 (0.004) | 2.0 | 1 / - |
| SigLIP | scratch | 256 | 100 | constant | 3 | 0.902 (0.007) | 0.908 (0.003) | 3.0 | 11 / 0 |
| SigLIP | scratch | 256 | 100 | cosine | 3 | 0.919 (0.004) | 0.918 (0.002) | 3.0 | 0 / 0 |
| SigLIP | scratch | 64 | 30 | constant | 3 | 0.910 (0.001) | 0.915 (0.004) | 2.0 | 1 / - |
| SigLIP | SSL Barlow Twins | 256 | 30 | constant | 3 | 0.925 (0.001) | 0.925 (0.002) | 2.0 | 0 / 0 |
| SigLIP | SSL Barlow Twins | 256 | 100 | constant | 3 | 0.911 (0.041) | 0.928 (0.003) | 3.0 | 0 / 0 |

### COCO subset

| Loss | Init | Batch | Epochs | LR schedule | Seeds | Final rsum (R@1+5+10, both directions, max 6) (range) | Final, BN re-estimated (range) | E90 | Dips (plain / BN re-est.) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SigLIP | scratch | 256 | 15 | constant | 3 | 0.682 (0.119) | 0.763 (0.097) | 8.0 | 0 / - |
| SigLIP | scratch | 256 | 50 | constant | 3 | 0.678 (0.028) | 0.746 (0.010) | 8.0 | 0 / 0 |
| SigLIP | scratch | 256 | 100 | constant | 3 | 0.732 (0.069) | 0.751 (0.040) | 12.0 | 0 / 0 |
| SigLIP | scratch | 256 | 100 | cosine | 3 | 0.747 (0.039) | 0.746 (0.047) | 14.0 | 0 / 0 |

**Finding.** Fastest early convergence on COCO (15 epochs) and marginally the best Crop36 level; it shows the most BatchNorm dips under the constant schedule (none after BN re-estimation).

**SSL initialisation (Crop36).** Image encoder initialised from the SSL Barlow Twins 200-epoch backbone of [SSL_Redundancy_BarlowTwins](https://github.com/Statera-Workshop/SSL_Redundancy_BarlowTwins) (seed-matched; conv1 folded to one input channel), Barlow vs scratch: SigLIP 30 epochs: E90 2.0 vs 2.0, final (BN re-est.) 0.925 vs 0.910; SigLIP 100 epochs: E90 3.0 vs 3.0, final (BN re-est.) 0.928 vs 0.908.

## Weights

Release **v1.0** of this repository (15 files, SHA-256 in `SHA256SUMS.txt`): `coco_siglip_e100_const_s777.pth`, `coco_siglip_e100_const_s778.pth`, `coco_siglip_e100_const_s779.pth`, `coco_siglip_e100_cosine_s777.pth`, `coco_siglip_e100_cosine_s778.pth`, `coco_siglip_e100_cosine_s779.pth`, `crop36_siglip_e100_const_barlow_s777.pth`, `crop36_siglip_e100_const_barlow_s778.pth`, `crop36_siglip_e100_const_barlow_s779.pth`, `crop36_siglip_e100_const_s777.pth`, `crop36_siglip_e100_const_s778.pth`, `crop36_siglip_e100_const_s779.pth`, `crop36_siglip_e100_cosine_s777.pth`, `crop36_siglip_e100_cosine_s778.pth`, `crop36_siglip_e100_cosine_s779.pth`.

```python
import torch, sys; sys.path.insert(0, 'code'); import mm_b as M
tok = M.Tok(32)            # Crop36: context 32, 1 input channel; COCO: M.Tok(64), 3 channels
m = M.DualEncoder('<loss>', tok.vocab, 32, 1)
m.load_state_dict(torch.load('crop36_<loss>_e100_<const|cosine>_s<seed>.pth', weights_only=True), strict=True)
```

**Data note.** COCO images follow the Flickr terms and are not redistributed; the image ids are listed in [MM_Wave1B_Crop36COCO](https://github.com/Statera-Workshop/MM_Wave1B_Crop36COCO) `results/data/coco_sub_list.json`.

## Run

```bash
cd code && python add_crop36_side.py && python prep_coco.py && bash smoke_b.sh
python mm_b.py --data crop36 --loss siglip --epochs 100 --seed 777 --eval-recal
```
