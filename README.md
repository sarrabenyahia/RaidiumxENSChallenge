# ENSxRaidium Challenge: Automatic Segmentation of Anatomical Structures in CT Scans

This repo contains the code and resources developed by us for the ENSxRaidium challenge. The goal of the challenge is to automatically segment the anatomical structures of the human body from CT scans without semantically identifying associated organs. The challenge focuses on identifying visible shapes on a CT scan, without exhaustive annotations.
You can see the details of the challenge [here](https://challengedata.ens.fr/participants/challenges/105/).

## Background
CT scans are widely used in medical imaging to diagnose and treat diseases. Automatic segmentation of anatomical structures from CT scans can help in various tasks, such as surgical planning, radiation therapy, and disease diagnosis. However, traditional supervised segmentation algorithms require extensive annotations for each structure of interest, which can be time-consuming, expensive, and prone to errors. Moreover, generalizing to new anatomical structures not seen before is not possible with supervised learning.

The ENSxRaidium challenge addresses these limitations by focusing on unsupervised or weakly supervised methods that can segment multiple structures simultaneously without explicit labels or semantic information. The challenge provides a dataset of CT scans with partial annotations of a few structures and asks participants to predict the segmentation masks of other structures in the same scans.

## Our approach
We tackled the challenge by using a combination of deep learning and image processing techniques. Specifically, we used a pre-trained U-Net model to segment the visible structures in the CT scans, followed by post-processing steps to refine the segmentation masks and separate overlapping structures. We also experimented with data augmentation and transfer learning using SAM and MedSAM to improve the performance of the model.

## Data
1. Create the data folder
2. Download the data [here](https://challengedata.ens.fr/participants/challenges/105/)
3. Insert data as follow:
```
-- RaidiumxENSChallenge/
    |-- README.md
    |-- requirements.txt
    |-- .gitignore
    |-- src/
        |-- medsam/
            |-- work_dir/
                |-- SAM/
                    |-- sam_vit_b_01ec64.pth
                |-- MedSAM/
                    |-- medsam_20230423_vit_b_0.0.1.pyh
            |-- utils/
                |-- precompute_img_embed.py
            
        |-- sam/
            |-- SAM_inference.ipynb
        |-- unet/
            |-- load_data.py
            |-- unet_medical_seg.ipynb
    |-- data/
        |-- Y_train.csv
        |-- X_train/
            |-- 0.png
            |-- ...
            |-- 999.png
        |-- X_test/
            |-- 0.png
            |-- ...
            |-- 499.png
        |-- Supp_train/
            |-- plots/
                |-- seg_0.png
                |-- ...
                |-- seg_999.png              
            |-- segmentations/
                |-- 0.png
                |-- ...
                |-- 999.png                
            
```

## Installation
1. Create a virtual environment `python3.10 -m venv .env` and activate it `source .env/bin/activate`
2. Install [Pytorch 2.0](https://pytorch.org/get-started/locally/)
3. `git clone https://github.com/sarrabenyahia/RaidiumxENSChallenge`
4. Run `pip install -r requirements.txt`

## To-do-list
- [ ] Enlarge the dataset
- [ ] Run inference with MedSam on bounding box mode and stack the predictions
- [ ] Try running the code on GPU
- [ ] Explore other fine-tuning/pre-trained inference methods 

## Acknowledgements
We express our deep gratitude to the challenge organizers and the owners of the dataset for their invaluable contribution to the community by providing a public dataset. We would also like to extend our appreciation to Meta AI for generously sharing the source code for segmenting anything.Furthermore, we would like to extend our heartfelt appreciation to the team at Toronto University for sharing their valuable work on fine-tuning MedSAM, and for their eagerness to allow us to contribute to their project.

You can see the original repo of MedSam [here](https://github.com/bowang-lab/MedSAM).

## License
This project is licensed under the MIT License.
