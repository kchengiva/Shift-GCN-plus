# ShiftGCN++
The implementation for "[Extremely Lightweight Skeleton-Based Action Recognition with ShiftGCN++](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9515708)" (TIP2021). ShiftGCN++ further boosts the efficiency of [ShiftGCN](https://github.com/kchengiva/Shift-GCN), which achieves comparable performance with 6× less FLOPs and 2× practical speedup.

![image](https://github.com/kchengiva/Shift-GCN-plus/blob/main/ShiftGCN_plus.png)

![image](https://github.com/kchengiva/Shift-GCN-plus/blob/main/flops_acc.png)

![image](https://github.com/kchengiva/Shift-GCN-plus/blob/main/fps.png)

## Prerequisite

 - PyTorch 0.4.1
 - Cuda 9.0
 - g++ 5.4.0

## Compile cuda extensions

  ```
  cd ./model/Temporal_shift
  bash run.sh
  ```

## Data Preparation

 - Download the raw data of [NTU-RGBD](https://github.com/shahroudy/NTURGB-D) and [NTU-RGBD120](https://github.com/shahroudy/NTURGB-D). Put NTU-RGBD data under the directory `./data/nturgbd_raw`. Put NTU-RGBD120 data under the directory `./data/nturgbd120_raw`. 
 
 - For NTU-RGBD, preprocess data with `python data_gen/ntu_gendata.py`. For NTU-RGBD120, preprocess data with `python data_gen/ntu120_gendata.py`. 
  
 - Generate the bone data with `python data_gen/gen_bone_data.py`.

 - Generate the motion data with `python data_gen/gen_motion_data.py`.

## Training & Testing

  - NTU X-view

    `python main.py --config ./config/nturgbd-cross-view/train_joint.yaml`

    `python main.py --config ./config/nturgbd-cross-view/train_bone.yaml`

    `python main.py --config ./config/nturgbd-cross-view/train_joint_motion.yaml`

    `python main.py --config ./config/nturgbd-cross-view/train_bone_motion.yaml`

  - NTU X-sub

    `python main.py --config ./config/nturgbd-cross-subject/train_joint.yaml`

    `python main.py --config ./config/nturgbd-cross-subject/train_bone.yaml`

    `python main.py --config ./config/nturgbd-cross-subject/train_joint_motion.yaml`

    `python main.py --config ./config/nturgbd-cross-subject/train_bone_motion.yaml`

  - For NTU-RGBD dataset, we provide trained teacher models for knowledge distillation in `./teacher_models`.
  - For NTU120-RGBD dataset, change the dataset path in config files, and change `num_class` in config files from 60 to 120. You need to train teacher models before train ShiftGCN++ on NTU120-RGBD. 

## Multi-stream ensemble

To ensemble the results of 4 streams. Change models name in `ensemble.py` depending on your experiment setting. Then run `python ensemble.py`.

## Trained models

We release several trained models:

Model|Dataset|Setting|Top1(%)
-|-|-|-
./save_models/ntu_ShiftGCN-plus_joint_xview.pt|NTU-RGBD|X-view|94.8
./save_models/ntu_ShiftGCN-plus_bone_xview.pt|NTU-RGBD|X-view|94.7
./save_models/ntu_ShiftGCN-plus_joint_xsub.pt|NTU-RGBD|X-sub|87.9
./save_models/ntu_ShiftGCN-plus_bone_xsub.pt|NTU-RGBD|X-sub|88.3

## Citation
If you find this model useful for your research, please use the following BibTeX entry.

    @article{cheng2021extremely,
    title={Extremely Lightweight Skeleton-Based Action Recognition With ShiftGCN++},
    author={Cheng, Ke and Zhang, Yifan and He, Xiangyu and Cheng, Jian and Lu, Hanqing},
    journal={IEEE Transactions on Image Processing},
    volume={30},
    pages={7333--7348},
    year={2021},
    publisher={IEEE}
    }
