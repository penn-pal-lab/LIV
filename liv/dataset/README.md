# LIV Datasets Download Instruction

We provide instruction for downloading and using our real-robot dataset used for LIV fine-tuning. The dataset contains 100 teleoperated demonstrations for 9 table-top tasks specified via language.  

## RealRobot Dataset
```bash
$ cd liv/liv/dataset
$ sh download_data.sh
```
Then, generate ``manifest.csv`` file by running
```
python generate_manifest.py
```
with the correct absoulte path of the downloaded dataset folder on line 8. 


Then, we can fine-tune LIV using the realrobot dataset:
```bash
$ cd ..
$ python train_liv.py dataset=realrobot training=finetune
```