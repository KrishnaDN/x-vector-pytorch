# x-vector-pytorch
This repo contains the implementation of the paper "Spoken Language Recognition using X-vectors" in Pytorch
Paper: https://danielpovey.com/files/2018_odyssey_xvector_lid.pdf
Tutorial : https://www.youtube.com/watch?v=8nZjiXEdMH0

## Installation

I suggest you to install Anaconda3 in your system. First download Anancoda3 from https://docs.anaconda.com/anaconda/install/hashes/lin-3-64/
```bash
bash Anaconda2-2019.03-Linux-x86_64.sh
```
## Clone the repo
```bash
https://github.com/KrishnaDN/x-vector-pytorch.git
```
Once you install anaconda3 successfully, install required packges using requirements.txt
```bash
pip iinstall -r requirements.txt
```

## Create manifest files for training and testing
This step creates training and testing files.
```
python datasets.py --processed_data  /media/newhd/youtube_lid_data/download_data --meta_store_path meta/ 
```

## Training
This steps starts training the X-vector model for language identification 
```
python training_xvector.py --training_filepath meta/training.txt --testing_filepath meta/testing.txt --validation_filepath meta/validation.txt
                             --input_dim 40 --num_classes 8 --batch_size 32 --use_gpu True --num_epochs 100
                             
```

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
For any queries contact : krishnadn94@gmail.com
## License
[MIT](https://choosealicense.com/licenses/mit/)