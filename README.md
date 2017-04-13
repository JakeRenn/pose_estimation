# pose_estimation

Steps to run this project:

* run the command `./script/init_dir.sh` to create necessary directories.
* place your training data into the `data` directory
* place your label inio the `labels/txt` directory
* modify the class `Config()` in `train.py`
* modify the data path and label path for reader object, which locate at LINE 69 in `train.py`

run you model with folloing command

    python train
or

    ./script/run.sh
