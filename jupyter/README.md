## Usage 


#### 1. Clone the repositories
```bash
$ git clone https://github.com/pdollar/coco.git
$ cd coco/PythonAPI/
$ module load bwpy/2.0.0-pre1; module load cudatoolkit
$ make
$ python setup.py build
$ python setup.py install
$ cd ../../


#### 2. Preprocessing

```bash
$ python build_vocab.py   
$ python resize.py
$ python resize_val.py
```

#### 3. Train the model

```bash
$ python train.py    
```

#### 4. Test the model

```bash
$ python test.py
```

