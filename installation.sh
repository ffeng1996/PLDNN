pip install -r requirements.txt --user

git clone git://github.com/lisa-lab/pylearn2.git
python ./pylearn2/setup.py develop
python ./pylearn2/setup.py develop --user

# lasagne version: 0.2.dev
git clone https://github.com/Lasagne/Lasagne.git
cd Lasagne
python setup.py install

conda install -c conda-forge caffe
