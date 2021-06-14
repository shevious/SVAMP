#!/bin/bash

#apt-get update -y
#python3 -m pip install --upgrade pip
#python3 -m pip install tensorboard==1.12.2 tensorboardX==1.2 gensim==3.8.0
#python3 -m pip install torch==1.9.0.dev20210526+cu111 torchtext==0.10.0.dev20210526 torchvision==0.10.0.dev20210526+cu111 -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html
#python3 -m pip install transformers==4.6.1
#python3 -m pip install pandas==1.2.4 attrdict==2.0.1 nltk==3.4.5

python3 -m pip install torch==1.8.1 torchtext==0.9.1 torchvision==0.9.1 transformers==4.6.1
python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('beomi/KcELECTRA-base')"
python3 -c "from transformers import AutoModel; AutoModel.from_pretrained('beomi/KcELECTRA-base')"
