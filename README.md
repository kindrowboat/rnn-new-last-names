# rnn-new-last-names
generated new last names using recurrent neural networks

## Steps
### start GCP instance with nvidia-docker
```bash
gcloud beta compute instances create nvidia-docker \
    --machine-type n1-standard-4 \
    --min-cpu-platform "Intel Broadwell" \
    --zone us-west1-c \
    --boot-disk-size=100GB --boot-disk-auto-delete --boot-disk-type=pd-ssd \
    --accelerator type=nvidia-tesla-k80,count=1 \
    --image ubuntu-1604-xenial-v20170307 \
    --image-project ubuntu-os-cloud \
    --maintenance-policy TERMINATE \
    --restart-on-failure \
    --metadata startup-script='#!/bin/bash
    sudo apt-get update
    sudo apt-get install -y wget
    sudo apt-get install -y linux-headers-$(uname -r)
    sudo apt-get install -y gcc
    sudo apt-get install -y make
    sudo apt-get install -y g++
    wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
    sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
    sudo apt-get update
    sudo apt-get install -y cuda
    export PATH=$PATH:/usr/local/cuda/bin
    sudo curl -fsSL https://get.docker.com/ | sh
    sudo curl -fsSL https://get.docker.com/gpg | sudo apt-key add -
    wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
    sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb
```

### prepare workspace on GCP instance
1. SSH onto GCP instance
1. clone this repo `git clone https://github.com/motevets/rnn-new-last-names.git`
1. start torch-rnn container with repo mounted `nvidia-docker run -v rnn-new-last-names:/host --rm -ti crisbal/torch-rnn:cuda7.5 bash`

### preprocess, train, and generate new names
```bash
python scripts/preprocess.py \
--input_txt /host/2010_census/input.txt \
--output_h5 /host/2010_census/preprocessing.h5 \
--output_json /host/2010_census/preprocessing.json
```

```bash
th train.lua \
-input_h5 /host/2010_census/preprocessing.h5 \
-input_json /host/2010_census/preprocessing.json
```

```bash
th sample.lua -checkpoint cv/checkpoint_20600.t7 -length 10000 > /host/2010_census/generated_from_checkpoint_206000.txt
```

Copy training before exiting container
```bash
cp -r cv /host/2010_census
```

## Credits
* https://github.com/PipelineAI/pipeline/wiki/GCP-GPU-Tensorflow-Docker
* https://github.com/crisbal/docker-torch-rnn
* https://github.com/jcjohnson/torch-rnn
