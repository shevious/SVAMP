FROM nvidia/cuda:11.1.1-runtime-ubuntu18.04
LABEL submitter=AGC2021
ENV DEBIAN_FRONTEND=noninteractive

### Change Ubuntu repository to daum or kakao mirror (ftp.daum.net or mirror.kakao.com)
RUN rm -rf /var/lib/apt/lists/* && sed -i 's/kr.archive.ubuntu.com/ftp.daum.net/g' /etc/apt/sources.list && sed -i 's/archive.ubuntu.com/ftp.daum.net/g' /etc/apt/sources.list && sed -i 's/security.ubuntu.com/ftp.daum.net/g' /etc/apt/sources.list && sed -i 's/extras.ubuntu.com/ftp.daum.net/g' /etc/apt/sources.list && sed -i 's/mirror.kakao.com/ftp.daum.net/g' /etc/apt/sources.list

### Install Default Package. and Configure Locale, TimeZone###
RUN apt-get update &&  apt-get install -y --no-install-recommends apt-utils locales tzdata make build-essential wget curl tar unzip gcc g++ git zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libffi-dev openjdk-8-jdk libssl-dev && locale-gen ko_KR.UTF-8
ENV LC_ALL=ko_KR.UTF-8
ENV TZ=Asia/Seoul

### Configure User ###
ENV USER=agc2021
ENV UID=1001
ENV GID=3000
ENV GROUPS=2000
ENV HOME=/home/${USER}
RUN adduser --disabled-password --gecos "Default user" --uid ${UID} ${USER}

### Install Python with pyenv ###
ARG PYTHON_VER=3.8.3
ENV PYENV_ROOT=${HOME}/.pyenv
ENV PATH=$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
ENV PATH=$PYENV_ROOT/versions/${PYTHON_VER}/bin:$PATH
RUN mkdir -p ${HOME}/.local/Fbin && mkdir -p ${HOME}/.local/lib && mkdir -p ~/.pip && git clone https://github.com/pyenv/pyenv.git ${HOME}/.pyenv && pyenv install ${PYTHON_VER} && pyenv global ${PYTHON_VER} &&  echo "export PYENV_ROOT='$HOME/.pyenv'" >> ~/.bashrc && echo "export PATH='$HOME/.local/bin:$PATH'" >> ~/.bashrc && echo "export PATH='$PYENV_ROOT/bin:$PATH'" >> ~/.bashrc && echo "eval '$(pyenv init - --no-rehash)'" >> ~/.bashrc && echo '[global]\nindex-url=http://ftp.daumkakao.com/pypi/simple\ntrusted-host=ftp.daumkakao.com' >> ~/.pip/pip.conf
ARG DISABLE_CACHE=None

### ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ ###
# Do not modify the code above.               
# We are not responsible for any problems     
# that arise when the above code is modified. 
# ------------------------------------------- 
# 상단의 코드를 수정하지 마십시오.               
# 상단의 코드를 수정하여 발생하는 문제에 대해서는 
# 저희는 책임지지 않습니다.                      
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ #

### 사용자 공간
RUN python3 -m pip install torch==1.9.0.dev20210526+cu111 torchtext==0.10.0.dev20210526 torchvision==0.10.0.dev20210526+cu111 -f "https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html" && python3 -m pip install transformers==4.6.1 && python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('beomi/KcELECTRA-base')" && python3 -c "from transformers import AutoModel; AutoModel.from_pretrained('beomi/KcELECTRA-base')"

### ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ ###
# Do not modify the code below.               
# We are not responsible for any problems     
# that arise when the below code is modified. 
# ------------------------------------------- 
# 하단의 코드를 수정하지 마십시오.               
# 하단의 코드를 수정하여 발생하는 문제에 대해서는 
# 저희는 책임지지 않습니다.                      
# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ #
RUN chown -R ${USER}:${USER} ${PYENV_ROOT}/ && chown -R ${USER}:${USER} ${HOME}/
RUN mkdir -p /home/agc2021/
RUN chown -R ${USER}:${USER} /home/agc2021/
COPY --chown=${USER}:${USER} . ${HOME}/
USER ${USER}

### Set Working directory ###
WORKDIR ${HOME}

### Update python package manger ###
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel --user
RUN python3 -m pip install kubernetes==17.17.0 grpcio==1.37.0 grpcio-tools==1.37.0 pika==1.2.0  --user
RUN python3 -m pip install -r requirements.txt --user

