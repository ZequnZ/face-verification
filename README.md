# Face Verfification

Use **MTCNN** and **FaceNet** to achieve face verification

## Install

In order to run the code, you need to install all the dependencies in *requirements.txt*:  
`pip install -r requirements`

As the code cannot be run with python3.8 or newer version, I suggest to run it a **virtual env** or ***docker container*。
However, I run this code in Macbook, such that docker container cannot use my camera in the laptop.
The only way to run *fv_carema* is with virtual env

### Virtual env

1. Install [virtualenv](https://virtualenv.pypa.io/en/latest/installation.html)  

2. `virtualenv -p=python3.7 venv`  

3. `source venv/bin/activate`

4. `pip install --no-cache-dir --upgrade --upgrade-strategy=eager -r requirements.txt`

### Run jupyter notebook in a docker container

For the first time, we need to build the container:  
`
docker-compose --project-name face-recognition  -f ./docker-notebook.yml up --build
`

Once you have built the container, you can spin up the notebook with the command:  
`
docker-compose --project-name face-recognition  -f ./docker-notebook.yml up --no-build
`


### Model

Pretrained model can be download [here](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-)

### Flask example

I build a simple web server using *flash*.  
Check `server.py` for details.  
