## Create a Gitea Project
- Start docker-compose
```commandline
cd ~/02_mlops_docker

docker-compose up -d mysql gitea
```
- Browser: http://localhost:3000/
- Name: **fastapi**
- On vm terminal
```commandline
mkdir ~/07_fastapi

cd 07_fastapi

touch README.md
git init

git add README.md
git commit -m "first commit"
# This will force you to complete global configs

git remote add origin http://localhost:3000/jenkins/fastapi.git
git push -u origin master
```
### provide credentials 
- Use what you set installing gitea
  - user: jenkins
  - password: Ankara_06

## Open Pycharm Project
- Use **Get from VCS**

![](images/01_create_pycharm_project_from_VCS.png)

-----

## Create virtualenv and add requirements.txt
```
fastapi[all]==0.86.0
uvicorn[standard]==0.18.3
```

## Create .gitignore
```commandline
.venv
.idea
.env
```
- Commit and push

## On VM pull and create virtualenv
- From vm terminal pull
- Create a virtualenv
```commandline
conda create --name fastapi python=3.8
conda activate fastapi
pip install -r requirements.txt
```

## On PyCharm project add main.py
```python
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}
```
- Push

## On VM run uvicorn
- `  uvicorn main:app --host 0.0.0.0 --port 8002 --reload ` 
- You will see
```commandline
INFO:     Will watch for changes in these directories: ['/home/train/07_fastapi']
INFO:     Uvicorn running on http://0.0.0.0:8002 (Press CTRL+C to quit)
INFO:     Started reloader process [11321] using WatchFiles
INFO:     Started server process [11323]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

- Browser:  http://127.0.0.1:8002

` {"message":"Hello World"} ` 

# Explain codes step by step

- ` from fastapi import FastAPI  `  
FastAPI is a Python class that provides all the functionality for your API.

- ` app = FastAPI() `  
 "instance" of the class FastAPI. This will be the main point of interaction to create all your API.

- Path
A "path" is also commonly called an "endpoint" or a "route".

- Operation   
"Operation" here refers to one of the HTTP "methods".

- path operation function  

- return the content   

## Simple app elements
- Import FastAPI.
- Create an app instance.
- Write a path operation decorator (like @app.get("/")).
- Write a path operation function (like def root(): ... above).
- Run the development server (like uvicorn main:app --reload).
