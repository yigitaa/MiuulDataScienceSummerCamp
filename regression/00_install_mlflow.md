## 1. Copy 02_mlops_docker
- Copy from drive to vm
- Change directory ` cd 02_mlops_docker/`

- Make wait-for-it.sh executable  
` chmod +x wait-for-it.sh `  

### 1.1. Install chrony
- To synchronize time
```commandline
sudo yum -y install chrony
sudo systemctl enable chronyd
sudo systemctl start chronyd
```
## 2. Start docker compose
### 2.1. Upgrade compose
```
sudo curl -L "https://github.com/docker/compose/releases/download/v2.17.3/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

docker-compose --version
```

` docker-compose up --build -d `

- docker-compose will take approximately 30 mins.

## 3. Change prod and test containers volume ownership
```commandline
sudo chown train:train -R prod/home/
sudo chown train:train -R test/home/
```

## 4. Port Forwarding on Virtualbox
![Port forwarding](images/virtual_box_port_forwarding.png 'VirtualBox Port forwarding')

Additional reference port forwarding: https://www.xmodulo.com/access-nat-guest-from-host-virtualbox.html#:~:text=On%20VirtualBox%2C%20choose%20the%20guest,can%20configure%20port%20forwarding%20rules.

## 5. Web UIs
- After defining port forwarding rules on Virtuabox network open your browser

### 5.1. MLflow Web UI
http://127.0.0.1:5000


!['MLflow Web UI'](images/mlflow_web_ui.png 'MLflow Web UI')

### 5.2. MinIO Web UI
- http://127.0.0.1:9001  
- trainkey  
- trainsecret

### 5.3. Gitea Web UI and Initial Installation
http://127.0.0.1:3000 

!['Gitea Web UI'](images/gitea_web_ui_install.png 'Gitea Web UI')

- Click ` Install Gitea `

- If you see page not found. Don't worry. It is not problem. Refresh page youl will see login screen.

### 5.4. Gitea User create
![Gitea User create](images/07_gitea_create_user.png 'Gitea User create')
- user: jenkins
- mail: jenkins@vbo.local
- Password Ankara_06
### 5.5. Gitea User create
![Gitea User create2](images/08_gitea_create_user2.png 'Gitea User create2')


### 5.6. Gitea change UI language
![Change Gitea Language1](images/change_gitea_ui_language1.png 'Change Gitea Language1')

--------------------------------

![Change Gitea Language2](images/change_gitea_ui_language2.png 'Change Gitea Language2')

## 6. Close docker-compose
` docker-compose down `

## 7. Start particular sevices
` docker-compose up -d mlflow mysql minio `
