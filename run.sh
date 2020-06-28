#/bin/bash

sudo supervisorctl update
sudo supervisorctl stop all
sudo supervisorctl start 任务名