```sh
# Last Update: Last modified: 2026-07-08T10:15:11
# Author: Qi Zhou
```

## Prepare the Docker

### 1. Warmup
Make sure you are at the deploy folder:
```sh
ls
# You should see folders like:
# compose-Flow-Alert.yml
# Dockerfile
```
---

### 1. Run the docker compose
```sh
docker compose -f compose-Flow-Alert.yml up -d

# -f compose-Flow-Alert.yml >> use this file instead of default yml
# up >> start the services defined in compose-Flow-Alert.yml
# -d >> detached mode, run in the background
# --build fa-image:v1dot3 >> rebuild the Docker image before starting
```
---

### 2. Check the status
```sh
docker compose -f compose-Flow-Alert.yml ps
```
or show the logs
```sh
docker compose -f compose-Flow-Alert.yml logs --tail=100 flow-alert
```
---

### 3. Stop the whole services
```sh
docker compose -f compose-Flow-Alert.yml down
```
---


### 4. Replace a py file 
You can replace a py file and restart the service
```sh
# Make sure you are in the directory containing t3_main.py
docker cp t3_main.py deploy-flow-alert-1:/app/functions/liveshow/t3_main.py
```
---

### 5. Restart one service
```sh
docker compose -f compose-Flow-Alert.yml restart flow-alert
```
---
