```sh
# Last Update: Last modified: 2026-07-03T20:29:45
# Author: Qi Zhou
```

## Prepare the Docker

### 1. Warmup
Make sure you are at the project folder:
```sh
ls
# You should see folders like:
# config
# data
# docs
# ...
```
---

### 2. Build Docker image
Make sure your Docker desktop App is running.

```sh
docker build -f deploy/Dockerfile.Flow-Alert -t fa-image:v1dot3 .
# docker build >> create a Docker image
# -f deploy/Dockerfile.Flow-Alert >> use this specific Dockerfile
# -t fa-image:v1dot3 >> name the image "fa", tag it as "dev"
# . >> use current directory as the build context (project root)

# for Lightsail
# docker build -f deploy/Dockerfile.Flow-Alert --platform linux/amd64 -t fa-image:v1dot3 .


# force remove the image
# docker rmi -f fa-image:v1dot3 .
```
---

### 3. Check your image or package version
DISK USAGE is around 3 GB, not bad.
```sh
docker images
# you will see:
# IMAGE          ID             DISK USAGE   CONTENT SIZE   EXTRA
# fa-image:v1dot3   4a901ceeb85c       2.81GB          685MB    
```
---

### 4. Run a .py file in Docker container
enter the container
```sh
docker run --rm -it --entrypoint bash fa-image:v1dot3
```

activate env
```sh
micromamba activate env-qz # "env-qz" is the env name in docker container

# install a package: micromamba install -c conda-forge paramiko
```

run script
```sh
python functions/toolkit/env_test.py # exit # exit docker
```

exit docker
```sh
exit
```

---


### 5. Develop stage without rebuild
The "code changes" are mounted into the Docker container,
all edits on QZ's machine are instantly visible inside Docker without rebuilding.

```sh
docker run -it --rm \
  -v $(pwd):/project \
  -w /project \
  fa-image:v1dot3 \
  func/toolkit/env_test.py

# -it >> run container in interactive terminal mode
# --rm >> Automatically removes the container after it exits
# -v $(pwd):/project >> mount local project folder "$(pwd)" into "/project" inside container
# -w /project >> set working directory to "/project"
# fa-image:v1dot3 >> image name "fa", and tag name "dev"
# func/toolkit/env_test.py >> script to run (no rebuild needed)
```
---


### 6. Save and load the image
Save to local PC
```sh
docker save -o deploy/fa_v1dot3.tar fa-image:v1dot3
```

Once you upload it server, you can use it
```sh
# now you should in "deploy" folder
docker load -i fa_v1dot3.tar

# delete it by
docker rmi fa-image:v1dot3
```
---
