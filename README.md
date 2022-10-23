# stable-diffusion

## How to run
```bash
cat env_tmp > .env
# put your huggingface token inside .env

docker build -t stable_diffusion:latest .
docker run -it -p 7860:7860 stable_diffusion:latest
```
App will be run on [127.0.0.1:7860](http://127.0.0.1:7860)
