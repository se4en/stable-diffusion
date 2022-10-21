# stable-diffusion

## How to run
```bash
cat env_temp > .env
# put your huggingface token inside .env

docker build -t stable_diffusion:latest .
docker run -it -p 7860:7860 stable_diffusion:latest
```
