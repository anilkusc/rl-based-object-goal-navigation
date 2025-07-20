sudo docker build -t test .
sudo docker run -dit --gpus all -v ./outputs/:/app/outputs test