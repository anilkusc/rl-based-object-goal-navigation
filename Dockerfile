FROM fairembodied/habitat-challenge:testing_2022_habitat_base_docker
WORKDIR /app
COPY . .
RUN chmod 777 entrypoint.sh
ENTRYPOINT ./entrypoint.sh

#sudo docker run --gpus all -v ./outputs/:/app/outputs test