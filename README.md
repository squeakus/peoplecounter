# peoplecounter
Web interface to show how many people are in a room
# build
docker build . -t webserver
# run
docker run --network=host -it --rm webserver
