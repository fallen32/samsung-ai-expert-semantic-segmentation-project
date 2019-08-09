#! /bin/bash

function gdown(){
    # usage: gdown [id] [name]
    CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
    rm -rf /tmp/cookies.txt
}

mkdir -p data
cd data
gdown 1LY_Q2_uRKAuPF-xdOMi46ca5bVNRNjvM data.tar.gz
tar -xzvf data.tar.gz
