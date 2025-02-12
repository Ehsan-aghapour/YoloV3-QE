
find val2017/ -maxdepth 1 -type f | shuf -n 100 | xargs -I {} cp {} sample/

