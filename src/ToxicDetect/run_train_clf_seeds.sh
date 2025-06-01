mkdir logs/train_$1/
nohup python -u train_$1_clf.py 0 14592 > logs/train_$1/seed14592.log &
nohup python -u train_$1_clf.py 1 3278 > logs/train_$1/seed3278.log &
nohup python -u train_$1_clf.py 2 36048 > logs/train_$1/seed36048.log &
nohup python -u train_$1_clf.py 0 32098 > logs/train_$1/seed32098.log &
nohup python -u train_$1_clf.py 2 29256 > logs/train_$1/seed29256.log &
nohup python -u train_$1_clf.py 1 18289 > logs/train_$1/seed18289.log &
