until python3 main.py --dataset=rtvtr --model=resnet50 --data=../dataset2/ --pruning-method=22 --batch=32 --pruning_config=./configs/imagenet_resnet50_prune72.json --lr-decay-every=10 --momentum=0.9 --epochs=25 --pruning=False > training_log.log 2>&1 & disown; do
    echo "system crashed : error : $?, respawning... " &> crash.log
    sleep 1
done