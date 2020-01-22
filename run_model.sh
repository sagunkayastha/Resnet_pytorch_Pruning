until python3 main_server.py > training_log.log 2>&1 & disown; do
    echo "system crashed : error : $?, respawning... " &> crash.log
    sleep 1
done