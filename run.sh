#!/bin/bash
python3 /bin/server.py &
apache2ctl -D FOREGROUND
