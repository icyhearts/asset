#!/bin/bash
kill -9 $(ps aux | grep [m]pv | awk '{print $2}')
