#!/bin/bash

DEBIAN_ROOT=/dev/sda10
DEBIAN_HOME=/dev/sda4

mount ${DEBIAN_ROOT} /mnt/debian
mount ${DEBIAN_HOME} /mnt/debian/home/
mount -t proc /proc/ /mnt/debian/proc/
mount --rbind /dev/ /mnt/debian/dev/
mount --rbind /sys/ /mnt/debian/sys/
mount --make-rslave /mnt/debian/dev/
mount --make-rslave /mnt/debian/sys/
