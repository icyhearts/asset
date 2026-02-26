#!/bin/bash

GENTOO_ROOT=/dev/sda5
GENTOO_HOME=/dev/sda4
GENTOO_BOOT=/dev/sda3

mount ${GENTOO_ROOT} /mnt/gentoo
mount ${GENTOO_HOME} /mnt/gentoo/home/
mount ${GENTOO_BOOT} /mnt/gentoo/boot/
mount -t proc /proc/ /mnt/gentoo/proc/
mount --rbind /dev/ /mnt/gentoo/dev/
mount --rbind /sys/ /mnt/gentoo/sys/
mount --make-rslave /mnt/gentoo/dev/
mount --make-rslave /mnt/gentoo/sys/
