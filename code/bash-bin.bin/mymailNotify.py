#!/usr/bin/python3
# -*- coding:utf-8 -*-
'''
# idea: http://www.zhidaow.com/post/python-send-email-with-smtplib
when:Mon Sep 19 13:30:13 CST 2016
what: send email from 163 to 163
'''
import sys
import smtplib  
from email.mime.text import MIMEText  # 引入smtplib和MIMEText
def sendMail(mailReceiver=['icyhearts@163.com','like_cumt@163.com'],mailSubject="Hello world",mailContent="This is a mail\n"):
	host = 'smtp.163.com'  # 设置发件服务器地址
	port = 25  # 设置发件服务器端口号。注意，这里有SSL和非SSL两种形式
	sender = 'cumtprinter@163.com'  # 设置发件邮箱，一定要自己注册的邮箱
	password = 'Cumt2013'  # 设置发件邮箱的密码，等会登陆会用到
	msg=MIMEText(mailContent,'plain','utf-8')
	msg['subject']=mailSubject
	msg['from']=sender
	msg['to']=";".join(mailReceiver)
	s=smtplib.SMTP(host,port)
	s.login(sender,password)
	s.sendmail(sender,mailReceiver,msg.as_string())
if(len(sys.argv) != 2):
	print("usage: xxx.py mailSubject")
	sys.exit(1)
mailSubject = sys.argv[1]
sendMail(mailReceiver=["cumtprinter@163.com"], mailSubject=mailSubject)
