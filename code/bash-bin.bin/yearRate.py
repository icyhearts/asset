import datetime
import time
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('-t1', default='2019-10-08', type=str, help='year(4)-month(2)-date(2)')
parser.add_argument('-p1', default='338.58', type=str, help='price 1')
parser.add_argument('-t2', default='2020-02-20', type=str, help='year(4)-month(2)-date(2)')
parser.add_argument('-p2', default='354.402', type=str, help='price 2')



def main():
    global args
    args = parser.parse_args()
    secPerDay = 24.0*60*60
    takenStr = args.t1
    form = "%Y-%m-%d"
    takenDatetime = datetime.datetime.strptime(takenStr, form) # datetime.datetime
##
    timetuple = takenDatetime.timetuple() # time.struct_time(tm_year=2009, tm_mon=9, tm_mday=20, tm_hour=14, tm_min=35, tm_sec=32, tm_wday=6, tm_yday=263, tm_isdst=-1)
    ## 得到纪元时间
    seconds1 = time.mktime(takenDatetime.timetuple())
    days1 = seconds1/secPerDay

###
    takenStr = args.t2
    form = "%Y-%m-%d"
    takenDatetime = datetime.datetime.strptime(takenStr, form) # datetime.datetime
##
    timetuple = takenDatetime.timetuple() # time.struct_time(tm_year=2009, tm_mon=9, tm_mday=20, tm_hour=14, tm_min=35, tm_sec=32, tm_wday=6, tm_yday=263, tm_isdst=-1)
    ## 得到纪元时间
    seconds2 = time.mktime(takenDatetime.timetuple())
    days2 = seconds2/secPerDay

    deltaDay = days2-days1
    p1 = eval(args.p1)
    p2 = eval(args.p2)
    yearRatio = ((p2-p1)/p1 ) /deltaDay*365   * 100
    print('yearRatio=', yearRatio)
if __name__ == '__main__':
    main()
