# -*- coding: utf-8 -*-

import math
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse
import mysql.connector as mariadb
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from nbconvert import HTMLExporter
import codecs
import nbformat

class DataSet:

    def __init__(self, tStart, tStop, lat=46, lon=151, rad=1000, maxdays=30):
        """
        Creates dataset based upon parameters given in constructor
        :param tStart:
        :param tStop:
        :param lat:
        :param lon:
        :param rad:
        """

        self.lat = lat
        self.lon = lon
        self.rad = rad
        self.tStart = tStart
        self.tStop = tStop
        self.rawDataSet = []
        self.maxdays = maxdays
        self.maxtime = self.maxdays * (24 * 60 * 60)

        # Start: positions in database
        self.datePos = 2
        self.timePos = 3

        self.latPos = 4
        self.lonPos = 5
        self.depPos = 6

        self.mbPos = 7
        self.MSPos = 8

        self.ExpPos = 24

        self.MrrPos = 25
        self.MttPos = 27
        self.MppPos = 29
        self.MrtPos = 31
        self.MrpPos = 33
        self.MtpPos = 35

        self.XeigPos = 38
        self.XPluPos = 39
        self.XAziPos = 40
        self.YeigPos = 41
        self.YPluPos = 42
        self.YAziPos = 43
        self.ZeigPos = 44
        self.ZPluPos = 45
        self.ZAziPos = 46

        self.ScMoPos = 47
        # End: positions in database

        self.createDataSet()

    def createDataSet(self):
        """
        Executes query towards database based upon tStart and tStop
        After query result is returned, each events distance from given
        center is compared to _rad_ argument and included in dataset if
        distance is equal to or less than _rad_
        :return:
        """
        queryResult = runQuery(self.tStart, self.tStop)

        rawDataSet = self.getInRadius(self.rad, queryResult)

        print("Created new DataSet containing", len(rawDataSet), "seismic events, ( will be using",
              len(rawDataSet)-2, ")\nRadius:",
              self.rad, " km\nCenter:", self.lat, "°,", self.lon, "°\nFrom:", self.tStart, "\nTo:", self.tStop[:10])

        # Start: for scaling/normalizing
        self.normMag = minmaxabs(rawDataSet, [self.mbPos, self.MSPos])
        self.MTexp = minmaxabs(rawDataSet, self.ExpPos)
        self.normMT = minmaxabs(rawDataSet, [self.MrrPos, self.MttPos, self.MppPos,
                                             self.MrtPos, self.MrpPos, self.MtpPos])
        self.normLat = minmaxabs(rawDataSet, self.latPos)
        self.normLon = minmaxabs(rawDataSet, self.lonPos)
        self.normDep = minmaxabs(rawDataSet, self.depPos)
        self.normScMo = minmaxabs(rawDataSet, self.ScMoPos)
        self.normEig = minmaxabs(rawDataSet, [self.XeigPos, self.YeigPos, self.ZeigPos])
        self.normPlu = minmaxabs(rawDataSet, [self.XPluPos, self.YPluPos, self.ZPluPos])
        self.normAzi = minmaxabs(rawDataSet, [self.XAziPos, self.YAziPos, self.ZAziPos])

        # subtract this value
        self.MTexpSubtr = 18
        # End: for scaling/normalizing

        self.rawDataSet = rawDataSet

    def getInRadius(self, radius, data):
        """
        Method which will return an array of all the events from _data_ which
        are inside of a circle with a center in (self.lat, self.lon) and with
        a radius of _radius_ kilometers.
        :param radius: radius in kilometers
        :param data: data to perform the radius on
        :return: all the seismic events within the circle of radius _radius_
        """
        arr = []

        for i in range(0, len(data)):
            row = data[i]
            if getDistance(self.lat, self.lon, row[self.latPos], row[self.lonPos]) <= radius:
                arr.append(row)

        return arr

    def formatData(self, config=0, norm=1, shape=0):
        """
        Formats a raw dataset.
        :param config: int (0-2) - choose how much data to include. default is 0.
        :param norm: int (0-4) normalize values; 0: no, 1: [-1, 1], 2: [-1, 1] sklearn, 3: sklearn scale, 4: [0, 1] sklearn
        :param shape: 0: [[[f01],[f02],...],[[f11],[f12],...]], 1: [[[f01,f02,...]],[[f11,f12,...]]]
        :return: 3D matrix
        """

        arr = []

        # format data row by row
        for i in range(1, len(self.rawDataSet) - 1):
            dateTime0 = dateAndTime(self.rawDataSet[i - 1][self.datePos], self.rawDataSet[i - 1][self.timePos])
            arr.append(self.formatRow(self.rawDataSet[i], dateTime0, config, norm))

        nparr = np.array(arr)

        if norm is 2:
            nparr = scaledata(nparr, 0)
        elif norm is 3:
            nparr = scaledata(nparr, 1)
        elif norm is 4:
            nparr = scaledata(nparr, 2)

        if shape is 0:
            return nparr.reshape(nparr.shape[0], nparr.shape[1], 1)
        else:
            return nparr.reshape(nparr.shape[0], 1, nparr.shape[1])

    def formatRow(self, row, dateTime0, config=0, norm=1):
        """
        Formats given row into numeric vector
        :param row: row to format
        :param dateTime0: datetime of the previous event
        :param config: int (0-2) - choose how much data to include. default is 0.
        :param norm: normalize values, 0 - no, 1 - yes
        :return: vector (1D)
        """

        arr = []
        dateTime1 = dateAndTime(row[self.datePos], row[self.timePos])
        timeBetween = secBetweenDates(dateTime0, dateTime1)

        if norm is 1:
            # To normalize the values
            normMag = self.normMag
            MTexp = self.MTexp
            normMT = self.normMT
            normLat = self.normLat
            normLon = self.normLon
            normDep = self.normDep
            normScMo = self.normScMo
            normEig = self.normEig
            normPlu = self.normPlu
            normAzi = self.normAzi

            MTexpSubtr = self.MTexpSubtr

            # Time
            arr.append(self.scaleTime(timeBetween))
        else:
            # if the values are not to be normalized/scaled
            normMag = MTexp = normMT = normLat = normLon = normDep = \
            normScMo = normEig = normPlu = normAzi = 1
            MTexpSubtr = 0

            # Time
            arr.append(timeBetween)

        # Magnitude
        arr.append(getMag(row[self.mbPos], row[self.MSPos]) / normMag)

        # Moment Tensor Exponent
        # values between 18 and 29
        arr.append((row[self.ExpPos] - MTexpSubtr) / MTexp)

        # Moment Tensor
        # values between -10 and 10
        arr.append(row[self.MrrPos] / normMT)
        arr.append(row[self.MttPos] / normMT)
        arr.append(row[self.MppPos] / normMT)
        arr.append(row[self.MrtPos] / normMT)
        arr.append(row[self.MrpPos] / normMT)
        arr.append(row[self.MtpPos] / normMT)

        if (config == 1) or (config == 2):

            # Latitude, longitude, depth
            arr.append((row[self.latPos]) / normLat)
            arr.append((row[self.lonPos]) / normLon)
            arr.append((row[self.depPos]) / normDep)

            # Scalar moment
            arr.append((row[self.ScMoPos]) / normScMo)

        if config == 2:

            # Eigenvalue, Plunge, Azimuth
            arr.append((row[self.XeigPos]) / normEig)
            arr.append((row[self.XPluPos]) / normPlu)
            arr.append((row[self.XAziPos]) / normAzi)
            arr.append((row[self.YeigPos]) / normEig)
            arr.append((row[self.YPluPos]) / normPlu)
            arr.append((row[self.YAziPos]) / normAzi)
            arr.append((row[self.ZeigPos]) / normEig)
            arr.append((row[self.ZPluPos]) / normPlu)
            arr.append((row[self.ZAziPos]) / normAzi)

        return np.array(arr)

    def formatDataDiscrete(self, categories=10, config=0, norm=1, shape=0):
        """

        :param categories:
        :param config:
        :param norm:
        :param shape:
        :return:
        """

        discr = np.array(makeDiscrete(categories, self.formatData(config, norm, shape=0)))

        if shape is 1:
            discr = discr.reshape(discr.shape[0], discr.shape[2], discr.shape[1])

        return discr

    def GTMagInDays(self, mag, days=1, output=0):
        """
        Will there be an earthquake of magnitude greater than _mag_ and within _days_.
        :param mag: magnitude - (ex: 5.1)
        :param days: int in range (0 < days < maxDays). - (ex 5)
        :param output: type of output
        :return: [[int]] or [[int, int]]
        """
        # må kanskje endre på daysmaxdays, bruke den samme scaleren

        if output == 0:
            true = [1]
            false = [0]

        elif output == 1:
            true = [1, 0]
            false = [0, 1]

        elif output == 2:
            true = [1, -1]
            false = [-1, 1]

        if 0 < days < self.maxdays:
            daysmaxdays = daysToSec(days)/self.maxtime
            events = np.array(self.getGroundTruthTime(mag)).flatten()
            return np.array(list(map(lambda x: true if (x <= daysmaxdays) else false, events)))

    def getGroundTruthTime(self, mag):
        """
        Returns array of float values representing the (normalized) time
        between each event present in dataset and the next seismic event
        with magnitude equal or greater than mag.
        :param mag: magnitude - (ex: 5.1)
        :return: [[double]] - array of the normalized time in seconds to the next event >= _mag_
        """

        lenDataSet = len(self.rawDataSet) - 2

        arr = lenDataSet * [None]
        for i in range(1, len(self.rawDataSet) - 1):
            start = i + 1
            while start <= lenDataSet + 1:
                if getMag(self.rawDataSet[start][self.mbPos], self.rawDataSet[start][self.MSPos]) >= mag:
                    # add time between the events
                    timeBetween = self.scaledTimeBetween(i, start)
                    arr[i-1] = timeBetween

                    break
                else:
                    # check next
                    start += 1

        newTime = dateForward(self.tStop + ' 00:00:00', self.maxdays)
        dataToMaxTime = self.getInRadius(self.rad, runQuery(self.tStop, newTime))
        nextEvent = None

        for j in range(len(dataToMaxTime)):
            if getMag(dataToMaxTime[j][self.mbPos], dataToMaxTime[j][self.MSPos]) >= mag:
                nextEvent = dataToMaxTime[j]
                break

        if nextEvent is not None:
            for i in range(len(arr)):
                if arr[i] is None:
                    timeBetween = secBetweenDates(
                        dateAndTime(self.rawDataSet[i][self.datePos],
                                    self.rawDataSet[i][self.timePos]),
                        dateAndTime(nextEvent[self.datePos],
                                    nextEvent[self.timePos]))

                    arr[i] = self.scaleTime(timeBetween)

        for i in range(len(arr)):
            if arr[i] is None:
                arr[i] = 1.0

        return arr

    def gt_time_disc(self, mag, categories=10, minimum=None, maximum=None):
        arr = np.array(self.getGroundTruthTime(mag)).astype(np.float)
        arr = np.array(getDiscr3t3(categories, arr, minimum=minimum, maximum=maximum))
        
        return np.array(arr)

    def getGroundTruthNextMag(self):
        """
        Return an array where each element is the magnitude of the next event.
        This is using the function getMag() to find the right magnitude.
        :return: a vector (1D) consisting of the magnitude of the next event.
        """
        arr = []

        for i in range(1, len(self.rawDataSet) - 1):
            arr.append(getMag(self.rawDataSet[i + 1][self.mbPos], self.rawDataSet[i + 1][self.MSPos])/self.normMag)

        return np.array(arr)

    def gt_mag_disc(self, categories=10, minimum=None, maximum=None):
        arr = np.array(self.getGroundTruthNextMag()).astype(np.float)
        arr = getDiscr3t3(categories, arr, minimum=minimum, maximum=maximum)

        return np.array(arr)

    def gttimemaglatlon(self):
        """
        Ground Truth method that returns time, mag, lat, lon.
        :return: 2D numpy-array
        """

        arr = []
        for i in range(1, len(self.rawDataSet) - 1):

            arr.append(self.scaledTimeBetween(i))
            arr.append(getMag(self.rawDataSet[i][self.mbPos], self.rawDataSet[i][self.MSPos]) / self.normMag)
            arr.append(self.rawDataSet[i][self.latPos] / self.normLat)
            arr.append(self.rawDataSet[i][self.latPos] / self.normLon)

        nparr = np.array(arr)
        return nparr.reshape(int(len(nparr)/4), 4)

    def scaledTimeBetween(self, i0, i1=0):
        """
        Input two indices that refers to positions of dates in the raw data set,
        and get the scaled time between them in return.
        :param i0: index of the first date
        :param i1: index of the second date. (default is i0+1)
        :return: scaled time between two dates
        """

        if i1 is 0:
            i1 = i0+1

        timeBetween = secBetweenDates(
            dateAndTime(self.rawDataSet[i0][self.datePos],
                        self.rawDataSet[i0][self.timePos]),
            dateAndTime(self.rawDataSet[i1][self.datePos],
                        self.rawDataSet[i1][self.timePos]))

        return self.scaleTime(timeBetween)

    def scaleTime(self, seconds):
        """
        Retuns a scaled time between events. If time is above maxtime the scaledtime is 1.
        :param seconds: number of seconds since previous event.
        :return: scaled time (between 0 and 1)
        """
        scaletothemax = seconds / self.maxtime

        if scaletothemax >= 1:
            scaledTime = 1
        else:
            scaledTime = scaletothemax

        return scaledTime

    def getRawDataSet(self):
        """
        :return:
        """
        return self.rawDataSet

#############
# Functions #
#############


def makeDiscrete(cat, arr):
    """

    :param cat: Number of categories
    :param arr: The array to process
    :return: 3D array
    """

    arrDisc = []

    for i in range(len(arr)):
        arrDisc.append([])

    for i in range(len(arr[0])):
        valArr = len(arr) * [None]
        for j in range(len(arr)):
            valArr[j] = arr[j][i]

        valArrDisc = getDiscr3t3(cat, valArr)
        idx = 0
        for discArr in valArrDisc:
            for elem in discArr:
                arrDisc[idx].append([elem])

            idx += 1

    return arrDisc


def getDiscr3t3(categories, array, minimum=None, maximum=None):
    """
    Dynamic classification method. Just works (tm).
    :param categories: how many categories to split the data in
    :param array: the data which should be categorized, in a 2-dim array
    :param minimum: which value should be the lowest category. defaults to min(_array_)
    :param maximum: which value should be the highest category. defaults to max(_array_)
    :return: a 2-dim array of discretized/categorized values based on _array_
    """
    array = np.array(array).flatten().astype(np.float)

    if minimum is None:
        minimum = min(array)

    if maximum is None:
        maximum = max(array)

    round_dec = 5
    arr = []
    for i in range(len(array)):
        row = categories * [0]
        row[int((array[i] - round(minimum, round_dec)) * ((categories - 1) / (maximum - round(minimum, round_dec))))] = 1
        arr.append(row)

    return arr


def runQuery(tStart, tStop):
    """
    Get all the rows from all the events within the time period of _tStart_ and _tStop_.
    :param tStart: start of the time period
    :param tStop: stop of the time period
    :return: an array of all the events between _tStart_ and _tStop_
    """
    mariadb_connection = mariadb.connect(user='root', password='password', database='CMT', host='db')
    cursor = mariadb_connection.cursor(buffered=True)

    query = "select * from CMT.CMT where DoRE >= "
    query += '"' + tStart + '"'
    query += " and "
    query += "DoRE <= "
    query += '"' + tStop + '"'
    query += " order by DoRE, ToRE ASC"

    cursor.execute(query)
    result = cursor.fetchall()

    return result


def daysToSec(days):
    """
    Converts from days to seconds.
    :param days:
    :return:
    """
    return days*86400


def dateAndTime(date, time):
    """
    :param date:
    :param time:
    :return: formatted date and time (in the format we use)
    """
    return date + " " + time[:8]


def secBetweenDates(dateTime0, dateTime1):
    """
    :param dateTime0:
    :param dateTime1:
    :return: The number of seconds between two dates.
    """
    dt0 = datetime.strptime(dateTime0, '%Y/%m/%d %H:%M:%S')
    dt1 = datetime.strptime(dateTime1, '%Y/%m/%d %H:%M:%S')

    timeDiff = ((dt1.timestamp()) - (dt0.timestamp()))
    return timeDiff


def dateForward(dateTime0, days):
    """
    Jump forward _days_ from _dateTime0_ and return the new date.
    :param dateTime0:
    :param days:
    :return:
    """
    dt = datetime.strptime(dateTime0, '%Y/%m/%d %H:%M:%S')
    newDate = dt + relativedelta(days=days)
    return newDate.strftime('%Y/%m/%d %H:%M:%S')


def withinDays(datetime0, datetime1, days):
    """
    Check whether or not two dates have more than _days_ days distance.
    :param datetime0:
    :param datetime1:
    :param days:
    :return: true if _datetime0_ and _datetime1_ is <= _days_ separated.
    """
    return (secBetweenDates(datetime0, datetime1)) <= daysToSec(days)


def getMag(mb, MS):
    """
    :param mb:
    :param MS:
    :return: The available magnitude, or the average if both exist.
    """
    if mb > 0 and MS > 0:
        return (mb + MS) / 2
    elif mb == 0:
        return MS
    else:
        return mb


def getDistance(lat1, lon1, lat2, lon2):
    """
    Returns *APPROXIMATE* distance in km between the two coordinates
    :param lat1:
    :param lon1:
    :param lat2:
    :param lon2:
    :return:
    """
    radius = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c
    return d


def minmaxabs(data, pos):

    nparr = np.array([pos]).flatten()
    df = pd.DataFrame(data)
    return max(list(map(lambda x: max(abs(float(df.max()[x])), abs(float(df.min()[x]))), nparr)))


def scaledata(data, scaletype):

    # Scale data
    if scaletype is 0:  # sklearn [-1, 1]
        mm_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        data = mm_scaler.fit_transform(data.astype(np.float))
    elif scaletype is 1:  # sklearn.scale()
        data = preprocessing.scale(data.astype(np.float))
    elif scaletype is 2:  # sklearn [0, 1]
        mm_scaler = preprocessing.MinMaxScaler()
        data = mm_scaler.fit_transform(data.astype(np.float))

    return data


def roundlistat(values, at):
    return list(map(lambda x: roundat(x, at), values))


def roundat(value, at):

    if value < at:
        return 0
    else:
        return 1


def accu(predictions, targets):

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(len(targets)):

        pred = predictions[i][0]
        targ = targets[i][0]

        if targ == 1:
            tp += int(pred == targ)
            fn += int(pred != targ)
        else:
            fp += int(pred != targ)
            tn += int(pred == targ)

    return tp, tn, fp, fn


def accu_discrete(predictions, targets, absolute=0):

    arr = []

    for i in range(len(targets)):
        arr.append(np.argmax(predictions[i]) - np.argmax(targets[i]))

    if absolute is 1:
        arr = list(map(abs, arr))

    return np.array(arr)


def accu_plot_disc(predictions, targets, absolute=0):

    accu_list = accu_discrete(predictions, targets, absolute)
    unique, counts = np.unique(accu_list, return_counts=True)

    plt.bar(unique, counts)
    plt.show()

    total = sum(counts)

    if 0 in unique:
        unipos = np.where(unique == 0)
        correct = counts[unipos[0][0]]
        wrong = total - counts[unipos[0][0]]
        percentage = int(round((correct * 100) / total))

    else:
        percentage = 0
        correct = 0
        wrong = total

    print("Number of correct:", correct)
    print("Number of wrong:", wrong)
    print("Percent correct: " + str(percentage) + '%')

    return percentage


def S_model_Predict(GroundT, enable_random_normal=False):
    s_guess = []
    minim = min(GroundT)
    maxim = max(GroundT)
    len_gt = len(GroundT)
    avg = np.average(GroundT)

    if not enable_random_normal:
        for i in GroundT:
            s_guess.append(avg)
    else:
        standard_deviation = np.std(GroundT)

        while len(s_guess) < len_gt:
            stuff = np.random.normal(avg, standard_deviation)
            if stuff >= minim and stuff <= maxim:
                s_guess.append(stuff)

    return s_guess


def output_HTML(read_file, output_file):

    exporter = HTMLExporter()
    # read_file is '.ipynb', output_file is '.html'
    output_notebook = nbformat.read(read_file, as_version=4)
    output, resources = exporter.from_notebook_node(output_notebook)
    codecs.open(output_file, 'w', encoding='utf-8').write(output)


def best_per_pos(xs, x):
    """
    :param xs: List of numbers
    :param x: Number
    :return: The positions in the list _xs_ where _x_ appears
    """
    return [i for i, j in enumerate(xs) if j == x]


def org_table(ll):
    sep = '|'
    text = ''
    len_ll = len(ll)

    for i in range(len_ll - 1):
        text += sep + join_list(ll[i], sep) + sep + '\n'
    text += sep + join_list(ll[-1], sep) + sep
    return text


def join_list(l, separator=''):
    return separator.join(str(n) for n in l)


def experiment_settings(settings=0):

    act = ['relu', 'sigmoid']
    day = [30, 180]
    epo = [2, 200]
    cal = [0, 1]
    ear = ['', 'es']

    permut = [[0, 0, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 1, 1],
              [0, 1, 0, 0],
              [0, 1, 1, 0],
              [0, 1, 1, 1],
              [1, 0, 0, 0],
              [1, 0, 1, 0],
              [1, 0, 1, 1],
              [1, 1, 0, 0],
              [1, 1, 1, 0],
              [1, 1, 1, 1]]

    act = act[permut[settings][0]]
    day = day[permut[settings][1]]
    epo = epo[permut[settings][2]]
    cal = cal[permut[settings][3]]
    ear = ear[permut[settings][3]]

    return act, day, epo, cal, ear
