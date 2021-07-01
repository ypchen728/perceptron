import numpy as np
import random
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import filedialog
import tkinter as tk
import re

threshold = -1
correction_rate = 0

def readFile(filename):
    minimum_x = 9999
    maximum_x = -9999
    minimum_y = 9999
    maximum_y = -9999
    f = open(filename)
    lines = f.readlines()
    line = []
    for index in range(0,lines.__len__(),1):
        word = []
        dimension = 0
        for words in lines[index].split():
            word.append(float(words))
            dimension = dimension + 1
        line.append(word)
        if maximum_x < word[0]:
            maximum_x = word[0]
        elif minimum_x >word[0]:
            minimum_x = word[0]
        if maximum_y < word[1]:
            maximum_y = word[1]
        elif minimum_y >word[1]:
            minimum_y = word[1]
    dimension = dimension - 1
    return dimension, line, minimum_x, minimum_y, maximum_x, maximum_y

def random_w0(dimension):
    w0 = []
    for index in range(dimension+1):
        buf='{:.8f}'.format(random.uniform(0.1,1))
        w0.append(float(buf))
    return w0

#分類訓練資料、測試資料
def generateTestData(line):
    total_testcase = line.__len__()
    training_testcase = round(total_testcase*2/3)
    shuffle_line = []
    trainingdata = []
    testdata = []
    for index in range(total_testcase):
        shuffle_line.append(random.choice(line))
        line.remove(shuffle_line[index])
    #print("s: ",shuffle_line)
    for index in range(total_testcase):
        if index < training_testcase:
            trainingdata.append(shuffle_line[index])
        else:
            testdata.append(shuffle_line[index])
    return trainingdata,testdata,training_testcase

def adjustWVector_minus(w0,trainingdata,learning_rate):
    trainingdata = [i * learning_rate for i in trainingdata]
    w0 = [w0[i] - trainingdata[i] for i in range(len(w0))] 
    return w0

def adjustWVector_plus(w0,trainingdata,learning_rate):
    trainingdata = [i * learning_rate for i in trainingdata]
    w0 = [w0[i] + trainingdata[i] for i in range(len(w0))]
    return w0

def calculate_output(w0, trainingdata, training_testcase, dimension,learning_rate, N):
    #trainingdata = [[0,0,1],[0,1,1],[1,0,-1],[1,1,1]]
    training_x = []
    for train in trainingdata:
        training_x.append([threshold,train[0],train[1]])
    training_x.append(threshold)
    group_number = trainingdata[0][dimension]
    for index in range(N):
        v = np.inner(w0,training_x[index % training_testcase])
        if v > 0 :
            if trainingdata[index % training_testcase][dimension] == group_number: continue
            elif trainingdata[index % training_testcase][dimension] != group_number:
                w0 = adjustWVector_minus(w0,training_x[index % training_testcase],learning_rate)
        elif v < 0 :
            if trainingdata[index % training_testcase][dimension] != group_number: continue
            elif trainingdata[index % training_testcase][dimension] == group_number:
                w0 = adjustWVector_plus(w0,training_x[index % training_testcase],learning_rate)
    return w0   

def testing(w, trainingdata, testdata, dimension, v_buf, group_number):
    testing_correct_amount = 0
    testdata_x = []
    testingdata_amount = 0
    for test in testdata:
        testdata_x.append([threshold,test[0],test[1]])

    for index in range(testdata_x.__len__()):
        testingdata_amount = testingdata_amount + 1
        v = np.inner(w, testdata_x[index])
        if v < 0 : 
            v_buf1 = -1
        if v > 0 :
            v_buf1 = 1
        if v_buf == v_buf1 and testdata[index][dimension] == group_number:
            testing_correct_amount = testing_correct_amount + 1
        elif v_buf != v_buf1 and testdata[index][dimension] != group_number:
            testing_correct_amount = testing_correct_amount + 1
    
    print("testingdata_amount:",testingdata_amount)
    print("testing_correct_amount:",testing_correct_amount)
    testing_correction_rate = (testing_correct_amount / testingdata_amount) * 100
    print("testing_correction_rate:",testing_correction_rate,"%")
    return testing_correction_rate

def calculate_correctionRate(w, trainingdata, testdata, dimension):
    trainingdata_amount = 0 
    trainingdata_correct_amount = 0 
    group_number = trainingdata[0][dimension]
    training_x = []
    for train in trainingdata:
        training_x.append([threshold,train[0],train[1]])
    v0 = np.inner(w, training_x[0])
    if v0 < 0 : 
        v_buf = -1
    if v0 > 0 :
        v_buf = 1
    for index in range(training_x.__len__()):
        trainingdata_amount = trainingdata_amount + 1
        v = np.inner(w, training_x[index])
        if v < 0 : 
            v_buf1 = -1
        if v > 0 :
            v_buf1 = 1
        if v_buf == v_buf1 and trainingdata[index][dimension] == group_number:
            trainingdata_correct_amount = trainingdata_correct_amount + 1
        elif v_buf != v_buf1 and trainingdata[index][dimension] != group_number:
            trainingdata_correct_amount = trainingdata_correct_amount + 1
    print("trainingdata_amount:",trainingdata_amount)
    print("trainingdata_correct_amount:",trainingdata_correct_amount)
    training_correction_rate = (trainingdata_correct_amount / trainingdata_amount) * 100
    print("training_correction_rate:",training_correction_rate,"%")

    testing_correction_rate = testing(w, trainingdata, testdata, dimension,v_buf,group_number)
    return training_correction_rate, testing_correction_rate

def drawPicture(trainingdata,dimension,group_number,testdata,w,minimum_x, minimum_y, maximum_x, maximum_y):
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    figure = Figure(figsize=(3,3),dpi=100)
    plot = figure.add_subplot(1,1,1)
    for t in trainingdata:
        if t[dimension] == group_number:
            x1.append(t[0])
            y1.append(t[1])
        else:
            x2.append(t[0])
            y2.append(t[1])
    for t in testdata:
        if t[dimension] != group_number:
            x2.append(t[0])
            y2.append(t[1])
        else:
            x1.append(t[0])
            y1.append(t[1])
    l_x = []
    l_y = []
    #l_x.append(float('{:.8f}'.format(w[0]/w[1])))
    #l_x.append(0)
    #l_y.append(0)
    #l_y.append(float('{:.8f}'.format(w[0]/w[2])))
    
    l_x.append(float('{:.8f}'.format((w[0] - minimum_y * w[2])/w[1])))
    l_x.append(float('{:.8f}'.format((w[0] - maximum_y * w[2])/w[1])))
    l_y.append(minimum_y)
    l_y.append(maximum_y)

    plot.scatter(x1, y1, color = "Blue", marker = ".")   #畫點
    plot.scatter(x2, y2, color = "Red", marker = ".")    #畫點
    plot.plot(l_x, l_y, color = "Green", linewidth = 1)    #畫線
    canvas = FigureCanvasTkAgg(figure,window)
    canvas.get_tk_widget().place(x = 10,y = 70)

def SC(l1,l2,window,text1,text2,text3,text4):
    learning_rate_str = l1.get()
    learning_rate = float(learning_rate_str)
    N_str = l2.get()
    N = int(N_str)
    filename =  filedialog.askopenfilename(initialdir = "/",title = "Select File",filetypes = (("TEXT Files","*.txt"),("All Files","*.*")))
    dimension, line, minimum_x, minimum_y, maximum_x, maximum_y = readFile(filename)  #讀檔
    trainingdata, testdata, training_testcase = generateTestData(line)  #隨機分類訓練、測試資料
    w0 = random_w0(dimension)
    group_number = trainingdata[0][dimension]
    
    #print("w0:", w0)
    #print("trainingdata:", trainingdata)
    #print("testdata:", testdata)
    w = calculate_output(w0,trainingdata,training_testcase,dimension,learning_rate,N)
    print("w:",w)

    training_correction_rate, testing_correction_rate = calculate_correctionRate(w, trainingdata, testdata, dimension)

    drawPicture(trainingdata,dimension,group_number,testdata,w,minimum_x, minimum_y, maximum_x, maximum_y)

    label_trainingData = tk.Label(window, text = '訓練辨識率(%) :')
    label_trainingData.place(x = 0, y = 380)
    text1.set(str(training_correction_rate))
    label_trainingData_rate = tk.Label(window, textvariable = text1)
    label_trainingData_rate.place(x = 90, y = 380)

    label_trainingData = tk.Label(window, text = '測試辨識率(%) :')
    label_trainingData.place(x = 0, y = 400)
    text2.set(str(testing_correction_rate))
    label_testingData_rate = tk.Label(window, textvariable = text2)
    label_testingData_rate.place(x = 90, y = 400)

    label_weight = tk.Label(window, text = '初始鍵結值 :')
    label_weight.place(x = 0, y = 420)
    text3.set(str(w0))
    label_weight_value = tk.Label(window, textvariable = text3)
    label_weight_value.place(x = 90, y = 420)

    label_weight_last = tk.Label(window, text = '鍵結值 :')
    label_weight_last.place(x = 0, y = 440)
    text4.set(str(w))
    label_weight_last_value = tk.Label(window, textvariable = text4)
    label_weight_last_value.place(x = 90, y = 440)

if __name__ == '__main__':
    window = tk.Tk()
    text1 = tk.StringVar()
    text2 = tk.StringVar()
    text3 = tk.StringVar()
    text4 = tk.StringVar()
    window.title('My Window')
    window.geometry('500x500')
    label_learning_rate = tk.Label(window, text = '學習率:') 
    label_learning_rate.place(x = 0, y = 0)
    l1 = tk.StringVar()
    e1 = tk.Entry(window, textvariable = l1, show = None, font=('Arial', 10))
    e1.place(x = 70, y = 0)
    label_condition = tk.Label(window, text = '收斂條件(訓練次數):')
    label_condition.place(x = 0, y = 20)
    l2 = tk.StringVar()
    e2 = tk.Entry(window, textvariable = l2, show = None, font=('Arial', 10))
    e2.place(x = 70, y = 20)
    label_data = tk.Label(window, text = '選擇訓練測試資料:') 
    label_data.place(x = 0, y = 40)
    button_data = tk.Button(window, text = '選取檔案', command = lambda: SC(l1,l2,window,text1,text2,text3,text4))
    button_data.place(x = 100, y = 40)

    window.mainloop()