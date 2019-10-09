future_cog41 = []
future_cog42 = []
future_cog43 = []
future_cog44 = []
future_cog45 = []

ilist1 = []
ilist2 = []
ilist3 = []
ilist4 = []
ilist5 = []

naccid = cogscores4[0][1]
for i in range(len(cogscores4)-1):
    if (cogscores4[i][1] != naccid):
        naccid = cogscores4[i][1]
    if (cogscores4[i+1][1] == naccid and i < 17700):
        train = np.insert(cogscores4[i,2:37], 1, cogscores4[i+1, 3:5])
        label = cogscores4[i+1,18:37]
        thing = np.concatenate((train, label))
        ilist1.append(i+1)
        future_cog41.append(thing)
    if (cogscores4[i+1][1] == naccid and 17700 <= i < 34700):
        train = np.insert(cogscores4[i,2:37], 1, cogscores4[i+1, 3:5])
        label = cogscores4[i+1,18:37]
        thing = np.concatenate((train, label))
        ilist2.append(i+1)
        future_cog42.append(thing)
    if (cogscores4[i+1][1] == naccid and 34700 <= i < 51700):
        train = np.insert(cogscores4[i,2:37], 1, cogscores4[i+1, 3:5])
        label = cogscores4[i+1,18:37]
        thing = np.concatenate((train, label))
        ilist3.append(i+1)
        future_cog43.append(thing)
    if (cogscores4[i+1][1] == naccid and 51700 <= i < 68400):
        train = np.insert(cogscores4[i,2:37], 1, cogscores4[i+1, 3:5])
        label = cogscores4[i+1,18:37]
        thing = np.concatenate((train, label))
        ilist4.append(i+1)
        future_cog44.append(thing)
    if (cogscores4[i+1][1] == naccid and 68400 <= i):
        train = np.insert(cogscores4[i,2:37], 1, cogscores4[i+1, 3:5])
        label = cogscores4[i+1,18:37]
        thing = np.concatenate((train, label))
        ilist5.append(i+1)
        future_cog45.append(thing)            

ticker =  0
for i in range(len(future_cog41)):
    j = ilist1[i]
    if (np.count_nonzero(np.subtract(future_cog41[i][35:54], cogscores4[j, 17:36])) > 0):
        ticker+=1
print(ticker)

ticker =  0
for i in range(len(future_cog42)):
    j = ilist2[i]
    if (np.count_nonzero(np.subtract(future_cog42[i][35:54], cogscores4[j, 17:36])) > 0):
        ticker+=1
print(ticker)

ticker =  0
for i in range(len(future_cog43)):
    j = ilist3[i]
    if (np.count_nonzero(np.subtract(future_cog43[i][35:54], cogscores4[j, 17:36])) > 0):
        ticker+=1
print(ticker)

ticker =  0
for i in range(len(future_cog44)):
    j = ilist4[i]
    if (np.count_nonzero(np.subtract(future_cog44[i][35:54], cogscores4[j, 17:36])) > 0):
        ticker+=1
print(ticker)

ticker =  0
for i in range(len(future_cog45)):
    j = ilist5[i]
    if (np.count_nonzero(np.subtract(future_cog45[i][35:54], cogscores4[j, 17:36])) > 0):
        ticker+=1
print(ticker)
        
train1 = future_cog42+future_cog43+future_cog44+future_cog45
train2 = future_cog41+future_cog43+future_cog44+future_cog45
train3 = future_cog41+future_cog42+future_cog44+future_cog45
train4 = future_cog41+future_cog42+future_cog43+future_cog45
train5 = future_cog41+future_cog42+future_cog43+future_cog44

train1 = np.array(train1)
train2 = np.array(train2)
train3 = np.array(train3)
train4 = np.array(train4)
train5 = np.array(train5)

xend = 36
yend = 55

trainx1 = train1[:,:xend]
trainy1 = train1[:, xend:yend]
trainx2 = train2[:,:xend]
trainy2 = train2[:, xend:yend]
trainx3 = train3[:,:xend]
trainy3 = train3[:, xend:yend]
trainx4 = train4[:,:xend]
trainy4 = train4[:, xend:yend]
trainx5 = train5[:,:xend]
trainy5 = train5[:, xend:yend]


test1 = future_cog41
test2 = future_cog42
test3 = future_cog43
test4 = future_cog44
test5 = future_cog45

test1 = np.array(test1)
test2 = np.array(test2)
test3 = np.array(test3)
test4 = np.array(test4)
test5 = np.array(test5)


testx1 = test1[:,:xend]
testy1 = test1[:, xend:yend]
testx2 = test2[:,:xend]
testy2 = test2[:, xend:yend]
testx3 = test3[:,:xend]
testy3 = test3[:, xend:yend]
testx4 = test4[:,:xend]
testy4 = test4[:, xend:yend]
testx5 = test5[:,:xend]
testy5 = test5[:, xend:yend]


trainx1 = np.array(trainx1)
trainy1 = np.array(trainy1)
trainx2 = np.array(trainx2)
trainy2 = np.array(trainy2)
trainx3 = np.array(trainx3)
trainy3 = np.array(trainy3)
trainx4 = np.array(trainx4)
trainy4 = np.array(trainy4)
trainx5 = np.array(trainx5)
trainy5 = np.array(trainy5)

testx1 = np.array(testx1)
testy1 = np.array(testy1)
testx2 = np.array(testx2)
testy2 = np.array(testy2)
testx3 = np.array(testx3)
testy3 = np.array(testy3)
testx4 = np.array(testx4)
testy4 = np.array(testy4)
testx5 = np.array(testx5)
testy5 = np.array(testy5)

print(np.count_nonzero((np.subtract(trainx2[1:3, 16:35], trainy2[0:2]))))

epochs = 100
batch_size = 32
learning_rate = .0005
concat = 1

errors_list1 = []
errors_list4 = []
falsepos4 = []
falseneg4 = []
errorsum = []
roundederrorsum = []

trainx1 = np.reshape(trainx1, (len(trainx1), 36, 1))
trainx2 = np.reshape(trainx2, (len(trainx2), 36, 1))
trainx3 = np.reshape(trainx3, (len(trainx3), 36, 1))
trainx4 = np.reshape(trainx4, (len(trainx4), 36, 1))
trainx5 = np.reshape(trainx5, (len(trainx5), 36, 1))

testx1 = np.reshape(testx1, (len(testx1), 36, 1))
testx2 = np.reshape(testx2, (len(testx2), 36, 1))
testx3 = np.reshape(testx3, (len(testx3), 36, 1))
testx4 = np.reshape(testx4, (len(testx4), 36, 1))
testx5 = np.reshape(testx5, (len(testx5), 36, 1))

traindatax = [trainx1, trainx2, trainx3, trainx4, trainx5]
traindatay = [trainy1, trainy2, trainy3, trainy4, trainy5]

testdatax = [testx1, testx2, testx3, testx4, testx5]
testdatay = [testy1, testy2, testy3, testy4, testy5]
    
print("######")
for i in range(5):
    for j in range(6):
        trmean = np.mean(traindatax[i][:, j])
        traindatax[i][:, j] = traindatax[i][:, j]-trmean
        traindatax[i][:, j] = traindatax[i][:, j]/np.abs(traindatax[i][:, j]).max(axis=0)
        temean = np.mean(testdatax[i][:, j])
        testdatax[i][:, j] = testdatax[i][:, j]-temean
        testdatax[i][:, j] = testdatax[i][:, j]/np.abs(testdatax[i][:, j]).max(axis=0)
    
for i in range(5):
    print(i)
    model4 = Sequential()
    model4.add((Conv1D(128,1, activation = "relu", input_shape = (36, 1))))
    model4.add((Conv1D(64,1, activation = "relu")))
    model4.add((MaxPooling1D(1)))
    model4.add(LSTM(100, activation = "tanh", return_sequences = True))
    model4.add(LSTM(100, activation = "tanh"))
    model4.add(Dense(100, activation = "relu"))
    model4.add(Dense(19))
    optimizer = keras.optimizers.Adam(lr=learning_rate)
    model4.compile(loss='mae',optimizer=optimizer)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.25,patience=20, min_lr=1e-20, verbose=1)
    history = model4.fit(x = traindatax[i], y =traindatay[i], batch_size = batch_size,
                         epochs=epochs,
                         validation_data=(testdatax[i], testdatay[i]),
                         callbacks=[reduce_lr])
    print("########################")
    my_predict = model4.predict(testdatax[i])
    diff = np.subtract(testdatay[i], my_predict)
    errors_list4.append(np.mean(np.absolute(diff)))
    falsepos4.append(len(diff[diff > 0]))
    falseneg4.append(len(diff[diff < 0]))
    print(len(diff[diff < 0]))
    print(len(diff[diff > 0]))
    print(np.sum(np.absolute(diff)))
    print(np.sum(np.absolute(np.subtract(testdatax[i][:,17:36, 0], testdatay[i]))))
    errorsum.append(np.sum(np.absolute(diff)))
    my_predict = np.rint(my_predict*2)/2
    diff = np.subtract(testdatay[i], my_predict)
    print(np.sum(np.absolute(diff)))
    errorsum.append(np.sum(np.absolute(diff)))
    if (i==0):
        model4.save("/Users/Montague/Desktop/DStuff/naccrnn1")
    if (i==1):
        model4.save("/Users/Montague/Desktop/DStuff/naccrnn2")
    if (i==2):
        model4.save("/Users/Montague/Desktop/DStuff/naccrnn3")
    if (i==3):
        model4.save("/Users/Montague/Desktop/DStuff/naccrnn4")
    if (i==4):
        model4.save("/Users/Montague/Desktop/DStuff/naccrnn5")
    
    
    
    
    










