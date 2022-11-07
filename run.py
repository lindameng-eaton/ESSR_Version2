from azure_lib import *
import numpy as np
import os
from HiZLib import *
sys.path.append(
    "C:\\Users\\E0644597\\Eaton\\ESSR - CIP\\PythonCode_Linda\\azure-storage-repo\\azure-data-storage")


def run_timecorr(blob_list, connect_str, container_name, t0, tend, Label, thr, tauU_outBound, tauD_outBound, alarm_threshold, alarm_dur, savePlot=False, plot=False):
    """run time correlation through the datasets in cloud
    Args:
    blob_list: the list of the files in the blob storage
    connect_str: connect string for the azure blob storage account
    container_name: container name in the blob storage account
    t0: start time
    tend: end time
    Label: a list of Labels for the files in the blob_list. 0: normal 1: fault
    thr: threshold for baseline calculation
    tauU_outBound, tauD_outBound: time constant for the decision logic
    alarm_threshold: the threshold for the decision logic 
    alarm_dur: duration for alarm continuously being out of boundary to report fault
    savePlot: boolean. True: save plot; False: not save
    plot: boolean. True: plot time correlation, False: not plot 

    Output:tuple
    thr, alarm_threshold, alarm_dur,tauU_outBound: as above
    Accur: Accuracy
    F1: F1 score
    False_alarm:false alarm
    mean_respond : mean responding time




    """

    temp_file = os.path.join('.', 'temp_file.h5')
    channel_name = 'Ipt4'
    delayT = []
    # Label = np.array([0]*21 + [1]* 68)
    # t0=2
    # tend =20
    i = 0
    for blob_name in blob_list:
        blob_client = BlobClient.from_connection_string(
            conn_str=connect_str, container_name=container_name, blob_name=blob_name)
        with open(temp_file, "wb") as my_blob:
            blob_data = blob_client.download_blob()
            blob_data.readinto(my_blob)

        if blob_name.endswith('.h5') and (not blob_name.__contains__('5p288')):
            data = data_acquisition(channel_name, temp_file)
            os.remove(temp_file)

            T = data.iloc[:, 0]
            data = data.loc[(T > t0) & (T < tend), :]
            fs = 5/(T[5]-T[0])
            if fs > 10000:
                data = data.iloc[::10, :]
                T = data.iloc[:, 0].values-data.iloc[0, 0]
                fs = 5/(T[5]-T[0])
                print(fs)
            else:
                T = data.iloc[:, 0].values-data.iloc[0, 0]
            f0 = 50
            stat = data.iloc[:, -1].to_numpy()
            indx = np.nonzero(np.diff(stat))
            if len(indx[0]):
                tst = T[indx[0]]

            else:
                tst = None

            sig = data.iloc[:, 4:7]

            wsize = int(fs/f0*2)
            # wstep = int(fs/f0)

            file = os.path.basename(blob_name)
            filename = file.split('.')[0]
            print(filename, Label[i])
            # deltaT =wstep/fs
            # beta1 = 1-deltaT/0.2
            # beta2 = 1-deltaT/2
            # print('beta1={}, beta2={}'.format(beta1,beta2))
            column_list = sig.columns
            # thr = 0.4
            rmvf = 6 * f0
            delay_time = plotTimeCorrelation(sig, column_list, wsize, fs, thr, rmvf=rmvf, saveFolder='.\\plot_timecorr', figname=filename, savePlot=savePlot,
                                             tauU_outBound=tauU_outBound, tauD_outBound=tauD_outBound, alarm_threshold=alarm_threshold, alarm_dur=alarm_dur, tst=tst, plot=plot)

            delayT.append(delay_time)
            i += 1

    delayT = np.array(delayT)
    Label = Label[np.arange(len(delayT))]

    delayT = [k[0] if isinstance(k, np.ndarray) else k for k in delayT]
    y = np.array(delayT)
    positive_indx = (y < 1)*1
    Accur, False_alarm, F1 = calculate_evaluation(positive_indx, Label)

    print("threshold: {}, alarm_threhold:{}, alarm_dur: {}, tauU:{}".format(
        thr, alarm_threshold, alarm_dur, tauU_outBound))
    print('Time correlation Accuracy: {}, F1: {}, False_alarm: {}'.format(
        Accur, F1, False_alarm))
    mean_respond = np.nanmean(y[y < 1])+alarm_dur
    print('average detecttime is {}'.format(mean_respond))
    return (thr, alarm_threshold, alarm_dur, tauU_outBound, Accur, F1, False_alarm, mean_respond)


def run_multualInfo_local():
    """run multualInformation through the entire local datasets"""

    chanel_list = ['Test_V', 'Test_I', 'ground_I',
                   'Voltage', 'Test/Fault Current', 'Leakage Current']
    # root = "C:\\Users\\E0644597\\Eaton\\ESSR - ERL\\Technology Development\\\Franksville Test\\20220728\\7-28-22 csv backup"
    root = "C:\\Users\\E0644597\\Eaton\\ESSR - ERL\\Technology Development\\\Franksville Test\\20220727\\7-27-22 csv backup"

    dir_list = os.listdir(root)
    f0 = 60
    buffer_dur = 0.5
    for i in range(len(dir_list)):
        file_path = os.path.join(root, dir_list[i])
        data = data_acquisition(chanel_list, file_path,
                                keywords_brk='solenoid')
        filename = dir_list[i].split('.')[0]

        V = data.loc[:, chanel_list[0]]
        I = data.loc[:, chanel_list[1]]

        nbins = 100
        rmvf = 2*f0

        T = data.iloc[:, 0].to_numpy()
        fs = 5/(T[5]-T[0])
        f0 = 60
        column_list = [chanel_list[0]]
        wsize = int(fs/f0*4)
        noverlap = wsize//2

        thr = 2
        stat = data.iloc[:, -1].to_numpy()
        indx = np.nonzero(np.diff(stat))
        if len(indx[0]):
            tst = T[indx[0]]
            print(tst)
        else:
            tst = None

        tauU_outBound = 0.2
        tauD_outBound = 0.2
        beta1_dur = 0.2
        beta2_dur = 3
        alarm_dur = 0.4
        alarm_threshold = 0.5

        plot_mulInformation(V, I, T, fs, column_list, nbins, rmvf, thr, filename, tst, wsize, noverlap, tauU_outBound, tauD_outBound,
                            beta1_dur=beta1_dur, beta2_dur=beta2_dur, buffer_dur=buffer_du, alarm_threshold=alarm_threshold, alarm_dur=alarm_dur, savePlot=True)


def run_LSTM():
    """run LSTM with the entire datasets in cloud"""

    os.environ["HTTP_PROXY"] = "http://zproxy.etn.com:8080"

    # os.environ["HTTPS_PROXY"] = "http://zproxy.etn.com:8080"
    connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    print(connect_str)
    # Create the BlobServiceClient object which will be used to create a container client
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    # Create a unique name for the container

    container_name = "ieee34"
    local_path = "."

    # Create the container
    container_client = blob_service_client.get_container_client(container_name)
    # walk_container(container_client,connect_str, container_name,local_path)
    blob_list = get_blob_list(container_client)

    temp_file = os.path.join('.', 'temp_file.h5')
    channel_name = 'Ipt4'
    undersampling = False

    target = np.array([0]*21 + [1] * 68)

    # parameters

    # number of types
    n_type = len(np.unique(target))
    # LSTM time sequence length
    window_size_LSTM = 40
    # number of shifts
    nShift_LSTM = 1

    # batch size
    batch_size = 30

    # first 2 seconds of data is deleted
    t0 = 2
    tend = 8

    # list of the feature arrays for different cases
    feature_list0 = []
    feature_list1 = []

    i = 0
    for blob_name in blob_list:

        blob_client = BlobClient.from_connection_string(
            conn_str=connect_str, container_name=container_name, blob_name=blob_name)
        with open(temp_file, "wb") as my_blob:
            blob_data = blob_client.download_blob()
            blob_data.readinto(my_blob)

        if i >= len(target):
            break

        if (blob_name.endswith('.h5')) and (not blob_name.__contains__('5p288')):

            tg = target[i]
            data = data_acquisition(channel_name, temp_file)
            os.remove(temp_file)
            T = data.iloc[:, 0]
            data = data.loc[(T > t0) & (T < tend), :]
            T = data.iloc[:, 0].values-data.iloc[0, 0]

            fs = 5/(T[5]-T[0])
            f0 = 50
            stat = data.iloc[:, -1].to_numpy()
            indx = np.nonzero(np.diff(stat))
            if len(indx[0]):
                tst = T[indx[0]]
                # print(tst)
            else:
                tst = None
            Npoints = int(fs/f0)
            window_size = Npoints * 2
            # fstep = fs/window_size/f0
            freq_list = np.arange(1, 15, 1)*f0
            nShift = Npoints//4
            print(blob_name, tg)

            if np.sum(stat > 0):

                d = data.loc[stat > 0, :].copy()
            else:
                d = data.copy()

            d.loc[:, 'Status'] = tg

            d = d.iloc[:, 1:8]
            d = d.to_numpy()
            # convert dataframe into tensorflow Dataset
            dataset = windowed_dataset(d, window_size, nShift)

            feature_list_perfile0 = []
            feature_list_perfile1 = []
            for win, targ in dataset:
                fft_array = np.array([])

                for j in range(win.shape[1]):
                    df = win[:, j].numpy()
                    _, fftH = myfft(df, fs, freq_list)

                    fft_array = np.append(fft_array, np.log10(
                        np.abs(fftH[1:])/np.abs(fftH[0])))
                fft_array = np.append(fft_array, targ)
                if targ == 0:
                    feature_list_perfile0.append(fft_array)
                else:
                    feature_list_perfile1.append(fft_array)

            # print(len(feature_list))
            if targ == 0:
                feature_list0.append(feature_list_perfile0)
            else:
                feature_list1.append(feature_list_perfile1)
            i += 1

    feature_list_new0 = []
    if n_type > 2:
        for FL in feature_list0:
            FL = np.array(FL)
            tg = FL[:, -1]
            tg = to_categorical(tg, n_type+1)
            FL_ = FL[:, :-1]
            FL = np.concatenate((FL_, tg), axis=1)
            feature_list_new0.append(FL)
    else:
        feature_list_new0 = feature_list0

    feature_list_new1 = []
    if n_type > 2:
        for FL in feature_list1:
            FL = np.array(FL)
            tg = FL[:, -1]
            tg = to_categorical(tg, n_type+1)
            FL_ = FL[:, :-1]
            FL = np.concatenate((FL_, tg), axis=1)
            feature_list_new1.append(FL)
    else:
        feature_list_new1 = feature_list1

    # creat tensorflow datasets

    if n_type > 2:
        col = tg.shape[1]*-1
    else:
        col = -1
        tg = tg.reshape(1, -1)

    k = 0
    for FL in feature_list_new0:
        FL = np.array(FL)

        ds = windowed_dataset(FL, window_size_LSTM, nShift_LSTM, col)
        # ds.map(lambda win, tg: (tf.cast(win,tf.float32),tf.cast(tg,tf.int32)))
        if k == 0:

            dataset0 = ds
        else:
            dataset0 = dataset0.concatenate(ds)
        k += 1

    k = 0
    for FL in feature_list_new1:
        FL = np.array(FL)

        ds = windowed_dataset(FL, window_size_LSTM, nShift_LSTM, col)
        print(len(list(ds)))
        if k == 0:

            dataset1 = ds
        else:
            dataset1 = dataset1.concatenate(ds)
        k += 1

    # Shuffle the windows
    # shuffle buffer
    shuffle_buffer0 = len(list(dataset0))
    dataset0 = dataset0.shuffle(shuffle_buffer0)
    shuffle_buffer1 = len(list(dataset1))
    dataset1 = dataset1.shuffle(shuffle_buffer1)

    # undersampling
    if undersampling:
        takesize = min(shuffle_buffer0, shuffle_buffer1)
        dataset0_sub = dataset0.take(takesize)
        dataset1_sub = dataset1.take(takesize)
        dataset = dataset0_sub.concatenate(dataset1_sub)
        dataset = dataset.shuffle(takesize*2)
        dataset_unchanged = dataset0.concatenate(dataset1)
        dataset_size = takesize*2
    else:
        dataset_size = shuffle_buffer0 + shuffle_buffer1
        dataset = dataset0.concatenate(dataset1)
        dataset = dataset.shuffle(dataset_size)
        dataset_unchanged = dataset

    Label = []
    for win, targ in dataset:
        Label.append(targ.numpy())
    Label = np.array(Label).flatten()
    print(np.sum(Label)/len(Label))
    unique_classes = np.unique(Label)
    # compute weights
    weights = sklearn.utils.class_weight.compute_class_weight(
        'balanced', unique_classes, Label)
    weights_ = {c: weights[i] for i, c in enumerate(unique_classes)}
    #train, test, split
    # dataset_size = takesize*2
    train_size = int(0.6*dataset_size)
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    TG = []
    fftImg = []
    for win, targ in train_dataset:
        fftImg.append(win.numpy())
        TG.append(targ.numpy())
    TG = np.array(TG).flatten()
    print('fault rate in training dataset: {}'.format(np.sum(TG)/len(TG)))
    fftImg = np.array(fftImg)
    scaler = StandardScaler()
    fftImg_transform = scaler.fit_transform(
        fftImg.reshape(fftImg.shape[0], -1))
    fftImg_transform = fftImg_transform.reshape(
        fftImg.shape[0], fftImg.shape[1], fftImg.shape[2])

    fftImg_test = []
    TG_test = []
    for win, targ in test_dataset:
        win_transform = scaler.transform(win.numpy().reshape(1, -1))
        win_transform = win_transform.reshape(fftImg.shape[1], fftImg.shape[2])
        fftImg_test.append(win_transform)
        TG_test.append(targ)
    fftImg_test = np.array(fftImg_test)
    TG_test = np.array(TG_test).flatten()
    print(fftImg_test.shape, TG_test.shape)

    # train_dataset = train_dataset.map(lambda win, targ: (tf.data.Dataset.from_tensor_slices(scaler.transform(win.numpy().reshape(1,-1)).reshape(fftImg.shape[1], fftImg.shape[2])),targ))
    # test_dataset = test_dataset.map(lambda win, targ: (tf.data.Dataset.from_tensor_slices(scaler.transform(win.numpy())),targ))

    # Create batches of windows

    # train_dataset = train_dataset.batch(batch_size,drop_remainder = True).prefetch(1)
    # test_dataset = test_dataset.batch(batch_size,drop_remainder = True).prefetch(1)

    feature_size = win.shape[-1]

    # LSTM model
    tf.keras.backend.clear_session()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            8, input_shape=[window_size_LSTM, feature_size], return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8)),

        tf.keras.layers.Dense(
            tg.shape[1], activation=tf.keras.activations.sigmoid)
    ])

    # lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    #     lambda epoch: 1e-8* 10**(epoch / 16))
    stoplearn = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-5)

    if n_type == 2:
        loss = tf.keras.losses.BinaryCrossentropy()
        metric = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC(
        ), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    else:
        loss = tf.keras.losses.CategoricalCrossentropy()
        metric = [tf.keras.metrics.CategoricalAccuracy()]

    model.compile(loss=loss, optimizer=optimizer, metrics=metric)

    history = model.fit(fftImg_transform, TG, batch_size=batch_size, epochs=100, callbacks=[
                        stoplearn, PlotLossesKeras()], validation_data=(fftImg_test, TG_test), class_weight=weights_)
    filename = "completed_model.joblib"
    joblib.dump(model, filename)

    yreal = []
    fftImg_total = []
    for win, targ in dataset_unchanged:
        win_array = win.numpy()
        win_transform = scaler.transform(win_array.reshape(1, -1))
        win_transform = win_transform.reshape(
            win_array.shape[0], win_array.shape[1])
        fftImg_total.append(win_transform)

        yreal.append(targ)
    fftImg_total = np.array(fftImg_total)
    ypred = model.predict(fftImg_total)
    # ypred = np.array(ypred)
    yreal = np.array(yreal)
    ypred_f = ypred.squeeze().flatten()
    yreal_f = yreal.squeeze().flatten()
    plt.scatter(yreal_f, ypred_f)

    fraction_of_positives, mean_predicted_value = calibration_curve(
        yreal_f, ypred_f, n_bins=50)
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], 'k:', label="Perfectly calibrated", lw=4)
    ax1.plot(mean_predicted_value, fraction_of_positives,
             "s-", label='LSTM', lw=4, color='b')
    ax1.set_xlabel('mean_predicted_value', size=20)
    ax1.legend(bbox_to_anchor=(0.5, 1.0), borderpad=2, fontsize=20)
    ax1.set_ylabel('fraction of positive', size=20)
    ax2.hist(ypred_f, range=(0, 1), bins=50, histtype="step", lw=4)
    ax2.set_xlabel('predicted_value', size=20)
    plt.tight_layout()
