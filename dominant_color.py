import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def find_histogram(clt):
    '''
    create a histogram with k clusters
    :param: clt
    :return:hist
    '''
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist

def plot_colors2(hist, centroids):
    bar = np.zeros((250, 250, 3), dtype="uint8")

    color = centroids[np.argmax(hist)].astype("uint8").tolist()
    cv2.rectangle(bar, (0, 0), (250, 250), color, -1)
    cv2.putText(bar,"RGB: ("+','.join(map(str,color))+")",(0,245),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

    # return the bar chart
    return bar

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret==True:
        # Create a copy of the ROI from frame
        roi = frame[190:290,270:370].copy()
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        roi = roi.reshape((roi.shape[0] * roi.shape[1],3)) # represent as row*column, channel number
        clt = KMeans(n_clusters=3) # cluster number
        clt.fit(roi)

        hist = find_histogram(clt)
        bar = plot_colors2(hist, clt.cluster_centers_)

        # Place bounding box in frame
        frame = cv2.rectangle(frame,(270,190),(370,290),(0,0,255),3)

        bar = cv2.cvtColor(bar,cv2.COLOR_BGR2RGB)
        cv2.imshow('frame',frame)
        cv2.imshow('dominant color',bar)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()