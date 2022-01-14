import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import Counter

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, img = cap.read()
    img_small = img[190:190+100,270:270+100]
    img_small = img_small.reshape((img_small.shape[0] * img_small.shape[1],3)) #represent as row*column,channel number

    clt = KMeans(n_clusters=3) #cluster number
    clt.fit(img_small)

    count = Counter(clt.labels_)
    common_label = count.most_common(1)[0][0]
    common_color = clt.cluster_centers_[common_label]


    cv2.rectangle(img,(270,190),(270+100,190+100),common_color,5)
    cv2.imshow('frame',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
