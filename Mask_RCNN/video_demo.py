import cv2
from visualize_cv2 import model, display_instances, class_names, apply_mask_without_color
import sys
import numpy as np

def maakHistogram(image):
    histogramLijst = []
    color = ('b', 'g', 'r')
    # Maak voor elke kleurcomponent een histogram = matrix met vorm 256 x 1
    # Hou histogrammen bij in histogramLijst ==> histogramLijst.append
    for i, col in enumerate(color):
        histogram = cv2.calcHist([image], [i], None, [256], [10, 256])
        histogramLijst.append(histogram)
    return histogramLijst

def interpreteerHistogram(histogramLijst):
    # Voor elk histogram (3 in totaal) in de histogramLijst:
    tellerBlauw = 0
    tellerGroen = 0
    tellerRood = 0
    tellerTot = 0
    for i in range(150, 256):
        tellerBlauw = tellerBlauw + histogramLijst[0][i][0] * i / 100
        tellerGroen = tellerGroen + histogramLijst[1][i][0] * i / 100
        tellerRood = tellerRood + histogramLijst[2][i][0] * i / 100
    tellerTot = tellerBlauw + tellerGroen + tellerRood
    verhoudingGroen = tellerGroen / tellerTot
    print("B: " + str(tellerBlauw) + " G: " + str(tellerGroen) + " R: " + str(tellerRood) + " Tot: " + str(tellerTot))
    # print("Verhouding G/Tot: " + str(verhoudingGroen))
    if tellerRood > tellerGroen and tellerRood > tellerBlauw and verhoudingGroen > 0.35:
        print("Op basis van histogram: gele banaan")
    elif tellerGroen > tellerBlauw and tellerGroen > tellerRood:
        print("Op basis van histogram: Groene banaan")
    elif tellerRood > tellerGroen and tellerRood > tellerBlauw and verhoudingGroen < 0.35:
        print("Op basis van histogram: Bruine banaan")
    else:
        print("Op basis van histogram: onzeker")

args = sys.argv
if (len(args) < 2):
    print("run command: python video_demo.py 0 or video file name")
    sys.exit(0)
name = args[1]
if (len(args[1]) == 1):
    name = int(args[1])

stream = cv2.VideoCapture(name)

while True:
    ret, frame = stream.read()
    if not ret:
        print("unable to fetch frame")
        break
    results = model.detect([frame], verbose=0)

    # Visualize results
    r = results[0]
    masked_image = display_instances(frame, r['rois'], r['masks'], r['class_ids'],
                                     class_names, r['scores'])
    N = r['rois'].shape[0]
    masks = r['masks']
    resultaten = []
    for i in range(N):
        oppervlakteROI = (abs(r['rois'][i][0] - r['rois'][i][2])) * (abs(r['rois'][i][1] - r['rois'][i][3]))
        verhoudingROI = oppervlakteROI / (frame.shape[0] * frame.shape[1])

        if (r['class_ids'][i] == 47 and r['scores'][i] >= 0.90 and verhoudingROI > 0.10):
            print("Banaan gevonden (Verhouding ROI/Afbeelding = " + str(verhoudingROI) + ", Score: " + str(r['scores'][i]) + ")")

            im = frame.astype(np.uint32).copy()
            mask = masks[:, :, i]
            im = apply_mask_without_color(im, mask)
            for j in range(im.shape[2]):
                im[:, :, j] = im[:, :, j] * mask
            # im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2RGB)
            resultaten.append(im.astype(np.uint8))
            cv2.imshow("Resultaat van Mask R-CNN", im.astype(np.uint8))

        elif (r['class_ids'][i] == 47 and r['scores'][i] < 0.90 and verhoudingROI > 0.10):
            print("Banaan gevonden maar met onvoldoende zekerheid")

    for res in resultaten:
        res = cv2.resize(res, (200, 150), interpolation = cv2.INTER_AREA)
        histogramLijst = maakHistogram(res)
        interpreteerHistogram(histogramLijst)

    cv2.imshow("masked_image", masked_image)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break
stream.release()
cv2.destroyWindow("masked_image")