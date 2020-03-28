from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
import cv2
from matplotlib import colors
from skimage.color import rgb2gray
import os
import sys

# Root directory van het project: let op! Gebruik geen '\'  maar gebruik '/' in de padnaam
ROOT_DIR = os.path.abspath("C:/Python37/Scripts/Mask_RCNN/")

# Importeer de Mask R-CNN-architectuur
sys.path.append(ROOT_DIR)
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Import COCO config
import coco

# Directory om logfiles bij te houden
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Pad waar de pre-trained gewichten terug te vinden zijn, indien niet aanwezig worden deze gedownload (+/- 250 MB)
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory van afbeeldingen waarop detectie + segmentatie moet op worden uitgevoerd
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# ## Configuratie
# Er wordt een model gebruikt dat al getrained is op basis van de MS-COCO-dataset. De configuratie van dit model
# staat in de 'CocoConfig'-klasse in de file 'coco.py'.

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Laadt de gewichten van het getrainede model in
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


class Afbeelding:
    # Constructor voor afbeelding
    def __init__(self, pad, naam):
        self.image = cv2.imread(pad)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.name = naam
        self.size = self.image.shape[0] * self.image.shape[1]

    # Segmenteer de afbeelding op basis van Otsu's Thresholding
    def Segment_Otsus_Threshold(self):
        # Nog voor thresholding zelf: afbeelding afvlakken
        # Eerste parameter: maximum afvlakking, tweede parameter: max iteraties
        shifted = cv2.pyrMeanShiftFiltering(self.image, 25, 51)

        # Zet afgevlakte afbeelding om naar grijswaarden
        gray = cv2.cvtColor(shifted, cv2.COLOR_RGB2GRAY)

        # Voer Otsu's thresholding uit op afbeelding in grijswaarden
        threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Maak een masker aan op basis van de threshold-afbeelding
        mask = cv2.inRange(threshold, 0, 10)

        # Bitwise and of the mask and the original picture
        result = cv2.bitwise_and(self.image, self.image, mask=mask)

        return result, mask

    # Segmenteer de afbeelding op basis van K-means clustering
    def Segment_Kmeans(self):
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        image = image / 255  # pixelwaarden liggen nu tussen 0 en 1
        image_n = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
        image_n.shape
        kmeans = KMeans(n_clusters=2, init='k-means++').fit(image_n)
        pic2show = kmeans.cluster_centers_[kmeans.labels_]
        cluster_pic = pic2show.reshape(image.shape[0], image.shape[1], image.shape[2])
        if (kmeans.cluster_centers_[0][0] * 255 > 220 and kmeans.cluster_centers_[0][1] * 255 > 220 and
                kmeans.cluster_centers_[0][2] * 255 > 220):
            #print(str(kmeans.cluster_centers_[1][0] * 255) + ',' + str(kmeans.cluster_centers_[1][1] * 255) + ',' + str(
            #    kmeans.cluster_centers_[1][2] * 255))
            return kmeans.cluster_centers_[1] * 255
        else:
            #print(str(kmeans.cluster_centers_[0][0] * 255) + ',' + str(kmeans.cluster_centers_[0][1] * 255) + ',' + str(
            #    kmeans.cluster_centers_[0][2] * 255))
            return kmeans.cluster_centers_[0] * 255
        # plt.subplot(121), plt.imshow(image, cmap='gray')
        # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        # plt.subplot(122), plt.imshow(cluster_pic, cmap='gray')
        # plt.title('Clusters'), plt.xticks([]), plt.yticks([])
        # plt.imshow(cluster_pic)
        # plt.show()

    def segment_Mask_RCNN(self):
        # Run detection
        results = model.detect([self.image], verbose=0)
        r = results[0]

        # Zet volgende lijn uit commentaar om totaalresultaat te zien.
        # visualize.display_instances(self.image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

        # Aantal objecten gevonden op de afbeelding
        N = r['rois'].shape[0]
        masks = r['masks']
        resultaten = []
        for i in range (N):
            # Oppervlakte van de Region Of Interest (ROI)
            oppervlakteROI = (abs(r['rois'][i][0] - r['rois'][i][2])) * (abs(r['rois'][i][1] - r['rois'][i][3]))

            # Verhouding van de grootte van de ROI t.o.v. de afbeeldingsgrootte
            verhoudingROI = oppervlakteROI / self.size

            # Object moet klasseNr. 47 hebben (Banaan), een zekerheidsscore van minstens 90 % en de Region Of Interest
            # van het object moet minstens 10 % van de totale afbeelding beslaan
            if (r['class_ids'][i] == 47 and r['scores'][i] >= 0.90 and verhoudingROI > 0.10):
                print("Banaan gevonden (Verhouding ROI/Afbeelding = " + str(verhoudingROI) + ", Score: " + str(r['scores'][i]) + ")")

                # De volgende lijnen dienen om de afzonderlijke maskers op de
                # oorspronkelijke afbeelding te plakken
                masked_image = self.image.astype(np.uint32).copy()
                mask = masks[:, :, i]
                masked_image = visualize.apply_mask_without_color(masked_image, mask)
                for j in range(masked_image.shape[2]):
                    masked_image[:, :, j] = masked_image[:, :, j] * mask
                masked_image = cv2.cvtColor(masked_image.astype(np.uint8), cv2.COLOR_BGR2RGB)
                resultaten.append(masked_image)

            elif (r['class_ids'][i] == 47 and r['scores'][i] < 0.90 and verhoudingROI > 0.10):
                print("Banaan gevonden maar met onvoldoende zekerheid")

            elif (r['class_ids'][i] != 47):
                print("Object gevonden dat geen banaan is")
        return resultaten


    # Sobel Edge Detection ==> geen meerwaarde
    def detect_edges(self):
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        # converting to grayscale
        gray = rgb2gray(image)

        # defining the sobel filters: array van gewichten
        sobel_horizontal = np.array([np.array([1, 2, 1]), np.array([0, 0, 0]), np.array([-1, -2, -1])])
        print(sobel_horizontal, 'is a kernel for detecting horizontal edges')
        sobel_vertical = np.array([np.array([-1, 0, 1]), np.array([-2, 0, 2]), np.array([-1, 0, 1])])
        print(sobel_vertical, 'is a kernel for detecting vertical edges')
        kernel_laplace = np.array([np.array([1, 1, 1]), np.array([1, -8, 1]), np.array([1, 1, 1])])
        print(kernel_laplace, 'is a laplacian kernel')
        out_h = ndimage.convolve(gray, sobel_horizontal, mode='reflect')
        out_v = ndimage.convolve(gray, sobel_vertical, mode='reflect')
        out_l = ndimage.convolve(gray, kernel_laplace, mode='reflect')
        plt.subplot(121), plt.imshow(image, cmap='gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(out_l, cmap='gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        plt.show()

    # Canny Edge Detection ==> geen meerwaarde
    def detect_edges_Canny(self):
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        edges = cv2.Canny(image, 100, 200)

        plt.subplot(121), plt.imshow(image, cmap='gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(edges, cmap='gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

        plt.show()


class Banaan:
    # Resultaat wordt herschaald naar 200x150
    def __init__(self, resultaat, masker):
        self.mask = masker
        self.result = resultaat
        self.result = cv2.resize(self.result, (200, 150), interpolation = cv2.INTER_AREA)
        self.size = self.result.shape

    # Geeft een array van 3 histogrammatrices weer
    def maakHistogram(self):
        histogramLijst = []
        color = ('b', 'g', 'r')
        # Maak voor elke kleurcomponent een histogram = matrix met vorm 256 x 1
        # Hou histogrammen bij in histogramLijst ==> histogramLijst.append
        for i, col in enumerate(color):
            histogram = cv2.calcHist([self.result], [i], None, [256], [10, 256])
            histogramLijst.append(histogram)
            plt.plot(histogram, color=col)
            plt.xlim([0, 256])
        plt.show()
        return histogramLijst

    # Bepaalt kleur op basis van histogram
    def interpreteerHistogram(self, histogramLijst):
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
        #print("B: " + str(tellerBlauw) + " G: " + str(tellerGroen) + " R: " + str(tellerRood) + " Tot: " + str(tellerTot))
        #print("Verhouding G/Tot: " + str(verhoudingGroen))
        if tellerRood > tellerGroen and tellerRood > tellerBlauw and verhoudingGroen > 0.35:
            print("Op basis van histogram: gele banaan")
        elif tellerGroen > tellerBlauw and tellerGroen > tellerRood:
            print("Op basis van histogram: Groene banaan")
        elif tellerRood > tellerGroen and tellerRood > tellerBlauw and verhoudingGroen < 0.35:
            print("Op basis van histogram: Bruine banaan")
        else:
            print("Op basis van histogram: onzeker")

    # Maak 3D-plot van HSV-componenten
    def maak3Dplot(self):
        hsv_result = cv2.cvtColor(self.result, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv_result)
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1, projection="3d")
        pixel_colors = self.result.reshape((np.shape(self.result)[0] * np.shape(self.result)[1], 3))
        norm = colors.Normalize(vmin=-1., vmax=1.)
        norm.autoscale(pixel_colors)
        pixel_colors = norm(pixel_colors).tolist()
        axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
        axis.set_xlabel("Hue")
        axis.set_ylabel("Saturation")
        axis.set_zlabel("Value")
        plt.show()

    def interpreteerClusterCenters(self, clusterCenterAray):
        if 173 < clusterCenterAray[0] < 255 and 141 < clusterCenterAray[1] < 321 and 24 < clusterCenterAray[2] < 86:
            print('banaan is geel\n')
        elif 95 < clusterCenterAray[0] < 179 and 123 < clusterCenterAray[1] < 218 and 43 < clusterCenterAray[2] < 85:
            print('banaan is groen\n')
        elif 86 < clusterCenterAray[0] < 160 and 51 < clusterCenterAray[1] < 118 and 25 < clusterCenterAray[2] < 91:
            print('banaan is bruin\n')

# Deze loop vormt het hoofdprogramma: de afbeeldingen in directory 'images' worden 1 voor 1 ingelezen en geanalyseerd.
# Eerst wordt een segmentatietechniek uitgevoerd. Deze staan in de klasse Afbeelding.
# Daarna wordt een object van klasse Banaan aangemaakt.
for i in range (0, 61):
    # Voeg object van klasse 'Afbeelding' toe aan lijst afbeeldingen
    afb= Afbeelding("./images/" + str(i) + ".jpg", "banaan" + str(i))

    """ Zet uit commentaar om Otsu's thresholding en K-means cluserting uit te voeren
    resultaat, masker = afb.Segment_Otsus_Threshold()
    centerArray = afb.Segment_Kmeans()
    banaan = Banaan(resultaat, masker)
    histogramLijst = banaan.maakHistogram()
    
    print("afbeelding " + str(i))
    banaan.interpreteerHistogram(histogramLijst)
    banaan.interpreteerClusterCenters(centerArray)
    """

    print("Voor afbeelding " + str(i) + ": ")
    results = afb.segment_Mask_RCNN()
    for res in results:
        cv2.imshow("Resultaat van Mask R-CNN", res)
        cv2.waitKey(0)
        banaan = Banaan(res, None)
        histogramLijst = banaan.maakHistogram()
        banaan.interpreteerHistogram(histogramLijst)
    print("\n")


