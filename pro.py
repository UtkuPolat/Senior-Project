from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import base64, os
import numpy as np
import time
import cv2
import os, random
import argparse

from yolo import YOLO

route = os.path.dirname(os.path.abspath(__file__))
des = os.path.join(route, "static", "test.jpeg")

app = Flask(__name__)

UPLOAD_FOLDER = r"/Users/utkupolat/Downloads/Web/Object-Detection-Application-master/Object/Detection/App/images"
DEVICE = "cuda"
MODEL = None


@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/cam')
def Camera():
    return render_template('cam.html')


@app.route('/saveimage', methods=['POST'])
def saveImage():
    data_url = request.values['imageBase64']
    image_encoded = data_url.split(',')[1]
    body = base64.b64decode(image_encoded.encode('utf-8'))
    file = open(des, "wb")
    file.write(body)
    return "ok"


imagen = " "


@app.route("/image", methods=["GET", "POST"])
def Image():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)
            global imagen
            imagen = image_file.filename
            return render_template("image.html", metin="Dosya Yüklendi ✔")
    return render_template("image.html", metin="Dosya Yüklenmedi ❌")


@app.route('/webcam')
def WebCam():
    return render_template('webcam.html')


@app.route('/process')
def process():
    return render_template('process.html')


@app.route('/process2')
def process2():
    return render_template('process2.html')


@app.route('/showimage')
def showImage():
    print("in showImage")
    return render_template('output.html')


@app.route('/output')
def output():
    dam = 0
    # load the COCO class labels our YOLO model was trained on
    labelsPath = "coco.names"
    LABELS = open(labelsPath).read().strip().split("\n")
    # initialize a list of colors to represent each possible class label
    np.random.seed(11)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                               dtype="uint8")
    # derive the paths to the YOLO weights and model configuration
    weightsPath = "models/mask.weights"
    configPath = "models/mask.cfg"
    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    # load our input image and grab its spatial dimensions
    # ************************************\
    image = cv2.imread("static/test.jpeg")
    # ************************************
    (H, W) = image.shape[:2]
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))
    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > 0.5:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,
                            0.3)
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

    # if dam==1:
    # 	dest=os.path.join(route,"static","result.jpeg")
    # 	os.remove(dest1)
    # 	cv2.imwrite(dest,image)
    # 	dam=0
    # 	imgval="../static/result.jpeg"
    # else:
    # 	dest1=os.path.join(route,"static","result1.jpeg")
    # 	os.remove(dest)
    # 	cv2.imwrite(dest1,image)
    # 	dam=1
    # 	imgval="../static/result1.jpeg"

    a = random.randrange(1, 5000, 1)
    b = str(a) + ".jpeg"

    dest = os.path.join(route, "static", b)
    cv2.imwrite(dest, image)
    imgval = r"static/{}".format(b)

    # data=session.query(Shop).filter_by(name=LABELS[classIDs[0]]).first()
    return render_template('output.html', imgval=imgval)


@app.route('/output2')
def output2():
    dam = 0
    # load the COCO class labels our YOLO model was trained on
    labelsPath = "coco.names"
    LABELS = open(labelsPath).read().strip().split("\n")
    # initialize a list of colors to represent each possible class label
    np.random.seed(11)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    # derive the paths to the YOLO weights and model configuration
    weightsPath = r"models\mask.weights"
    configPath = r"models\mask.cfg"
    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    # load our input image and grab its spatial dimensions
    # ************************************\
    image = cv2.imread("images/" + imagen)
    # ***********************************
    (H, W) = image.shape[:2]
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    print("\n")
    print(ln)
    print(net.getUnconnectedOutLayers())
    print("\n")
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    print(ln)
    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))
    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > 0.5:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    a = random.randrange(1, 5000, 1)
    b = str(a) + ".jpg"

    dest = os.path.join(route, "static", b)
    cv2.imwrite(dest, image)
    imgval = r"static\{}".format(b)

    # data=session.query(Shop).filter_by(name=LABELS[classIDs[0]]).first()
    return render_template('output2.html', imgval=imgval)


classes = ["maskeli", "maskesiz", "maskesiz"]


@app.route('/webcam')
def Webcam():
    yolo = YOLO("models/mask.cfg", "models/mask.weights", classes)
    colors = [(0, 255, 0), (0, 165, 255), (0, 0, 255)]

    print("starting webcam...")
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        width, height, inference_time, results = yolo.inference(frame)
        key = cv2.waitKey(50)
        for detection in results:
            id, name, confidence, x, y, w, h = detection
            cx = x + (w / 2)
            cy = y + (h / 2)

            # draw a bounding box rectangle and label on the image
            color = colors[id]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "%s (%s)" % (name, round(confidence, 2))
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

        cv2.imshow("preview", frame)

        rval, frame = vc.read()

        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break

    cv2.destroyWindow("preview")
    vc.release()

    return "ok"

if __name__ == '__main__':
    app.debug = True
    app.run(port=8088)
