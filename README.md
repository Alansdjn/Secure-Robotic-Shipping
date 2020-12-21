# **Handbook**

## **Project Title:**

**Secure Shipping Using Robotic Delivery System**

## **Please see Code files in GitHub** :

[https://github.com/Jiapei-Yang/Secure-Robotic-Shipping](https://github.com/Jiapei-Yang/Secure-Robotic-Shipping)

## **Acknowledgments** :

WangBenYan. [https://blog.csdn.net/qq\_18808965/article/details/90262113](https://blog.csdn.net/qq_18808965/article/details/90262113)

Harshvardhan Gupta. [https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch](https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch)

Li, W, Zhao, R, and Wang, X. &quot;Human Re-identification with Transferred Metric Learning.&quot; Asian conference on computer vision (2012): 31-44. Web.

## **Brief Introduction:**

**This document is divided into three parts:**

1. Explanation of files in project _Secure Shipping Using Robotic Delivery System_.

2. Guidance of simple experiment.

3. Guidance of system implementation (demo).

**To get the dataset _CUHK01_, visit**

[https://www.ee.cuhk.edu.hk/~xgwang/CUHK\_identification.html](https://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html)

Fig. 1 and Fig. 2 show how the system work. For more details please read the project&#39;s paper.

![](https://github.com/Jiapei-Yang/Secure-Robotic-Shipping/blob/master/Image/fig1.png)

Figure 1. The proposed robotic delivery system

![](https://github.com/Jiapei-Yang/Secure-Robotic-Shipping/blob/master/Image/fig2.png)

Figure 2. Details of off-line authentication methods

Fig. 3 illustrates the implementation of the system:

![](https://github.com/Jiapei-Yang/Secure-Robotic-Shipping/blob/master/Image/fig3.png)

Figure 3. The proposed system implementation

## **1. Explanation of files**

### Ⅰ. Proverif

This folder is for security analysis in the cooperative authentication of our system.

**Verifier.pv** : Use Proverif to analyse security of the communication protocol in our cooperative authentication. This includes analysing data disclosure and forged attacks.

**Result** : Result of verifier.pv. analyse

### Ⅱ. Non-cooperative part in Jupyter

This folder includes implement of our system&#39;s non-cooperative authentication in Jupyter platform. It is much more convenient to test our non-cooperative part&#39;s performance in this emulation with Jupyter.

**Data** : Dataset after pre-processing. It includes four sub-folders, and each one is corresponding to a Jupyter file. We can run classify.py to do pre-processing to add the training set folder and the test set folder here.

**classify.py** : Pre-processing code. It splits the original data into a training set and a test set.

**train.ipynb** : Model training code. It stores the trained model and names it as net\_test.pth.

**test\_CMC.ipynb** : Model testing code. It contains the test result (CMC figure).

**re\_identify\_prepare.ipynb** : Calculate the max distance between any two images of the client.

**re\_identify\_robot.ipynb** : Calculate the distance between the captured image of pedestrian and the client&#39;s image, and judge whether they are from the same person.

**net\_test.pth** : The saved model. We can generate it in train.ipynb.

### Ⅲ. Cooperative part

This folder includes implement of our system&#39;s cooperative authentication. We utilize three Python code files in Python 2.7 for cooperative authentication.

&quot; **true-client.py**&quot;: Run this file in the client. It controls behaviours of client in cooperative authentication: Client sends a request by TCP socket protocol to the server, and waits for a reply. Upon receiving the answer from the server, client derives public key and nonce in it. By using the nonce, client generates an authentication information, encrypts it with the public key and generates the QR code. Finally, the client shows the QR code in the screen.

&quot; **true-server.py**&quot;: Run this file in the server. It controls behaviours of server in cooperative authentication: Server firstly waits for the request from clients by TCP socket protocol. Upon receiving it, the server authenticates the identity of the client, and generate some nonce and a pair of one-time-use public key and private key. Then, server sends the nonce and public key back to the client. Next, server generates the same authentication information of that in the client, encrypts it with the public key and transfers it to the robot. Finally, server transfers file of the private key and the serial number to the robot, too.

&quot; **true-robot.py**&quot;: Run this file in the robot. It controls behaviours of robot in cooperative authentication: First, robot receives information from the server, and derives the serial number. With the help of it, the robot is able to quickly recognize the identity of the client after QR code scanning of him. If the scanning and recognition are successful, the robot decrypts two encrypted messages from the server and the client, and compare the plaintexts. As soon as they match, the robot finishes the delivery work. If the scanning process fails for 5 times, robot turns to non-cooperative mode, and waits for the signal of re-identification of the client, and once again tries to scan the QR code. If robot fails the QR code scanning part again, timeout of the delivery job occurs, and we stop the express.

### Ⅳ. Non-cooperative part

This folder includes implement of our system&#39;s non-cooperative authentication. We utilize 6 Python code files in Python 3.7 for non-cooperative authentication. 4 files are in the server and the other 2 runs in the robot.

**classify.py** : Run this file in the server. In dataset pre-processing, it splits the original data into a training set and a test set.

**train.py** : Run this file in the server. This code file trains the model of person re-identification. It randomly reads two images from the same or different person, and makes the distance between them larger for different people and closer for the same identity. Then we store the trained model.

**test\_CMC.py** : Run this file in the server. It assesses the model by drawing the Cumulative Match Curve (CMC) and comparing with other algorithm&#39;s performance.

**re\_identify\_prepare.py** : Run this file in the server. It calculates the max distance between any two images of the client. This value is used as a separate line: if any two pictures&#39; distance is closer than this, they are seen from the same person.

**capture.py** : Run this file in the robot. It directs robot to capture the image, do pedestrian detection, and resize the person&#39;s image.

**re\_identify\_robot.py** : Run this file in the robot. Robot calculates the distance between two images: One is the client&#39;s image, and the other is the captured image of pedestrian from capture.py. Compare this distance to the max distance from re\_identify\_prepare.py. If this distance is smaller, the two images are from the same person, which means robot has found the client.

## **2. Guidance of simple experiment**

### Ⅰ. Formal Security Analysis in Cooperative Authentication

Put verifier.pv in folder that contains proverif.exe.

Open command (CMD).

Change the current working directory into the folder that has proverif.exe.

Insert the command &quot;_proverif verifier.pv_&quot; and press enter.

### Ⅱ. Non-cooperative authentication&#39;s simulation in Jupyter

Put the CUHK01 dataset under &quot;/data/&quot;, and run classify.py to add the training set and test set open any Jupyter file and run it directly. However, training and testing could be very time-consuming. We can also see the result of previous running directly in Jupyter files.

This part&#39;s work is briefly introduced in the video:

https://www.youtube.com/watch?v=zgD6tC4vGLM

## **3. Guidance of system implementation (demo).**

We will introduce the environmental requirements, and how to run the demo.

### Ⅰ. Environmental requirements:

#### **a. Cooperative off-line authentication:**

Table 1 shows the implementation environment of cooperative authentication. We can use two laptops as the client and the server, and Turtlebot3 as the robot. Here we install Ubuntu 16.04 in the client and the server and Ubuntu mate 16.04 in the robot. Then utilize ROS kinetic, which is recommended in Turtlebot3 and supports Python 2, in the server and the robot and set the server as the master.

![](https://github.com/Jiapei-Yang/Secure-Robotic-Shipping/blob/master/Image/tab1.png)

Table 1. Implementation environment of cooperative authentication

#### **b. Non-cooperative off-line authentication:**

The following table shows the implementation environment of non-cooperative authentication. We use the same devices of the cooperative part, but focus on the server and the robot. In addition, _virtualenv_ is utilized for building the Python 3 virtual environment in two devices, and we use Python 3 to fulfil requirement of some libraries in the area of computer vision. To train and test our model we utilize CUHK01 dataset.

![](https://github.com/Jiapei-Yang/Secure-Robotic-Shipping/blob/master/Image/tab2.png)

Table 2. Implementation environment of non-cooperative authentication

### Ⅱ. Detailed operation

#### **a. Cooperative off-line authentication demo:**

We can see the implementation of cooperative authentication in the video:

[_https://www.youtube.com/watch?v=-cANuZxD9uQ_](https://www.youtube.com/watch?v=-cANuZxD9uQ)

##### **Step C1** : Prepare work:

Set the environment based on Python 2 in Table 5.a.

Use ROS to create a map of the workplace.

(see https://emanual.robotis.com/docs/en/platform/turtlebot3/slam/)

Change the current working directory to &quot;/home/username/Desktop/project&quot; in three terminals.

Put true-client.py in the client.

Put true-robot.py in the robot.

Put true-server.py in the server.

##### **Step C2** : Launch ROS and robot equipment:

In the server, launch ROS: _roscore_

In the robot, launch the robot: _roslaunch turtlebot3\_bringup turtlebot3\_robot.launch_

##### **Step C3** : Run scripts:

Server launches a new terminal and start services: _python true-server.py_

Robot launches a new terminal and start services: _python true-robot.py_

Next, client sends the request: _python true-client.py_

As a result, the robot shows &quot;please input Y when robot arrives:&quot;, which means server has received the client&#39;s request and distributed information to the client and the robot.

##### **Step C4** : navigation:

Here we make a manual navigation.

Server launches a new terminal and insert:

_export TURTLEBOT3\_MODEL=waffle\_pi_

Then run the command to launch _Rviz_ for navigation:

_roslaunch turtlebot3\_navigation turtlebot3\_navigation.launch map\_file:=$HOME/true-map.yaml_

Press &quot;2D Pose Estimate&quot; to correct start position, and &quot;2D Nav Goal&quot; to select aimed destination. Robot plans the path and go there automatically as shown in Fig. 4.

![](https://github.com/Jiapei-Yang/Secure-Robotic-Shipping/blob/master/Image/fig4.png)

Figure 4. Navigation in the implementation of cooperative authentication

##### **Step C5** : QR code scan:

Upon arriving the destination, insert &quot;Y&quot; in the robot as a signal of completed navigation. Then, the robot automatically tries to scan QR code shown by the client. The scanning work succeeds and the QR code is authenticated, so the robot shows &quot;matched!&quot; and complete the delivery as shown in Fig. 5 and Fig. 6.

![](https://github.com/Jiapei-Yang/Secure-Robotic-Shipping/blob/master/Image/fig5.png)

Figure 5. QR code scanning in the implementation of cooperative authentication

![](https://github.com/Jiapei-Yang/Secure-Robotic-Shipping/blob/master/Image/fig6.png)

Figure 6. Result of QR code scanning in cooperative authentication

#### **b. Cooperative off-line authentication demo:**

This part executes cooperative authentication twice: A failed one before shifting to the non-cooperative part, and the other successful one when the robot recognizes the client and shifts the mode back.

We can see a video of the implementation above:

[_https://www.youtube.com/watch?v=lN5tngrdmds_](https://www.youtube.com/watch?v=lN5tngrdmds)

##### **Step N1:** Prepare work

Do prepare work in Step C1 in the implementation of cooperative authentication.

Set the virtual environment based on Python 3 in Table 5.b.

(see https://code-maven.com/slides/python/virtualev-python3)

Put capture.py and re\_identify\_robot.py in the robot.

Put classify.py, train.py, test\_CMC.py, re\_identify\_prepare.py in the server.

Put dataset (CUHK01) as &quot;/data/CUHK01/campus&quot;.

Next, server activates python3 environment and does pre-processing:

_source ~/venv3/bin/activate_

_python classify.py_

To finish the pre-processing work in the server, put images of the client into &quot;/data/reid\_prepare&quot;, and copy one of them into &quot;/data/reid\_robot&quot; in the robot as the comparison image. Therefore, we have four folders to store dataset:

/data/training\_set: Training set. Stored in the server.

/data/test\_set: Testing set. Stored in the server.

/data/reid\_prepare: All images of the client. Stored in the server.

/data/reid\_robot: An image of the client (a captured photo of pedestrian will be added here later).

##### **Step N2:** Model training

Then, train a model in the server:

_python train.py_

The model is stored as &quot;/net\_test.pth&quot;, and copy it to the robot.

##### **Step N3:** Optional Model testing

Run the code in the server to plot CMC in test1.jpg:

_python test\_CMC.py_

To get the separate line used in the final comparison, run the code in the server:

_python re\_identify\_prepare.py_

##### **Step N4:** Capturing images

Do the cooperative authentication and make the QR code scanning unit fail as shown in Fig. 7

Robots runs:

_python capture.py_

In this way, we successfully captured an image of pedestrian, did pedestrian detection in it, resize the person&#39;s image and stored it in &quot;/data/reid\_robot/&quot;.

![](https://github.com/Jiapei-Yang/Secure-Robotic-Shipping/blob/master/Image/fig7.png)

Figure 7. Failed QR code scanning and shifting to non-cooperative mode

##### **Step N5:** Person re-identification

Next, robot runs the command to compare the similarity of the captured image and client&#39;s image in &quot;/data/reid\_robot/&quot;.

_python re\_identify\_robot.py_

If the output is &quot;same person&quot;, robot successfully recognized the client, and the system switches back to cooperative mode as shown in Fig. 8 and Fig. 9.

##### **Step N6:** QR code scanning again

Finally, input any character in the robot&#39;s terminal that runs true-robot.py so it can start to scan the QR code again and match as shown in Fig. 10 and Fig. 11.

![](https://github.com/Jiapei-Yang/Secure-Robotic-Shipping/blob/master/Image/fig8.png)

Figure 8. Person detection and re-identification

![](https://github.com/Jiapei-Yang/Secure-Robotic-Shipping/blob/master/Image/fig9.png)

Figure 9. Result of person detection and re-identification

![](https://github.com/Jiapei-Yang/Secure-Robotic-Shipping/blob/master/Image/fig10.png)

Figure 10. QR code scanning again

![](https://github.com/Jiapei-Yang/Secure-Robotic-Shipping/blob/master/Image/fig11.png)

Figure 11. Result of QR code scanning again
