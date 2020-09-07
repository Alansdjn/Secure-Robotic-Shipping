**README:**
This work is available in Github:
https://github.com/Jiapei-Yang/Secure-Robotic-Shipping

**Secure Shipping Using Robotic Delivery System**

--------------------------------------------------------------------------------------

This work is the code section of the dissertation &quot;Secure Shipping Using Robotic Delivery System&quot;. It includes some important figures, all codes and running results.

The following figure shows how the system work:

![Alt text](https://github.com/Jiapei-Yang/Secure-Robotic-Shipping/raw/master/fig1.jpg)

--------------------------------------------------------------------------------------
**Explanation of files:**

Ⅰ.Proverif

　　This folder is for security analysis in the cooperative authentication of our system.

　　**Verifier.pv** : Use Proverif to analyse security of the communication protocol in our cooperative authentication. This includes analysing data disclosure and forged attacks.

　　**Result** : Result of verifier.pv.


Ⅱ.Cooperative part

　　This folder includes implement of our system&#39;s cooperative authentication.

　　**true-client.py** : Code in client&#39;s terminal.

　　**true-server.py** : Code in server&#39;s terminal.

　　**true-robot.py** : Code in robot&#39;s terminal.


Ⅲ. Noncooperative part

　　This folder includes implement of our system&#39;s noncooperative authentication.

　　**classify.py** : Pre-processing of the dataset. Run this in the server.

　　**train.py** : Do model training for the aim of person re-identification. Run this in the server.

　　**test\_CMC.py** : Test the performance of the model. Run this in the server.

　　**re\_identify\_prepare.py** : Calculate the max distance between any two images of the client. This value is used as a separate line: if any two pictures&#39; distance is　closer than this, they are seen from the same person. Run this in the server.

　　**capture.py** : Robot captures the image, do pedestrian detection in it, and resize the person&#39;s image. Run this in the robot.

　　**re\_identify\_robot.py** : Robot calculates the distance between two images: One is the client&#39;s image, and the other is the captured image of pedestrian from capture.py. Compare this distance to the max distance from re\_identify\_prepare.py. If this distance is smaller, the two images are from the same person, which means robot has found the client. Run this in the robot.


Ⅳ. Noncooperative part in Jupyter

　　This folder includes implement of our system&#39;s noncooperative authentication in Jupyter platform. It is much more convenient to test our noncooperative part&#39;s performance in this emulation with Jupyter.

　　**Data** : Dataset after pre-processing. It includes four sub-folders, and each one is corresponding to a Jupyter file.

　　**train.ipynb** : Model training code. It stores the trained model and names it as net\_test.pth.

　　**test\_CMC.ipynb** : Model testing code. It contains the test result (CMC figure).

　　**re\_identify\_prepare.ipynb** : Calculate the max distance between any two images of the client.

　　**re\_identify\_robot.ipynb** : Calculate the distance between the captured image of pedestrian and the client&#39;s image, and judge whether they are from the same person.

　　**net\_test.pth** : Saved model.


--------------------------------------------------------------------------------------
**Guidance of simple experiment**

Ⅰ. Formal Security Analysis in Cooperative Authentication

　　Put verifier.pv in folder that contains proverif.exe.

　　Open command (CMD).

　　Change the current working directory into the folder that has proverif.exe.

　　Insert the command &quot;_proverif verifier.pv_&quot; and press enter.

Ⅱ. Noncooperative authentication&#39;s simulation in Jupyter

　　Put the CUHK01 dataset under &quot;/data/&quot;. open any Jupyter file and run it directly. However, training and testing could be very time-consuming.

　　This part&#39;s work is briefly introduced in the video:

　　https://www.youtube.com/watch?v=zgD6tC4vGLM

--------------------------------------------------------------------------------------
**Guidance for the whole system running in reality**

The figure shows how our system work in reality. We will introduce the environmental requirements, and introduce how to run each step of in detail:

![Alt text](https://github.com/Jiapei-Yang/Secure-Robotic-Shipping/blob/master/fig2.JPG?raw=true)

Ⅰ. Environmental requirements:

a. Cooperative off-line authentication:

　　Client : Laptop/PC (Ubuntu 16.04)

　　　　　　　Python 2.7

　　　　　　　IP address: 192.168.43.X

　　　　　　　Server: Laptop/PC (Ubuntu 16.04)

　　　　　　　Python 2.7

　　　　　　　Ros kinetic (Set as master)

　　　　　　　IP address: 192.168.43.97

　　Robot: 　Turtlebot3 (Ubuntu mate 16.04)

　　　　　　　Python 2.7

　　　　　　　Ros kinetic

　　　　　　　IP address: 192.168.43.74

b. Non-cooperative off-line authentication:

　　Server: 　Laptop/PC (Ubuntu 16.04)

　　　　　　　Python 3.7(Use _virtualenv_ to build a python3 virtual environment)

　　　　　　　Dataset: ./data/CUHK01/campus

　　　　　　　Robot: Turtlebot3 (Ubuntu mate 16.04)

　　　　　　　Python 3.7 (Use _virtualenv_ to build a python3 virtual environment)

Ⅱ. Detailed operation

**a.Setup environment.**

　　Setup environment based on python2 for cooperative off-line authentication.

　　Use ROS to create a map.

　　(see https://emanual.robotis.com/docs/en/platform/turtlebot3/slam/)

　　Use _virtualenv_ to setup python3 environment for noncooperative off-line authentication.

　　(see https://code-maven.com/slides/python/virtualev-python3)

**b. Pre-processing and model training.**

　　Then, change the current working directory to &quot;/home/username/Desktop/project&quot; in three terminals.

　　Put **true-client.py** in the client.

　　Put **true-robot.py** , **capture.py** and **re\_identify\_robot.py** in the robot.

　　Put **true-server.py** , **classify.py** , **train.py** , **test\_CMC.py** , **re\_identify\_prepare.py** in the server.

　　Put dataset (CUHK01) as &quot;/data/CUHK01/campus&quot;.

　　Next, server runs the command a new terminal to activate python3 environment:

_source ~/venv3/bin/activate_

　　Then Server runs the code for pre-processing:

_python classify.py_

　　To finish the pre-processing work in the server, put images of the client into &quot;/data/reid\_prepare&quot;, and copy one of them into &quot;/data/reid\_robot&quot; in the robot as the comparison image. Therefore, we have four folders to store dataset:

　　**/data/training\_set** : Training set. Stored in the server.

　　**/data/test\_set** : Testing set. Stored in the server.

　　**/data/reid\_prepare** : All images of the client. Stored in the server.

　　**/data/reid\_robot** : An image of the client (a captured photo of pedestrian will be added here later).

　　Then, train a model in the server:

_python train.py_

　　The model is stored as &quot;/net\_test.pth&quot;, and copy it to the robot.

　　**An optional step: Reidentification Model Test in Noncooperative Authentication: Run the code in the server to plot CMC in test1.jpg:**

_python test\_CMC.py_

　　To get the separate line used in the final comparison, run the code in the server:

_python re\_identify\_prepare.py_

c. Cooperative authentication

　　Then server launches a new terminal and start ROS:

_roscore_

　　Robot launches a new terminal and start equipment activities:

_roslaunch turtlebot3\_bringup turtlebot3\_robot.launch_

　　Server launches a new terminal and start services:

_python true-server.py_

　　Robot launches a new terminal and start services:

_python true-robot.py_

　　Client launches a new terminal and start services:

_python true-client.py_.

　　As a result, the robot shows &quot;please input Y when robot arrives:&quot;.

　　Here we make a manual navigation:

　　Server launches a new terminal and insert:

_export TURTLEBOT3\_MODEL=waffle\_pi_

　　Then run the command to launch _Rviz_ for navigation:

_roslaunch turtlebot3\_navigation turtlebot3\_navigation.launch map\_file:=$HOME/true-map.yaml_

　　Press &quot;2D Pose Estimate&quot; to correct original position, and &quot;2D Nav Goal&quot; to select aimed destination. Robot will plan the path and go there automatically.

　　Upon arriving the destination, insert &quot;Y&quot; in the robot. The robot will automatically try to scan QR code shown by the client. If the scanning work succeed and the QR code is authenticated, the robot will show &quot;matched!&quot; and complete the delivery. Otherwise, robot presents &quot;QRcode scan failed. Shift to noncooperative mode.&quot;, then our system switches to noncooperative authentication.

　　You can see the implementation of cooperative authentication in the video:

　　[_https://www.youtube.com/watch?v=-cANuZxD9uQ_](https://www.youtube.com/watch?v=-cANuZxD9uQ)

d. Noncooperative authentication

　　When we come into this stage, robot launches a new terminal and setup python3 environment:

_source ~/venv3/bin/activate_

　　In this way, we successfully captured an image of pedestrian, did pedestrian detection in it, resize the person&#39;s image and stored it in &quot;/data/reid\_robot/&quot;.

　　Next, robot runs the command to compare the similarity of the captured image and client&#39;s image in &quot;/data/reid\_robot/&quot;.

_python re\_identify\_robot.py_

　　If the output is &quot;same person&quot;, robot successfully recognised the client, and the system switches back to cooperative mode.

　　You can see a video of the implementation above:

　　[_https://www.youtube.com/watch?v=lN5tngrdmds_](https://www.youtube.com/watch?v=lN5tngrdmds)

e. Cooperative authentication once again

　　Insert any character in the robot&#39;s terminal that is running:

_python true-robot.py_

　　In this way we shift back to cooperative mode and robots tries to scan QR code.

　　We could set the max times of this kind of switch by changing &quot;max\_time&quot; in **true\_robot.py**. The default is 2. Therefore, if robot&#39;s QR code scanning period failed again, it cancels the delivery work and presents &quot;Delivery failed&quot;.
