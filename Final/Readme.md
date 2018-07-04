# **Few Shot Object Detection**
## Task 1
We use resnet to do classification on fashion-mnist.
## Task 2
In this task, we aim to make use of some sufficient labeled data to help learn some label which is lack of data.
Bowei implement the Matching net , Yi-lin implement about GNN (state-of-the-art) 2018 
GNN is better than matching net. e.g 20way-ten shot.(Matching net onle deserve about 5way-10shot)

## **Dataset**

### Task 1
Fashion-mnist. <br/>

### Task 2
CIFAR-100. <br/>

There are 80 class (base class) which have enough data while the other 20 classes (novel class) are lack of data.
I use pre-trained CNN to get better performance.

## **Requirement**
### Task 1
python == 3.6 <br/>
tensorflow == 1.6.0 <br/>
scikit-image == 0.14.0 <br/>


### Task 2

python == 3.6 <br/>
tensorflow == 1.6.0 <br/>
torchvision == 0.2.1 <br/>
scikit-image == 0.14.0 <br/>

OS == Linux system (Ubuntu 16.04LTS)

## **Execution**

### Task 1

#### **testing**

Test by default-setting: <br/>
`bash task1.sh [Fashion_MNIST_student/test] [output.csv]` <br/>


#### **training**

Train the model by default setting <br/>

`bash task1_train.sh [Fashion_MNIST_student/train] [Fashion_MNIST_student/test] [output.csv] ` <br/>


### Task 2


#### **testing**

Before testing, we need to download the models: <br/>
all downloaded model execution include in each .sh.

Test for 1-shot: <br/>
`bash main_1shot.sh [novel data directory] [test data directory] [output_name.csv]` <br/>
Test for 5-shot: <br/>
`bash main_5shot.sh [novel data directory] [test data directory] [output_name.csv]` <br/>
Test for 10-shot: <br/>
`bash main_10shot.sh [novel data directory] [test data directory] [output_name.csv]` <br/>

### Another model 

`Simple Relation net r_net.py , but the encoder of my relation net may change anthoer way to implement. `
`Prototypical net , loss can not go down after each training iteration.`

