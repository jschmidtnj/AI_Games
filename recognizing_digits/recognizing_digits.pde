/*

 Joshua Schmidt 2018

 MAIN TAB
 
 Original from Alasdair Turner (c) 2009
 Free software: you can redistribute this program and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 
 */


int totalTrain = 0;
int totalTest = 0;
int totalRight = 0;
float sucess = 0;
int testCard = 0;
int trainCard = 0;
boolean initial_train = true;
int initial_train_num = 50; //this is the amount it initially trains by, *500
boolean initial_test = true;
int initial_test_num = 30; //this is the amount it initially tests by
Card [] testing_set; // the set we use to train (2000)
Card [] training_set; // the set we use to train (8000)
ArrayList connec = new ArrayList();
ArrayList conStr = new ArrayList();
float LEARNING_RATE = 0.01;
float [] g_sigmoid = new float [200];

Network neuralnet;
Button trainB, testB;

void setup() {

  size(1000, 400);
  setupSigmoid();
  loadData();
  neuralnet = new Network(196, 49, 10);
  smooth();
  stroke(150);
  
  trainB = new Button(width*0.06, height*0.9, "Train");
  testB = new Button(width*0.11, height*0.9, "Test"); 
  if(initial_train){
    for(int i=0; i < initial_train_num; i++){
      mousePressed();
    }
    initial_train = false;
  }
  if(initial_test){
    for(int i=0; i < initial_test_num; i++){
      mousePressed();
    }
    initial_test = false;
  }
}

void draw() {

  background(255);
  neuralnet.display();
  
  fill(100);
  text("Test card: #" + testCard, width*0.18, height*0.89);
  text("Train card: " + trainCard, width*0.18, height*0.93);
  
  text("Total train: " + totalTrain, width*0.32, height*0.89);
  text("Total test: " + totalTest, width*0.32, height*0.93);
  
  if(totalTest>0) sucess = float(totalRight)/float(totalTest);
  text("Success rate: " + nfc(sucess, 2), width*0.44, height*0.89);
  text("Card label: " + testing_set[testCard].output, width*0.44, height*0.93);
  
  trainB.display();
  testB.display();
}

void mousePressed() {
  if (trainB.hover() || initial_train == true) {
    for (int i = 0; i < 500; i++) {
      trainCard = (int) floor(random(0, training_set.length));
      neuralnet.respond(training_set[trainCard]);
      neuralnet.train(training_set[trainCard].outputs);
      totalTrain++;
    }
  } else if (testB.hover() || initial_test == true){
    testCard = (int) floor(random(0, testing_set.length));
    neuralnet.respond(testing_set[testCard]);
    neuralnet.display();
    if(neuralnet.bestIndex == testing_set[testCard].output) totalRight ++;
    totalTest ++;
  }
  redraw();
}

class Button {

  PVector pos;
  String name;
  int radius = 20;

  Button(float _x, float _y, String _name) {
    pos = new PVector(_x, _y);
    name = _name;
  }

  void display() {
    if (hover()) fill(220);
    else noFill();
    stroke(150);
    ellipse(pos.x, pos.y, radius*2, radius*2);
    fill(150);
    text(name, pos.x-13, pos.y+4);
  }

  boolean hover() {
    PVector mouse = new PVector(mouseX, mouseY);
    if (mouse.dist(pos) < radius) return true;
    else return false;
  }
}

/*

Joshua Schmidt 2018

LOADING DATA

The data is a list of 10,000 handwritten digits resampled to a grid of 14x14 pixels by Alasdair Turner
The original set can be found here: http://yann.lecun.com/exdb/mnist/

*/

class Card { // This class contains all the functions to format and save the data

  float [] inputs;
  float [] outputs;
  int output;

  Card() {
    inputs = new float [196]; // the images are a grid of 14x14 pixels which makes for a total of 196
    outputs = new float[10]; // the number of possible outputs; from 0 to 9
  }

  void imageLoad(byte [] images, int offset) { // Images is an array of 1,960,000 bytes, each one representing a pixel (0-255) of the 10,000 * 14x14 (196) images
                                               // We know one image consists of 196 bytes so the location is: offset*196
    for (int i = 0; i < 196; i++) {
      inputs[i] = int(images[i+offset]) / 128.0 - 1.0; // We then store each pixel in the array inputs[] after converting it from (0 - 255) to (+1 - -1) as they vary on the greyscale 
    }
  }

  void labelLoad(byte [] labels, int offset) {  // Labels is an array of 10,000 bytes, each representing the answer of each image

    output = int(labels[offset]);
    
    for (int i = 0; i < 10; i++) {  // We then set the correct index in output[] to +1 if it corresponds to the ouput and -1 if not
      if (i == output) {
        outputs[i] = 1.0;
      } else {
        outputs[i] = -1.0;
      }
    }
  }
  
}

void loadData(){ // In this function we initialise all out data in two seperate arrays, training[] and test[]
  
  byte [] images = loadBytes("t10k-images-14x14.idx3-ubyte");
  byte [] labels = loadBytes("t10k-labels.idx1-ubyte");
  training_set = new Card [8000];
  int tr_pos = 0;
  testing_set = new Card [2000];
  int te_pos = 0;
  for (int i = 0; i < 10000; i++) {
    if (i % 5 != 0) { 
      training_set[tr_pos] = new Card();
      training_set[tr_pos].imageLoad(images, 16 + i * 196); // There is an offset of 16 bytes
      training_set[tr_pos].labelLoad(labels, 8 + i);  // There is an offset of 8 bytes
      tr_pos++;
    } else {
      testing_set[te_pos] = new Card();
      testing_set[te_pos].imageLoad(images, 16 + i * 196);  // There is an offset of 16 bytes 
      testing_set[te_pos].labelLoad(labels, 8 + i);  // There is an offset of 8 bytes
      te_pos++;
    }
  }
}

/*

 Joshua Schmidt 2018
 
 NETWORK
 
 This class is for the neural network, which is hard coded with three layers: input, hidden and output
 
 */

class Network {

  Neuron [] input_layer;
  Neuron [] hidden_layer;
  Neuron [] output_layer;
  int bestIndex = 0;

  Network(int inputs, int hidden, int outputs) {

    input_layer = new Neuron [inputs];
    hidden_layer = new Neuron [hidden];
    output_layer = new Neuron [outputs];

    for (int i = 0; i < input_layer.length; i++) {
      input_layer[i] = new Neuron();
    }

    for (int j = 0; j < hidden_layer.length; j++) {
      hidden_layer[j] = new Neuron(input_layer);
    }

    for (int k = 0; k < output_layer.length; k++) {
      output_layer[k] = new Neuron(hidden_layer);
    }
  }

  void respond(Card card) {

    for (int i = 0; i < input_layer.length; i++) {
      input_layer[i].output = card.inputs[i];
    }
    // now feed forward through the hidden layer
    for (int j = 0; j < hidden_layer.length; j++) {
      hidden_layer[j].respond();
    }
    for (int k = 0; k < output_layer.length; k++) {
      output_layer[k].respond();
    }
  }

  void display() {

    drawCon();

    // Draw the input layer
    for (int i = 0; i < input_layer.length; i++) {
      pushMatrix();
      translate(
        (i%14) * height / 20.0 + width * 0.05, 
        (i/14) * height / 20.0 + height * 0.13);
      input_layer[i].display();
      popMatrix();
    }

    // Draw the hidden layer
    for (int j = 0; j < hidden_layer.length; j++) {
      pushMatrix();
      translate(
        (j%7) * height / 20.0 + width * 0.53, 
        (j/7) * height / 20.0 + height * 0.32);
      hidden_layer[j].display();
      popMatrix();
    }

    // Draw the output layer
    float [] resp = new float [output_layer.length];
    float respTotal = 0.0;
    for (int k = 0; k < output_layer.length; k++) {
      resp[k] = output_layer[k].output;
      respTotal += resp[k]+1;
    }

    for (int k = 0; k < output_layer.length; k++) {
      pushMatrix();
      translate(
        width * 0.85, 
        (k%10) * height / 15.0 + height * 0.2);
      output_layer[k].display();
      fill(150);
      strokeWeight(sq(output_layer[k].output)/2);
      line(12, 0, 25, 0);
      text(k%10, 40, 5);
      text(nfc(((output_layer[k].output+1)/respTotal)*100, 2) + "%", 55, 5);
      popMatrix();
      strokeWeight(1);
    }
    float best = -1.0;
    for (int i =0; i < resp.length; i++) {
      if (resp[i]>best) {
        best = resp[i];
        bestIndex = i;
      }
    }
    stroke(255, 0, 0);
    noFill();
    ellipse(
      width * 0.85, (bestIndex%10) * height / 15.0 + height * 0.2, 
      25, 25);
  }

  void train(float [] outputs) {
    // adjust the output layer
    for (int k = 0; k < output_layer.length; k++) {
      output_layer[k].setError(outputs[k]);
      output_layer[k].train();
    }
    float best = -1.0;
    for (int i = 0; i < output_layer.length; i++) {
      if (output_layer[i].output > best) bestIndex = i;
    }
    // propagate back to the hidden layer
    for (int j = 0; j < hidden_layer.length; j++) {
      hidden_layer[j].train();
    }

    // The input layer doesn't learn: it is the input and only that
  }

  void drawCon() {

    for (int i = 0; i < hidden_layer.length; i++) {
      float [] res = hidden_layer[i].getStrength();
      stroke(200);
      strokeWeight(pow(10, res[1])/35);
      line(
        (i%7) * height / 20.0 + width * 0.53, 
        (i/7) * height / 20.0 + height * 0.32, 
        (int(res[0])%14) * height / 20.0 + width * 0.05, 
        (int(res[0])/14) * height / 20.0 + height * 0.13);
    }

    for (int i = 0; i < output_layer.length; i++) {
      float [] res = output_layer[i].getStrength();
      stroke(res[1]*200);
      strokeWeight(pow(10, res[1])/35);
      line(
        width * 0.85, 
        (i%10) * height / 15.0 + height * 0.2,
        (res[0]%7) * height / 20.0 + width * 0.53, 
        (res[0]/7) * height / 20.0 + height * 0.32);
    }
    strokeWeight(1);
  }
}

/*

 Joshua Schmidt 2018

 NEURON
 
 This class is for the neural network, which is hard coded with three layers: input, hidden and output
 
 */


class Neuron {

  Neuron [] inputs; // Strores the neurons from the previous layer
  float [] weights;
  float output;
  float error;

  Neuron() {
    error = 0.0;
  }

  Neuron(Neuron [] p_inputs) {

    inputs = new Neuron [p_inputs.length];
    weights = new float [p_inputs.length];
    error = 0.0;
    for (int i = 0; i < inputs.length; i++) {
      inputs[i] = p_inputs[i];
      weights[i] = random(-1.0, 1.0);
    }
  }

  void respond() {

    float input = 0.0;
    for (int i = 0; i < inputs.length; i++) {
      input += inputs[i].output * weights[i];
    }
    output = lookupSigmoid(input);
    error = 0.0;
  }

  void setError(float desired) {
    error = desired - output;
  }

  void train() {

    float delta =(1.0 - output) * (1.0 + output) *
      error * LEARNING_RATE;
    for (int i = 0; i < inputs.length; i++) {
      inputs[i].error += weights[i] * error;
      weights[i] += inputs[i].output * delta;
    }
  }

  void display() {
    stroke(200);
    fill(128 * (1 - output));
    ellipse(0, 0, 16, 16);
  }

  float [] getStrength() {
    float ind = 0.0;
    float str = 0.0;
    for (int i = 0; i < weights.length; i++) {
      if (weights[i] > str) {
        ind = i; 
        str = weights[i];
      }
    }
    float [] a = {ind, str};
    return a;
  }
}

/*

 Joshua Schmidt 2018

 SIGMOID
 Activation function
 
 A sigmoid function is the neuron's response to inputs the sigmoidal response ranges from -1.0 to 1.0
 For example, the weighted sum of inputs might be "2.1" our response would be lookupSigmoid(2.1) = 0.970
 This is a look up table for sigmoid (neural response) values which is valid from -5.0 to 5.0
 
 */

void setupSigmoid() {
  
  for (int i = 0; i < 200; i++) {
    float x = (i / 20.0) - 5.0;
    g_sigmoid[i] = 2.0 / (1.0 + exp(-2.0 * x)) - 1.0;
  }
}

// once the sigmoid has been set up, this function accesses it:
float lookupSigmoid(float x) {
  
  return g_sigmoid[constrain((int) floor((x + 5.0) * 20.0), 0, 199)];
}
