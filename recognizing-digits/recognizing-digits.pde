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
