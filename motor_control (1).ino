const int inA1=12;
const int inB1=11;
const int inA2=7;
const int inB2=8;

const int PWM1=5;
const int PWM2=3;

void setup() {
  pinMode(inA1, INPUT);
  pinMode(inB2, INPUT);
  pinMode(inA2, INPUT);
  pinMode(inB2, INPUT);
  pinMode(PWM1, INPUT);
  pinMode(PWM2, INPUT);
}

void loop() {

  //Left wheel forward
  digitalWrite(inA1, 1);
  digitalWrite(inB1, 0);
  analogWrite(PWM1, 255);
  
  delay(5000);

  //Left wheel reverse
  digitalWrite(inA1, 0);
  digitalWrite(inB1, 1);
  analogWrite(PWM1, 255);
  
  delay(5000);

  //Right wheel forward
  digitalWrite(inA2, 1);
  digitalWrite(inB2, 0);
  analogWrite(PWM2, 255);
  
  delay(5000);

  //Right wheel reverse
  digitalWrite(inA2, 0);
  digitalWrite(inB2, 1);
  analogWrite(PWM2, 255);

}


void left_forward(int speed){
  digitalWrite(inA1, 1);
  digitalWrite(inB1, 0);
  analogWrite(PWM1, speed);
}
