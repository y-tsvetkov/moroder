#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <Servo.h>
#define BNO055_SAMPLERATE_DELAY_MS (100)
Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28);

//defining max deflection for coxa, femur, tibia servos in radians
const float femurBound = 1;
const float tibiaBound = 2;
const float coxaBound = 1;

//network architecture
static const int inputNeurons = 5;
static const int hiddenNeurons = 8;
static const int outputNeurons = 12;

static const int hiddenWSize = hiddenNeurons * inputNeurons;
static const int outputWSize = outputNeurons * hiddenNeurons;
float timeStepLength = 1000 / 240; // the time between NN calculations must be roughly a factor of the default PyBullet time step
float timeSteps = 0;
float startTime, endTime, timeToNextFactor;

//upper and lower bounds for servo movement. For the left servos, their movement (and thus offsets) is inverted
float ubounds[] = {coxaBound, femurBound, 3.14, coxaBound, femurBound, 3.14, coxaBound, femurBound, tibiaBound, coxaBound, femurBound, tibiaBound};
float lbounds[] = { -coxaBound, -femurBound, 3.14 - tibiaBound, -coxaBound, -femurBound, 3.14 - tibiaBound, -coxaBound, -femurBound, 0, -coxaBound, -femurBound, 0};
float offsets[] = {0, 0, 0,  0, 0, 0,  0, 0, 0,   0, 0, 0}; //in case a servo does not perfectly follow the

int servopins[] = {2, 23, 38,   3, 22, 37,   4, 21, 36,   5, 20, 35}; //Front right, back right, front left and back left legs respectively
Servo servoarray[12];

//This class loads the parameters received from the ES training and computes the NN
class SineNN {
  public:
    SineNN() {
      getWB();
    }

    //matrix multiplication function (the rows of the right matrix must be equal to the columns of the left, this is why there are only 3 variables
    //calculates leftMatrix*rightMatrix and returs the result in outMatrix
    void mult(float leftMatrix[], float rightMatrix[], int leftRows, int leftCols, int rightCols, float outMatrix[]) {
      int i, j, k;
      for (i = 0; i < leftRows; i++)
        for (j = 0; j < rightCols; j++)
        {
          outMatrix[rightCols * i + j] = 0;
          for (k = 0; k < leftCols; k++)
            outMatrix[rightCols * i + j] = outMatrix[rightCols * i + j] + leftMatrix[leftCols * i + k] * rightMatrix[rightCols * k + j];
        }

    }

    //computes the NN
    void compute(float in[inputNeurons], float out[]) {
      for (int i = 0; i < inputNeurons; i++)
      {
        inputs[i] = in[i];
      }
      float hiddenActivs[hiddenNeurons];

      //applying weights, adding the biases and activating the hidden layer
      mult(hiddenWeights, inputs, hiddenNeurons, inputNeurons, 1, hiddenActivs);
      for (int i = 0; i < hiddenNeurons; i++)
      {
        hiddenActivs[i] += hiddenBiases[i];
        hiddenActivs[i] = sin(out[i]);
      }

      //doing the same operations for the output layer, but without the activation (linear layer
      mult(outputWeights, hiddenActivs, outputNeurons, hiddenNeurons, 1, out);
      for (int i = 0; i < outputNeurons; i++)
      {
        out[i] += outputBiases[i];
      }
    }

    //receive weight and bias matrices from a parameter vector
    void getWB() {
      float wb[] = { -2.3189966558146913 , -0.48152011981952975 , 3.4195499883771863 , 7.6775815471410285 , 6.030681334882363 , 1.1480159361627336 , -1.7528942064014486 , 5.460055065420295 , 8.200754194839643 , 1.9318917256282617 , 0.23347641648808828 , -5.54083487770059 , 2.1855143961185393 , 5.822199294145511 , -4.866817434173339 , 2.588846319979428 , -3.982973875156552 , 4.555377756311717 , 3.2354968024520345 , -1.7567596339002647 , 1.4764551846861504 , 5.91204775406742 , 5.214616091044845 , 1.5069079033780888 , -2.3472438414874035 , 1.5010685119229368 , -13.477271086678627 , -0.5694146727594289 , 4.629031641371418 , 2.470921961739005 , -0.3218514998375413 , -0.7166387022965589 , 0.04846159285199454 , -0.16846339563334914 , -3.6179350706729845 , 5.438201118937388 , 1.012438541328934 , -1.0174690708202225 , -5.829510017899755 , 2.2887448296052026 , -1.7655044895548766 , -2.4945973257721663 , -0.7700503224198058 , 5.565117162169942 , -5.203919614966911 , 4.112779036029637 , -4.611108336227448 , 0.3934659456468574 , 8.786941902113048 , 1.9599429127658945 , 1.875524723450157 , -2.1925360978057613 , -9.481405953552702 , -2.398328416478604 , -7.023425532993452 , -1.2675852998927872 , 0.7565933316277276 , -1.3960730787701239 , -7.598058961283291 , -9.080144867779659 , -11.567644302088752 , -0.25526922820190284 , -0.48010874158600597 , -4.613201696977585 , 4.2715302639931405 , -0.9621731135477484 , 0.24333154654875302 , 1.449749266610363 , 3.2041541714422603 , -1.7396748478169015 , -2.0150298114493173 , -2.4854552890956163 , -0.22337663498784668 , 2.5326681584938533 , 0.11189396457805342 , -1.9081341732738688 , 1.7504630222574047 , -1.9835969344586162 , -0.5724603722191299 , -2.039223762616938 , 0.6475477272668966 , 4.001572734491261 , 0.11326891783262574 , 3.481359248895213 , 1.107568155740255 , -0.5129512097890713 , 1.2809885566139207 , -0.6449946484738561 , -0.8512800688126776 , 0.7028606880248771 , -18.610322142355916 , 1.0501294956082718 , -0.07468149961704094 , -1.5020505878572765 , -0.3980439739583091 , -1.0675117342664284 , 0.011467135915699624 , 0.18896382554393298 , 0.6785999955062408 , -0.17171015111596516 , 0.46468686752891986 , 0.048265234070638756 , -11.934333989253522 , -0.3005068853474465 , -0.5016580688898896 , -0.14160827734780201 , -0.0070935621492299455 , 1.1553050581060713 , -1.3251186549805827 , -0.5481061828923645 , 0.09194943433122309 , 2.5304094344860566 , -0.16166495150136803 , -0.0965508962736199 , -8.570695028152143 , 0.46363614705092027 , 0.1670855380901918 , 3.2788927450878034 , 0.849543802330903 , -5.248673920833784 , 0.5362027235300351 , -1.2006201603198603 , 3.7717821519351844 , 2.2091236196312307 , 3.5719660586858875 , -3.0517024986299015 , -12.27083728146756 , -0.5668637115627351 , 0.1330213339197141 , -2.340157281100988 , 0.2952536260414452 , 6.208977866914687 , 0.5397451544344051 , -0.09753905282786182 , 0.4659001711718133 , 0.09873146602423047 , -0.5012617069811991 , -0.3297141108933921 , -13.584174603361802 , -0.0809222608190261 , 0.26669437703206283 , -0.3213789986248607 , 0.2141358446785135 , 0.6229845341803081 , -0.49889460080215553 , 2.419120674593615 , 1.022801258858648 , 1.2566368657851386 , -0.5590080308815141 , 1.7444520705284015 , -0.4202510210219565 , 4.022827747218553 , 0.6195774819224696 , 0.8726473930991667 , 2.017759151072103 , -2.039250060899823 , -3.236982817616063 , 0.3259326085165531 , -1.6429817307117676 , 0.1374041632407491 , 0.7047440004274114 , -0.7976117754238113 , -3.2773309870019895 , -0.2416225073764513 , -1.127489867751155 , 1.7518628172695723 , -2.833363594930881 , 4.688618917607647 , 2.1538400326785583 , 0.08183391724356326 , -2.380458370001662 , -0.9591159162172529 , 1.138812716165228 , 3.0003097624463253 , -5.3377216107679155 , -2.2623083084363214 , 0.3120310989909998 , -1.770167833271074 , 1.3438196530976307 , 2.3391499302375567 , -4.195796469172309 , 0.09702705194855027 , -1.4443884627741714 , 0.0485403694530055 , -2.511145810180378 , 1.4926862157402991 , -3.9145905647672756 , 0.1514650515825857 , 1.5059224574306156 , 3.312249588735263 , -2.1745220798640092 , 0.2696532444520421 , 0.8117273897225559 , -0.28309039132529346 , 0.8151072960100197 , 0.15512323256043986 , 0.42764760221221265 , 0.5231688471407057 , -17.542094985186974 , -0.3682411724143039 , -0.389823660122787 , 0.2299530283808525 , 0.33329293595774473 , 0.3069502667734135 , 0.7350243802319077 , 0.13417621090634163 , -0.23720545446379904 , -0.027132275424310445 , -0.4516934675259029 , -0.2474015734198397 , -17.589660739087456 , 0.3993990870821196 , 0.3628086826915592 , 0.16947706709714588 , 0.26932968760575615 , -0.5605287432525156 , -11.768750923232062 , 3.100861396038621 , -2.6954355357634943 , -9.301604003717143 , -4.629869951884433 , 9.386809686633455 , 8.5366432018075 , -12.399467156219924 , -10.274774447788094 , -9.305983645218106 , -5.2645820761968425 , 3.274898594198659};

      //in case a wrong hidden layer size is set
      if (sizeof(wb) / sizeof(wb[0]) != hiddenWSize + hiddenNeurons + outputNeurons + outputWSize) {
        pinMode(13, OUTPUT);
        while (true); {
          digitalWrite(13, HIGH);
          delay(100);
          digitalWrite(13, LOW);
          delay(100);
        }
      }

      //loading weights and biases in their respective arrays
      for (int i = 0; i < hiddenWSize; i++)
      {
        hiddenWeights[i] = round(wb[i] * 10) / 10;
      }
      for (int i = hiddenWSize; i < hiddenWSize + hiddenNeurons; i++)
      {
        hiddenBiases[i - hiddenWSize] = wb[i];
      }
      for (int i = hiddenWSize + hiddenNeurons; i < hiddenWSize + hiddenNeurons + outputWSize; i++)
      {
        outputWeights[i - hiddenWSize - hiddenNeurons] = wb[i];
      }
      for (int i = hiddenWSize + hiddenNeurons + outputWSize; i < hiddenWSize + hiddenNeurons + outputWSize + outputNeurons; i++)
      {
        outputBiases[i - hiddenWSize - hiddenNeurons - outputWSize] = wb[i];
      }
    }
  private:
    float inputs[inputNeurons]; //all vectors are collumn vectors!
    float  hiddenWeights[hiddenWSize];
    float  hiddenBiases[hiddenNeurons];
    float  outputWeights[outputWSize];
    float  outputBiases[outputNeurons];
};

SineNN nn;
void setup() {
  nn.getWB();
  
  // rapid blinking in case the IMU is not placed
  if (!bno.begin())
  {
    pinMode(13, OUTPUT);
    while (true); {
      digitalWrite(13, HIGH);
      delay(10);
      digitalWrite(13, LOW);
      delay(10);
    }
  }
  
  bno.setExtCrystalUse(true); // for better accuracy, most BNO055 sensors use an external crystal instead of the built-in one
  for (int i = 0; i < 12; i++) {
    servoarray[i].attach(servopins[i]);
  }
}

void loop() {
  startTime = millis();
  sensors_event_t orientationData , angVelocityData;
  //reading the IMU data
  bno.getEvent(&orientationData, Adafruit_BNO055::VECTOR_EULER);
  bno.getEvent(&angVelocityData, Adafruit_BNO055::VECTOR_GYROSCOPE);
  
  //Inputs correspond to Pitch, Roll, dPitch/dt, dRoll/dt
  float inputs[5] = { -orientationData.orientation.y * 3.14 / 180, orientationData.orientation.z * 3.14 / 180, angVelocityData.gyro.y * 3.14 / 180, -angVelocityData.gyro.z * 3.14 / 180, timeSteps}; 
  float angles[12]; 
  nn.compute(inputs, angles);
  
  //inverting the signals for the left servos and clipping them to their limited values
  for (int i = 0; i < 6; i++)
    angles[i] *= -1;
  for (int i = 0; i < outputNeurons; i++) {
    if (angles[i] >= ubounds[i])
      angles[i] = ubounds[i];
    if (angles[i] <= lbounds[i])
      angles [i] = lbounds[i];
  }

  for (int i = 6; i < 12; i++) {
    if (i == 6 || i == 9)
    {
      servoarray[i].write(90);
    }
    servoarray[i].write((angles[i] + offsets[i]) * 180 / 3.14);
  }

  servoarray[0].write(90);
  servoarray[1].write((angles[1] + offsets[1]) * 180 / 3.14);
  servoarray[2].write((angles[2] + offsets[2]) * 180 / 3.14);
  servoarray[3].write(90);
  servoarray[4].write((angles[4] + offsets[4]) * 180 / 3.14);
  servoarray[5].write((angles[5] + offsets[5]) * 180 / 3.14);

  endTime = millis();
  
  //ensuring that the cycle time is a multiple of the PyBullet step time
  delayTime = (endTime-startTime)%timeStepLength
  delay(delayTime); 
  
  timeSteps += 0.03 //the time factor is the same as the one used during the training of the network
}
