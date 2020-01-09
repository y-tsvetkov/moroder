#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <Servo.h>
#define BNO055_SAMPLERATE_DELAY_MS (100)
Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28);

//defining max deflection for coxa, femur, tibia servos in radians
const float femurBound = 3.14/4;
const float tibiaBound = 3.14/4;
const float coxaBound = 0.1;

//network architecture
static const int inputNeurons = 6;
static const int hiddenNeurons = 12;
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
      float wb[] = {-0.479759983110365 ,0.2333833539969395 ,-0.4328215181328638 ,-0.11045360389319127 ,0.05404079746002457 ,0.021224348909287316 ,-0.22158185853641038 ,0.5789160689596182 ,0.013528187106157722 ,0.004584325375507119 ,-0.03732200313930148 ,-0.2938135596305904 ,-0.35837988442421764 ,-0.03537272557282726 ,0.05734336730587766 ,-0.0012307203806779625 ,-0.1238404721662281 ,0.015765935605766883 ,0.4615631215970036 ,-0.27823773062483986 ,-0.6626088330608758 ,0.15311365380361047 ,0.10023229330530091 ,0.3026740536170878 ,0.52798249985086 ,0.023840880100000024 ,0.16113491379969197 ,-0.17173718851340514 ,-0.058277976813126464 ,-0.21207725229286065 ,-0.3829991915820151 ,-0.6058242310357954 ,0.5685142046593038 ,0.5172799111895463 ,0.013017803173475633 ,0.31416785105798306 ,0.26480202328227465 ,-0.334443327644523 ,0.2220475506769856 ,0.3208638653485016 ,0.004189711656996029 ,0.29667601356034584 ,0.6528097319764169 ,0.13546408196130216 ,-0.08978422699035533 ,0.19111367049208255 ,0.3496432799169225 ,-0.28806551039945194 ,-0.07323665270597667 ,0.7302174354697439 ,-0.319263955674456 ,0.5822248755898937 ,-0.02822330582412479 ,0.2917778006532134 ,-0.31882683578166693 ,-0.19068844548797084 ,-0.534201531916531 ,-0.20777527866754159 ,0.22889297428492625 ,0.01129430326794368 ,0.025313823672344922 ,0.4670412785328725 ,0.061230384769501746 ,-0.4425170215852796 ,0.3370168387475796 ,-0.18087634167283517 ,0.32664726313549164 ,0.32828726439974926 ,-0.47737341915762993 ,0.214358519318473 ,-0.19650200595767994 ,-0.27371259860755526 ,0.589513523387198 ,0.48239638253400907 ,0.14085882105842623 ,0.0520559206039491 ,0.3294617956469177 ,-0.30439567621824354 ,0.3792541160224216 ,0.340473585247851 ,-0.3038707273373653 ,0.17114172054243004 ,-0.2803244300734377 ,0.08438890226969968 ,-0.11670786774761782 ,-1.0340691194695537 ,0.08137672764509031 ,0.2908104023875347 ,0.0009711617919679668 ,0.31090052867364565 ,0.17605262580795195 ,0.12005295564383374 ,0.013238472874365203 ,-0.27615416117707375 ,0.11179053471420246 ,-0.1926338556561573 ,-0.34024353546445824 ,1.2378835029097983 ,0.3945505490733749 ,0.5819725556427484 ,-0.28101674447779607 ,-0.5569836898768565 ,-0.1686817473830248 ,-0.14348147734631442 ,0.31795662923018675 ,0.18801020230385782 ,-0.13411544096921993 ,0.44566529211726047 ,0.28983650808480405 ,1.4220769319635174 ,0.23435335243445915 ,-0.353243629510279 ,-0.30081891505212344 ,-0.4363170260491382 ,0.33289768280376286 ,-0.10646966213934493 ,0.28706709491750815 ,-0.0455733466819689 ,0.07235124098515909 ,0.28766237998385347 ,-0.15433211429298982 ,1.0378601483198016 ,-0.7044400179831133 ,0.042754612144994764 ,0.37181344940786537 ,-0.8152341899897518 ,-0.2329945471819642 ,0.5849673352277476 ,-0.5019405812480051 ,0.3325523919459479 ,-0.27194224967333713 ,0.31233303218915964 ,-0.1227747848278963 ,-0.4332038414173515 ,0.052101095754993956 ,-0.20178716179884099 ,0.0012189386010480249 ,0.0064549723625568975 ,-0.19633398965495985 ,-0.011659318930706939 ,0.07091326201958761 ,0.021092093311210057 ,0.020311399904821673 ,-0.029262926302251514 ,0.12396130345446935 ,-0.5573488949253999 ,-0.25498514085197915 ,0.2851754417077684 ,-0.5026554435017183 ,0.008549635688244923 ,0.846688878810023 ,-0.09728067751850106 ,-0.002117106187789988 ,-0.37430390187561824 ,0.06171679217490535 ,-0.7984407372384792 ,-0.1363053793647556 ,-0.4662634714736396 ,0.1939467978709958 ,-0.2187600079949439 ,-0.0682038923940321 ,0.3624255876483904 ,0.04403890752755368 ,-0.2329112156787927 ,-0.02076937913104259 ,-0.4268165620099349 ,-0.04082987533706077 ,-0.28714592788349524 ,-0.07586106384443253 ,-0.8999784420291813 ,-0.26638402763724034 ,0.2557328961182062 ,0.026205594331762137 ,-0.12604810887487322 ,-0.07546550763185181 ,0.17549729324730437 ,-0.20575204697278793 ,-0.17224905732359588 ,-0.041920675598691406 ,0.08295165317119738 ,1.3550085509840795 ,-1.2955815728156097 ,0.9128781242034812 ,0.22900729154444538 ,-0.10559496018745262 ,0.3329600938509365 ,0.2648705373298953 ,0.19262447326946333 ,0.1338614843909796 ,0.16574255426412887 ,0.0996692700452562 ,0.6283697209913809 ,0.1380167808185925 ,0.6038621937948038 ,-0.11634531082605001 ,-0.056523511540055825 ,-0.21647779211612098 ,0.025335761823803045 ,-0.12246868895256624 ,-0.0003734748128856574 ,-0.35162103976049625 ,0.2316677927731556 ,0.0010587885363334986 ,0.18436117851068937 ,0.05497369558265896 ,0.2222515453592193 ,0.005775688459772985 ,-0.22358925882310093 ,-0.043140853766824784 ,0.10098592965647546 ,-0.021700718779272744 ,-0.053540459478095666 ,0.055958658083131455 ,0.06165108045545972 ,0.013908973682143763 ,0.010100296264986845 ,0.3363738788264999 ,0.6572980540984474 ,-0.3814774919857772 ,-0.16332077484855648 ,0.3416995810395442 ,-0.011411028049930064 ,-0.4052032034645 ,0.20243618249050638 ,-0.09533353967725236 ,0.72235815987203 ,0.015792878995889673 ,0.1890487651917372 ,-0.23639616083114673 ,-0.17384397773742832 ,-0.009737470690234545 ,0.46775192819487604 ,0.03722795552836151 ,-0.8156860690501111 ,0.2971478585714241 ,0.2159572076993838 ,-0.6444822552656023 ,0.28525692114126244 ,0.032952243933734734 ,-0.08635851194075402} 
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
  //TODO: Add the orientation term in the right place (may need to be inverted)
  float inputs[6] = { -orientationData.orientation.y * 3.14 / 180, orientationData.orientation.z * 3.14 / 180, orientationData.orientation.x*3.14/180, angVelocityData.gyro.y * 3.14 / 180, -angVelocityData.gyro.z * 3.14 / 180, timeSteps}; 
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
