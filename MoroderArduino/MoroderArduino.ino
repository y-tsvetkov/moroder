//Simple class for executing a pre-trained neural network with a sinusoidal activation function.
//First index is rows, second index is cols. All vectors are collumn vectors.
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <Servo.h>
#define BNO055_SAMPLERATE_DELAY_MS (100)

// Check I2C device address and correct line below (by default address is 0x29 or 0x28)
//                                   id, address
Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28);

const float femurBound = 3.14/4;
const float tibiaBound = 3.14/4;
const float coxaBound = 0.1;

static const int inputNeurons = 6;
static const int hiddenNeurons = 12;
static const int outputNeurons = 12;

static const int hiddenWSize = hiddenNeurons * inputNeurons;
static const int outputWSize = outputNeurons * hiddenNeurons;
double timeStepLengthMillis = 1000 / 240;
double timeSteps = 0;
double startTime, endTime, timeToNextFactor;
/*float ubounds[] = {femurBound, 180, coxaBound, femurBound, 180, coxaBound, femurBound, tibiaBound, coxaBound, femurBound, tibiaBound, coxaBound};
  float lbounds[] = {-femurBound, 180-tibiaBound, coxaBound, -femurBound, 180-tibiaBound, coxaBound, -femurBound, 0, coxaBound, -femurBound, 0, coxaBound};*/
double ubounds[] = {coxaBound, femurBound, tibiaBound, coxaBound, femurBound, tibiaBound, coxaBound, femurBound, tibiaBound, coxaBound, femurBound, tibiaBound};
double lbounds[] = { -coxaBound, -femurBound, 31.4/180, -coxaBound, -femurBound, 31.4/180, -coxaBound, -femurBound, 31.4/180, -coxaBound, -femurBound, 31.4/180};
double offsets[] = {0, 0, 0,  0, 0, 0,  0, 0, 0,   0, 0, 0};
int servopins[] = {2, 23, 38,   3, 22, 37,   4, 21, 36,   5, 20, 35}; //FR,BR,FL,BL respectively
Servo servoarray[12];
class SineNN {
  public:
    SineNN(/*int inputN, int outputN, int hiddenN*/) {
      /*inputNeurons = inputN;
        outputNeurons = outputN;
        hiddenNeurons = hiddenN;*/
      //here is where getWB comes
      getWB();
    }
    void mult(double lMatrix[], double rMatrix[], int lrows, int lcols, int rcols, double outMatrix[]) {
      int i, j, k;
      for (i = 0; i < lrows; i++)
        for (j = 0; j < rcols; j++)
        {
          outMatrix[rcols * i + j] = 0;
          for (k = 0; k < lcols; k++)
            outMatrix[rcols * i + j] = outMatrix[rcols * i + j] + lMatrix[lcols * i + k] * rMatrix[rcols * k + j];
        }

    }

    void compute(double in[inputNeurons], double outnew[]) { //ensure that len(outArry) == outputNeurons
      for (int i = 0; i < inputNeurons; i++)
      {
        inputs[i] = in[i];
        // Serial.println(in[i]);
      }
      //Serial.println(' ');
      double out[hiddenNeurons];
      mult(hiddenWeights, inputs, hiddenNeurons, inputNeurons, 1, out);
      // Serial.println(out[1]);
      for (int i = 0; i < hiddenNeurons; i++)
      {
        out[i] += hiddenBiases[i]; //adding hidden biases (literally)
        out[i] = sin(out[i]); //activating the layer

      }

      mult(outputWeights, out, outputNeurons, hiddenNeurons, 1, outnew);
      for (int i = 0; i < outputNeurons; i++)
      {
        outnew[i] += outputBiases[i];
      }
      /*Serial.print(outnew[0]);
        Serial.print(" ");
        Serial.print(outnew[1]);
        Serial.print(" ");
        Serial.print(outnew[2]);
        Serial.print(" ");
        Serial.print(outnew[3]);
        Serial.println(" ");*/
      //Serial.println(' ');//IT GETS DESTROYED HERE, THAT'S WHY I ONLY RECEIVE ZERO
    }
    void getWB() {
      float wb[] = {-0.479759983110365 ,0.2333833539969395 ,-0.4328215181328638 ,-0.11045360389319127 ,0.05404079746002457 ,0.021224348909287316 ,-0.22158185853641038 ,0.5789160689596182 ,0.013528187106157722 ,0.004584325375507119 ,-0.03732200313930148 ,-0.2938135596305904 ,-0.35837988442421764 ,-0.03537272557282726 ,0.05734336730587766 ,-0.0012307203806779625 ,-0.1238404721662281 ,0.015765935605766883 ,0.4615631215970036 ,-0.27823773062483986 ,-0.6626088330608758 ,0.15311365380361047 ,0.10023229330530091 ,0.3026740536170878 ,0.52798249985086 ,0.023840880100000024 ,0.16113491379969197 ,-0.17173718851340514 ,-0.058277976813126464 ,-0.21207725229286065 ,-0.3829991915820151 ,-0.6058242310357954 ,0.5685142046593038 ,0.5172799111895463 ,0.013017803173475633 ,0.31416785105798306 ,0.26480202328227465 ,-0.334443327644523 ,0.2220475506769856 ,0.3208638653485016 ,0.004189711656996029 ,0.29667601356034584 ,0.6528097319764169 ,0.13546408196130216 ,-0.08978422699035533 ,0.19111367049208255 ,0.3496432799169225 ,-0.28806551039945194 ,-0.07323665270597667 ,0.7302174354697439 ,-0.319263955674456 ,0.5822248755898937 ,-0.02822330582412479 ,0.2917778006532134 ,-0.31882683578166693 ,-0.19068844548797084 ,-0.534201531916531 ,-0.20777527866754159 ,0.22889297428492625 ,0.01129430326794368 ,0.025313823672344922 ,0.4670412785328725 ,0.061230384769501746 ,-0.4425170215852796 ,0.3370168387475796 ,-0.18087634167283517 ,0.32664726313549164 ,0.32828726439974926 ,-0.47737341915762993 ,0.214358519318473 ,-0.19650200595767994 ,-0.27371259860755526 ,0.589513523387198 ,0.48239638253400907 ,0.14085882105842623 ,0.0520559206039491 ,0.3294617956469177 ,-0.30439567621824354 ,0.3792541160224216 ,0.340473585247851 ,-0.3038707273373653 ,0.17114172054243004 ,-0.2803244300734377 ,0.08438890226969968 ,-0.11670786774761782 ,-1.0340691194695537 ,0.08137672764509031 ,0.2908104023875347 ,0.0009711617919679668 ,0.31090052867364565 ,0.17605262580795195 ,0.12005295564383374 ,0.013238472874365203 ,-0.27615416117707375 ,0.11179053471420246 ,-0.1926338556561573 ,-0.34024353546445824 ,1.2378835029097983 ,0.3945505490733749 ,0.5819725556427484 ,-0.28101674447779607 ,-0.5569836898768565 ,-0.1686817473830248 ,-0.14348147734631442 ,0.31795662923018675 ,0.18801020230385782 ,-0.13411544096921993 ,0.44566529211726047 ,0.28983650808480405 ,1.4220769319635174 ,0.23435335243445915 ,-0.353243629510279 ,-0.30081891505212344 ,-0.4363170260491382 ,0.33289768280376286 ,-0.10646966213934493 ,0.28706709491750815 ,-0.0455733466819689 ,0.07235124098515909 ,0.28766237998385347 ,-0.15433211429298982 ,1.0378601483198016 ,-0.7044400179831133 ,0.042754612144994764 ,0.37181344940786537 ,-0.8152341899897518 ,-0.2329945471819642 ,0.5849673352277476 ,-0.5019405812480051 ,0.3325523919459479 ,-0.27194224967333713 ,0.31233303218915964 ,-0.1227747848278963 ,-0.4332038414173515 ,0.052101095754993956 ,-0.20178716179884099 ,0.0012189386010480249 ,0.0064549723625568975 ,-0.19633398965495985 ,-0.011659318930706939 ,0.07091326201958761 ,0.021092093311210057 ,0.020311399904821673 ,-0.029262926302251514 ,0.12396130345446935 ,-0.5573488949253999 ,-0.25498514085197915 ,0.2851754417077684 ,-0.5026554435017183 ,0.008549635688244923 ,0.846688878810023 ,-0.09728067751850106 ,-0.002117106187789988 ,-0.37430390187561824 ,0.06171679217490535 ,-0.7984407372384792 ,-0.1363053793647556 ,-0.4662634714736396 ,0.1939467978709958 ,-0.2187600079949439 ,-0.0682038923940321 ,0.3624255876483904 ,0.04403890752755368 ,-0.2329112156787927 ,-0.02076937913104259 ,-0.4268165620099349 ,-0.04082987533706077 ,-0.28714592788349524 ,-0.07586106384443253 ,-0.8999784420291813 ,-0.26638402763724034 ,0.2557328961182062 ,0.026205594331762137 ,-0.12604810887487322 ,-0.07546550763185181 ,0.17549729324730437 ,-0.20575204697278793 ,-0.17224905732359588 ,-0.041920675598691406 ,0.08295165317119738 ,1.3550085509840795 ,-1.2955815728156097 ,0.9128781242034812 ,0.22900729154444538 ,-0.10559496018745262 ,0.3329600938509365 ,0.2648705373298953 ,0.19262447326946333 ,0.1338614843909796 ,0.16574255426412887 ,0.0996692700452562 ,0.6283697209913809 ,0.1380167808185925 ,0.6038621937948038 ,-0.11634531082605001 ,-0.056523511540055825 ,-0.21647779211612098 ,0.025335761823803045 ,-0.12246868895256624 ,-0.0003734748128856574 ,-0.35162103976049625 ,0.2316677927731556 ,0.0010587885363334986 ,0.18436117851068937 ,0.05497369558265896 ,0.2222515453592193 ,0.005775688459772985 ,-0.22358925882310093 ,-0.043140853766824784 ,0.10098592965647546 ,-0.021700718779272744 ,-0.053540459478095666 ,0.055958658083131455 ,0.06165108045545972 ,0.013908973682143763 ,0.010100296264986845 ,0.3363738788264999 ,0.6572980540984474 ,-0.3814774919857772 ,-0.16332077484855648 ,0.3416995810395442 ,-0.011411028049930064 ,-0.4052032034645 ,0.20243618249050638 ,-0.09533353967725236 ,0.72235815987203 ,0.015792878995889673 ,0.1890487651917372 ,-0.23639616083114673 ,-0.17384397773742832 ,-0.009737470690234545 ,0.46775192819487604 ,0.03722795552836151 ,-0.8156860690501111 ,0.2971478585714241 ,0.2159572076993838 ,-0.6444822552656023 ,0.28525692114126244 ,0.032952243933734734 ,-0.08635851194075402}; 
      if (sizeof(wb) / sizeof(wb[0]) != hiddenWSize + hiddenNeurons + outputNeurons + outputWSize) {
        pinMode(13, OUTPUT);
        while (true) {
          digitalWrite(13, HIGH);
          delay(100);
          digitalWrite(13, LOW);
          delay(100);
        }

      }
      //Serial.println(sizeof(wb) / sizeof(wb[0]));
      for (int i = 0; i < hiddenWSize; i++)
      {
        hiddenWeights[i] = wb[i];
        //Serial.println(hiddenWeights[i]);
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
    double inputs[inputNeurons]; //all vectors are collumn vectors!
    double  hiddenWeights[hiddenWSize];
    double  hiddenBiases[hiddenNeurons];
    double  outputWeights[outputWSize];
    double  outputBiases[outputNeurons];
    //double static weightBiasData[hiddenWSize + hiddenNeurons + outputWSize + outputNeurons];
};
SineNN nn;
void setup() {
  Serial.begin(115200);
  nn.getWB();
  // put your setup code here, to run once:
  if (!bno.begin())
  {
    /* There was a problem detecting the BNO055 ... check your connections */
    Serial.print("No BNO055 detected ... Check your wiring or I2C ADDR!");
    while (1);
  }
  bno.setExtCrystalUse(true);
  /*double a[5] = {0, 0, 0, 0, 0};
    nn.compute(a);*/
  for (int i = 0; i < 12; i++) {
    servoarray[i].attach(servopins[i]);
  }
}

void loop() {
  nn.getWB();
  startTime = millis();
  sensors_event_t orientationData , angVelocityData;
  bno.getEvent(&orientationData, Adafruit_BNO055::VECTOR_EULER);
  bno.getEvent(&angVelocityData, Adafruit_BNO055::VECTOR_GYROSCOPE);
  double inputs[6] = { -orientationData.orientation.y * 3.14 / 180, orientationData.orientation.z * 3.14 / 180, 6.28-orientationData.orientation.x*3.14/180, angVelocityData.gyro.y * 3.14 / 180, -angVelocityData.gyro.z * 3.14 / 180, timeSteps}; //Pitch, Roll, dPitch, dRoll! Check sensor orientation after assembly!
  /*for (int i = 0; i<=5; i++)
    Serial.println(inputs[i]);
    Serial.println(" ");*/
  double data[12];
  double  ins[5] = {0, 0, 0, 0, timeSteps};
  nn.compute(inputs, data);

  for (int i = 0; i < outputNeurons; i++) {
    if (data[i] >= ubounds[i])
      data[i] = ubounds[i];
    if (data[i] <= lbounds[i])
      data [i] = lbounds[i];
  }

  for (int i = 0; i < 12; i++) {
      if (i % 3 == 1)
        data[i] += PI / 2; 
  }

  for (int i = 0; i < 6; i++) //invert the left servos
    data[i] = 3.14 - data[i];
    
  for (int i = 0; i < 12; i++) {
    if (i % 3 != 0) {
      servoarray[i].write((data[i] + offsets[i]) * 180 / 3.14);
    }
    else {
      servoarray[i].write(90);
    }
  }
  for (int i = 0; i < 6; i++) {
    if (i % 3 != 0) {
      Serial.print((data[i] + offsets[i]) * 180 / 3.14);
      Serial.print(" ");
    }
  } Serial.println(" ");

  delay(1);



  //delay(timeToNextFactor * timeStepLengthMillis - endTime + startTime); //it takes timeToNextFactor*timeStepLengthMillis miliseconds to reach the next "step".
  endTime = millis();
  timeToNextFactor = floor((endTime - startTime) / timeStepLengthMillis) + 1;
  /* Serial.println(" ");
    Serial.println(endTime - startTime);
    Serial.println(timeToNextFactor);
    Serial.println(" ");*/

  timeSteps += 0.03;//* timeToNextFactor; // To fix the time scaling, the time step factor is multiplied by timeToNextFactor.
}



/*
   [ 3.29611750e+00  1.02116747e-01  1.50709592e-01 -5.60376098e+00
  1.87612961e+00  7.43569292e-01 -1.68571773e-02 -5.20458733e+00
  4.39713487e-02  6.44657780e+00 -2.31225284e+00 -2.37172580e+00
  -4.30610368e+00  3.74608157e+00  1.12722930e+00  7.08753877e-01
  4.46384874e-01  1.39767399e-01  5.70416673e-03 -3.30867010e+00
  -1.52384885e+00  4.42486150e+00 -2.12241060e+00  2.68620336e+00
  3.31334250e+00 -2.39734240e+00 -3.77568482e+00  2.89325966e+00
  4.46587123e+00  4.99251790e-01  1.61382189e+00 -3.38941166e+00
  2.60069885e+00 -2.39707673e+00  3.62913774e+00  2.26972012e+00
  2.10966851e+00  4.91709485e+00 -2.98796558e+00  3.81061594e+00
  -9.58112153e-01  1.44007991e+00  2.52883245e+00 -2.67348871e+00
  -2.38399183e+00  1.37985427e+00 -9.49669918e-01  2.91071804e+00
  -6.72818192e-01 -5.14818612e-01  2.48457147e-01 -7.64302761e+00
  5.70726891e-01  3.02332528e-01  4.62092244e-01 -1.79514709e+00
  -1.32797165e+00 -1.27531019e+00 -2.54598794e+00  1.78793255e+00
  -2.11482321e+00 -6.35907943e+00 -1.83379345e+00 -4.09374346e+00
  -3.84231077e-01 -5.30065482e-01  2.19116721e-03  6.16882327e+00
  1.87989264e-01 -5.98358880e-02  2.65306392e-02 -4.62583835e-01
  2.05078080e+00 -5.00697470e-01 -1.39974551e+00 -8.23301558e+00
  -3.00235727e-01 -5.48194361e-01  9.34035779e-01  2.35377211e-01
  -1.19452040e-01  5.32567285e-01  2.12304170e-01  7.27687862e+00
  -8.17472071e-01 -4.17640078e-01  5.41098892e-01 -4.16401034e-01
  -8.11533726e-02  1.82173851e-01 -5.09746639e-01  9.20954618e+00
  -2.27885047e-01  2.17746064e-01 -3.22472327e-01  9.44423679e-02
  -1.21068341e-01 -4.10599196e-01  1.07849750e+00  1.47575270e+00
  2.96765681e+00  5.01855798e-01 -9.67296991e-02  5.12271236e-01
  -1.58838759e+00  1.62550014e+00  9.64683904e-01  9.18747545e+00
  -1.46985857e+00  8.64058204e-01  1.74487297e-02  2.84011654e-01
  1.74085490e-01  1.02341352e+00 -4.08123044e+00  3.75443458e+00
  -1.57418469e+00  6.27810836e-01 -8.10902304e-01  3.43544847e-01
  1.13173301e+00 -2.55075574e+00 -1.64261978e-01 -4.04741271e-01
  3.48335662e+00 -5.56051318e-01  6.65179427e-01 -2.74687265e+00
  -3.46581320e-01  1.21184631e-02  2.99231962e-01  5.41394060e+00
  -1.61327183e-01  1.37474524e-01  2.43917497e-01 -7.15739704e-01
  -5.29010991e-01  2.24715381e+00 -5.39934397e-01  8.67610820e+00
  6.59003273e-02  5.94710085e-02  1.24134553e+00 -3.82823518e-01
  -7.22497879e+00 -5.94437962e+00  6.18556941e+00 -5.86914061e-01
  -3.04683455e+00  5.75530702e+00  5.13867112e+00  1.21348465e+00
  -3.41134345e+00  3.98664785e+00 -3.34464216e+00  4.45107518e+00]

























 * */
