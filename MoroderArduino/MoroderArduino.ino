// This code is for transferring the trained sinusoidal neural network with course corrective ability to the real world;
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <Servo.h>
#define BNO055_SAMPLERATE_DELAY_MS (100)

// Connecting to IMU (0x28 is the IMU's I2C address);
Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28);

const float femurBound = 3.14/4; // Femur (hip) servo motor safety restrictions;
const float tibiaBound = 3.14/4; // Tibia (knee) servo motor safety restrictions;
const float coxaBound = 0.1; // Coxa (side hip) servo motor safety restrictions;

// Defining NN architecture:
static const int inputNeurons = 6;
static const int hiddenNeurons = 12;
static const int outputNeurons = 12;

static const int hiddenWSize = hiddenNeurons * inputNeurons;
static const int outputWSize = outputNeurons * hiddenNeurons;

// time step for the algorithm - same as the simulation time step;
double timeStepLengthMillis = 1000 / 240;

double timeSteps = 0;
double startTime, endTime, timeToNextFactor;

// upper and lower motor bounds arrays;
double ubounds[] = {coxaBound, femurBound, tibiaBound, coxaBound, femurBound, tibiaBound, coxaBound, femurBound, tibiaBound, coxaBound, femurBound, tibiaBound};
double lbounds[] = { -coxaBound, -femurBound, 0*31.4/360, -coxaBound, -femurBound, 0*31.4/360, -coxaBound, -femurBound, 0*31.4/180, -coxaBound, -femurBound, 0*31.4/180};
// since servo motors can't be precisely set to be at 0 degrees during assembly, offsets are used to correct their position;
double offsets[] = {-5*3.14/180, 0, 0,  4.5*3.14/180, 5*3.14/180, 0,  3.5*3.14/180, -3*3.14/180, 0,   7*3.14/180, -10*3.14/180, 0};

// leg motor pins;
int servopins[] = {2, 23, 38,   3, 22, 37,   4, 21, 36,   5, 20, 35}; 
Servo servoarray[12];


class SineNN {
  public:
    SineNN() {
      // the constructor assigns the weights and biases to the network;
      getWB();
    }
    
    // matrix multiplication function;
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
    void compute(double in[inputNeurons], double outnew[]) { 
      
      // getting inputs;
      for (int i = 0; i < inputNeurons; i++)
      {
        inputs[i] = in[i];
      }
      
      double out[hiddenNeurons];
      
      // hidden layer matrix multiplication, adding biases and applying activ. function;
      mult(hiddenWeights, inputs, hiddenNeurons, inputNeurons, 1, out);
      for (int i = 0; i < hiddenNeurons; i++)
      {
        out[i] += hiddenBiases[i]; 
        out[i] = sin(out[i]);
      }
      
      // output layer matrix multiplication, adding biases and applying activ. function;
      mult(outputWeights, out, outputNeurons, hiddenNeurons, 1, outnew);
      for (int i = 0; i < outputNeurons; i++)
      {
        outnew[i] += outputBiases[i];
      }
    }
    void getWB() { 
      
      // this array holds the weights and biases. Not done in a separate file since extern doesn't make sizeof easily accessible :( ;
      float wb[] = {-0.17454631099573611 ,0.5285045083877201 ,-0.34610973373720927 ,0.11310013948492183 ,-0.1427008427770935 ,0.5072594385027983 ,0.5115831733425533 ,0.05185217173775312 ,-0.2642859890125707 ,0.16313496671131125 ,-0.056098552905203096 ,0.31997563532236745 ,0.1790174432332857 ,0.016938480249755503 ,-0.11745480830790611 ,0.15700904163981433 ,-0.06572293192603836 ,-0.4904987523616311 ,0.7273679674991647 ,0.4159403477966344 ,0.2468848125831285 ,-0.03240078694095102 ,0.008103631837991218 ,-0.3190956262120963 ,0.34452103043084886 ,-0.25900303354803267 ,0.30443273614553135 ,-0.1695716459226148 ,-0.11300956651060629 ,-0.4222427226905978 ,0.3853646549528983 ,0.5564863642442024 ,0.26719846289881505 ,-0.16650660921528326 ,-0.414004307226437 ,0.8984379109068481 ,0.6061715126709639 ,1.3427939523024706 ,-0.35014744206557735 ,0.24224722258848091 ,-0.2604965539926771 ,0.35052593574251806 ,-0.27561212536011204 ,0.40879551640054246 ,-0.09382279675938154 ,0.4834154185018846 ,0.5505742942530701 ,-1.0558607760753262 ,0.21320899331741053 ,0.3733407344842784 ,-0.21211404964932484 ,-0.23962668030601883 ,-0.0394386664098217 ,-0.41274160932362175 ,-0.786100269896787 ,0.3887838663405175 ,0.8718166044917625 ,-0.16468170331393342 ,-0.023199975192744783 ,0.4106394533978456 ,0.1816299195289228 ,-1.3098525552849511 ,0.40923599440074154 ,0.04312483090638919 ,0.3011634959873627 ,0.6207433298423587 ,-0.36889187842101623 ,0.10401657815020243 ,0.6507151734849557 ,-0.7968261505873574 ,0.1740614742578873 ,0.47267200562155365 ,0.09755436858689356 ,0.24598036396442788 ,0.3080845597207575 ,0.06058150415821596 ,-0.4527009003526005 ,-0.7180134109770673 ,-0.21133671358959433 ,-1.4168474846117527 ,-0.27684698467294233 ,0.22575730934645807 ,-0.3706898047462813 ,0.04242471443079286 ,-0.5563304690637699 ,0.36804625063844343 ,0.08993412513438319 ,-0.637360283861049 ,0.3206571038616137 ,-0.306097048922716 ,-0.011608492698719462 ,-0.455677985115518 ,1.030061518428881 ,-0.45656806041866416 ,0.1777112388323242 ,0.3261471527038744 ,0.23450948740107228 ,-0.2631607857170859 ,0.08318199107163692 ,0.7661535046849495 ,-0.35058519227378154 ,0.020773364644873914 ,0.0020239387084619895 ,0.2867745315084322 ,-0.7027136749947106 ,0.5705530491935568 ,-0.04031567173061554 ,-0.15060774197270094 ,0.9371150632507455 ,0.44822734900400146 ,-0.18438944595618004 ,0.7743178372344774 ,-0.5470083077723502 ,0.04684975567726917 ,0.007196123004195377 ,0.2200055871537157 ,0.20956560977822994 ,0.8341694955576304 ,0.2521485776910554 ,0.03628581280940575 ,0.327037817354269 ,-0.3359304816981257 ,0.5929856287125158 ,1.2879526860484738 ,-0.16116893235756086 ,0.09163023209048657 ,-0.5073875302114937 ,-0.07060612293102292 ,-1.1066082954135483 ,0.748790850373389 ,-0.6078108723407325 ,0.28399665809478525 ,-0.19086952091363502 ,0.04975615434226362 ,-0.03230764849893167 ,-0.4366312413116412 ,0.12570882337220604 ,-0.05895446791654611 ,0.12511440852522335 ,-0.12544330979223522 ,0.613105893151964 ,-0.21925386250093756 ,0.06283208092774997 ,0.028700384907928853 ,-0.38567403965381036 ,-0.3205697168122553 ,0.5525216701344298 ,-0.8079676924207042 ,0.4363919449404008 ,0.09252460464962345 ,-0.16092725291569787 ,0.013933474434603015 ,0.26273766564962375 ,-0.38362933041864705 ,0.38056186750121707 ,0.11466368041834715 ,-0.07460587729303615 ,0.2928374067806136 ,0.11943078593655332 ,-0.7551214594656833 ,0.26079963678081347 ,-0.15146483657936924 ,0.3107476508062724 ,-0.6422811902677205 ,1.0937515459850997 ,-0.5099924662088205 ,0.2155429488325095 ,0.13881188269942946 ,-0.2639736943499637 ,0.02533673010987267 ,-0.06612241161808169 ,-0.958187227135604 ,0.4176926498561453 ,0.06050412528081471 ,0.04450522550067403 ,-0.3178441075070974 ,0.5895297752067086 ,-0.546687834877289 ,0.21928447679812144 ,0.2773461132781099 ,0.0743181730020231 ,0.4720522550567797 ,-0.060953191613077155 ,-0.40952275183204473 ,0.3054733177554495 ,-0.022819912336611257 ,0.09391155773774308 ,-0.12678971308047413 ,-0.4247092641293691 ,0.02827207906751283 ,0.3568716797899038 ,0.04249071316042701 ,1.0317178785231205 ,-0.16329239248767066 ,-0.019729754992186536 ,0.5540070871402968 ,-0.015467870785522786 ,0.07221943046314688 ,0.3185701188745764 ,0.09589069861158057 ,-1.9121789094652568 ,0.8120017231839781 ,-0.6085974100336089 ,-0.6960709031256724 ,0.10446285528353383 ,-0.03480616477242213 ,0.014376170692406579 ,0.24923460610591686 ,-0.09797909594338788 ,0.07415132297484323 ,-0.0849155624973438 ,0.09756396713097186 ,-0.3940559607629312 ,0.11910524851764001 ,-0.12480763244876068 ,-0.07243955567622093 ,0.3690014707842409 ,-0.38207805495911706 ,-0.04482194366422362 ,0.7859914711883544 ,-0.05136399253550768 ,0.2788166302639628 ,0.1596057515451803 ,-0.1556216437518278 ,-0.6332545533890213 ,-0.2835171317606897 ,-0.7322863852837518 ,0.22647325099563198 ,-0.01483854409210835 ,0.32516869400930964 ,-0.6380950487551068 ,-0.1654588747565167 ,0.11290362617816986 ,-0.9044639553716513 ,0.6189398801475201 ,0.43381053748374376 ,-1.3143434088318195 ,0.6124348535311347 ,0.10496528005015678 ,-1.0059659358376976};
      
      // Teensy light flashes if the expected number of parameters != the number of provided parameters; 
      if (sizeof(wb) / sizeof(wb[0]) != hiddenWSize + hiddenNeurons + outputNeurons + outputWSize) {
        pinMode(13, OUTPUT);
        while (true) {
          digitalWrite(13, HIGH);
          delay(100);
          digitalWrite(13, LOW);
          delay(100);
        }
      }
      
      // assigning weights and biases;
      for (int i = 0; i < hiddenWSize; i++)
      {
        hiddenWeights[i] = wb[i];
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
    double inputs[inputNeurons]; 
    double  hiddenWeights[hiddenWSize];
    double  hiddenBiases[hiddenNeurons];
    double  outputWeights[outputWSize];
    double  outputBiases[outputNeurons];
};

SineNN nn;
void setup() {
  
  // Teensy light flashes rapidly if connection with IMU wasn't established;
  if (!bno.begin())
  {
   pinMode(13, OUTPUT);
   while (true) {
    digitalWrite(13, HIGH);
    delay(1000);
    digitalWrite(13, LOW);
    delay(1000);
   }
  }
  
  // Using external crystal oscillator for greater accuracy;
  bno.setExtCrystalUse(true);

  // Enabling servos;
  for (int i = 0; i < 12; i++) {
    servoarray[i].attach(servopins[i]);
  }
}

void loop() {
  
  startTime = millis();

  // Receiving IMU data;
  sensors_event_t orientationData , angVelocityData;
  bno.getEvent(&orientationData, Adafruit_BNO055::VECTOR_EULER);
  bno.getEvent(&angVelocityData, Adafruit_BNO055::VECTOR_GYROSCOPE);
  
  double inputs[6] = { -orientationData.orientation.y * 3.14 / 180, 
  orientationData.orientation.z * 3.14 / 180, 
  6.28-orientationData.orientation.x*3.14/180, 
  angVelocityData.gyro.y * 3.14 / 180, 
  -angVelocityData.gyro.z * 3.14 / 180, 
  timeSteps};
  double data[12];
  nn.compute(inputs, data);

  // applying servo bounds;
  for (int i = 0; i < outputNeurons; i++) {
    if (data[i] >= ubounds[i])
      data[i] = ubounds[i];
    if (data[i] <= lbounds[i])
      data [i] = lbounds[i];
  }
  
  // adding pi/2 to coxa motor commands, since NN returns commands between -pi/2 and pi/2
  for (int i = 0; i < 12; i++) {
      if (i % 3 == 1)
        data[i] += PI / 2; 
  }
 // inverting left servo motor commands, since they are turned upside-down and as a result rotate in the opposite direction;
  for (int i = 0; i < 6; i++) 
    data[i] = 3.14 - data[i];
    
  // setting servo commands;
  for (int i = 0; i < 12; i++) {
    if (i % 3 != 0) {
      servoarray[i].write((data[i] + offsets[i]) * 180 / 3.14);
    }
    else {
      servoarray[i].write(90+offsets[i]*180/3.14);
    }
  }  

  // aplying delay so the network is evaluated in the same frequency as in simulation;
  endTime = millis();
  delay(timeStepLengthMillis-endTime+startTime);
  endTime = millis();

  timeSteps += 1;
}
