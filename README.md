# Autonomous-Truck
An autonomous driving system built to drive in a truck driving simulator

## Installation
Git clone the repository and ```cd``` into the directory
``` 
git clone https://github.com/greerviau/Autonomous-Truck.git && cd Autonomous-Truck
```
```git clone https://github.com/tidzo/pyvjoy``` into Autonomous-Truck  
Download and install vJoy from http://vjoystick.sourceforge.net/site/index.php/download-a-install/download  
Navigate to ```vJoy/x86``` in wherever you installed vJoy. Copy vJoyInterface.dll and paste it into the pyvjoy directory.  

## Usage 
### Data Collection
#### Game Settings
Make sure the game detects your gamepad<br/>
Make sure the controller-subtype is set to **Gamepad, joystick**<br/>
Make sure that the controller **B** button is bound to roof camera in game<br/> 
Use **1280x720** resolution in game<br/>  
Try to use the highest graphics settings possible while still being able to run the program effectively (this will take some fine-tuning)<br/> 

#### Recording Data
To collect data run ```python3 collect_data.py <session>``` Make sure to specify different sessions for every execution.<br/> 
While recording, use the **B** button to start a new recording, this will not create a new session but instead split into a new clip.<br/>  
Use this feature to start recording new clips of desired data. This will make data cleaning easier. Ex. Before changing lanes, press **B** before changing and press **B** after lane change is finished. This will create 3 clips, 1 before lane change, 1 of the lane change and the final will continue recording the rest of the drive. Then durring data cleaning simply delete the clip of the lane change.<br/>  
Recording sessions will be saved to ```data/roof_cam/raw/<session>/<clips>```
#### Cleaning
If additional cleaning is required, run ```python3 clip_video.py raw/<session>/<clip>``` While video is playing, press **q** to keyframe. Once video is done playing the program will split the video along key frames and saved to ```data/roof_cam/raw/<session>/<clips>/<splits>```<br/>   
Then simply move the clips that you want to keep to ```data/roof_cam/raw/<session>``` and discard the rest.
### Preprocessing
Once the data has been cleaned, run ```python3 preprocess.py``` This will preprocess all of the clips in ```data/roof_cam/raw``` The subfolders of this directory must have the file structure of ```<session>/<clips>``` with the mp4 and csv files within.<br/>  
This will save the preprocessed data to ```data/roof_cam/processed``` The data will be divided into sessions but the clips will be aggregated into X.npy and Y.csv<br/>   
There will also be a total aggregate of all sessions as X.npy and Y.csv 
### Train Steering Model
After preprocessing, open ```train_conv_net.py``` Make sure to specify the SAVE_PATH for the model as well as the hyperparams.<br/> 
Run ```python3 train_conv_net.py``` to train the model.
### Train Digit Recognition
To train the digit recognition for monitoring speed run ```python3 train_digit_rec.py```
### Train Brake Prediction model
To train the conv net for brake prediction run ```python3 train_brake_net.py```

### Testing
Open your game and in gameplay settings set your input as **Keyboard + vJoy Device**. If vJoy is not detected then run ```python3 detect_vjoy_ingame.py``` while your game is open and it should ask you to use vJoy as a controller. Like the Xbox controller, make sure the controller-subtype is set to **Gamepad, joystick**<br/>   
In ```test_autopilot.py``` specify the CONV_NET_MODEL directory for your saved model. Also specify if you want to record data from the test. Run ```python3 test_autopilot.py``` if you want to record data from the test, specify the session as an argument in the command line execution. Data will be saved to ```data/roof_cam/raw_autonomous```<br/>   
Once the program is running open the game (if you have 2 monitors it makes it easier to monitor the program while testing) Get your truck onto the highway and up to reasonable speed. Press **B** to engage the autopilot. If your button bindings are set up correctly this should also switch to the roof camera.<br/>   
**LB** and **RB** activate respective lane changes.<br/>

## Notes
For data collection and cleaning, removing data of changing lanes and odd outliers drastically improves model performance. This system is essentialy meant to be an advanced lane assist with additional features. So removing data that is not staying in lane is ideal.<br/>
For data collection and testing, using the same truck also improves performance. In my testing I bought the cheapest Peterbilt truck and used it for my data collection and testing. This is because different trucks have different roof heights which affects the height of the roof camera.<br/>

## References
* Python vJoy library https://github.com/tidzo/pyvjoy<br/>
* Capture Xbox controller inputs using inputs.py https://github.com/kevinhughes27/TensorKart/blob/master/utils.py<br/>
* End to End CNN for predicting steering wheel commands https://devblogs.nvidia.com/deep-learning-self-driving-cars/<br/>
