# Autonomous-Truck
An autonomous driving system built to drive in a truck driving simulator

## Installation
Git clone the repository and ```cd``` into the directory, install requirements and clone pyvjoy into Autonomous-Truck
``` 
git clone https://github.com/greerviau/Autonomous-Truck.git && cd Autonomous-Truck
pip install -r requirements.txt
git clone https://github.com/tidzo/pyvjoy
```
Download and install vJoy from http://vjoystick.sourceforge.net/site/index.php/download-a-install/download  
Navigate to ```vJoy/x86``` in wherever you installed vJoy. Copy vJoyInterface.dll and paste it into the pyvjoy directory.  
___
## Usage 
### Game Settings
Make sure the game detects the gamepad<br/>
Make sure the _Controller subtype_ is set to **Gamepad, joystick**<br/>
Use **1280x720** resolution in game<br/>
Try to use the highest graphics settings possible while still being able to run the program effectively (this will take some fine-tuning)<br/>
With the gamepad plugged in, set the _Steering axis_ to **Joy X Axis**.<br/>
Set the _Acceleration and Brake axis_ to **Joy RY Axis**, this will be converted to **Joy Y Rotation** when using **Keyboard + vJoy Device** as input. Set the _Acceleration axis mode_ to **Centered** and the _Brake axis mode_ to **Inverted and Centered**. (This is used for the autopilot to accelerate and brake, you do not have to use the Y axis for data collection).<br/>
Bind _Light Modes_ to **L**<br/>
Bind _Roof Camera_ to the controller **B** button and the **P** key<br/>
___
### Data Collection
#### Recording Data
For collecting data make sure the input is set to **Keyboard + XInput Gamepad 1** and the _Controller subtype_ is set to **Gamepad, joystick**<br/>
To collect data run ```python3 collect_data.py <session>``` Make sure to specify different sessions for every execution.<br/> 
While recording, use the **B** button to start a new recording, this will not create a new session but instead split into a new clip.<br/>  
Use this feature to start recording new clips of desired data. This will make data cleaning easier. Ex. Before changing lanes, press **B** before changing and press **B** after lane change is finished. This will create 3 clips, 1 before lane change, 1 of the lane change and the final will continue recording the rest of the drive. Then durring data cleaning simply delete the clip of the lane change.<br/>  
Recording sessions will be saved to ```data/roof_cam/raw/<session>/<clips>```
#### Cleaning
If additional cleaning is required, run ```python3 clip_video.py raw/<session>/<clip>``` While video is playing, press **q** to keyframe. Once video is done playing the program will split the video along key frames and saved to ```data/roof_cam/raw/<session>/<clips>/<splits>```<br/>   
Then simply move the clips that you want to keep to ```data/roof_cam/raw/<session>``` and discard the rest.
___
### Preprocessing
Once the data has been cleaned, run ```python3 preprocess.py``` This will preprocess all of the clips in ```data/roof_cam/raw``` The subfolders of this directory must have the file structure of ```<session>/<clips>``` with the mp4 and csv files within.<br/>  
This will save the preprocessed data to ```data/roof_cam/processed``` The data will be divided into sessions but the clips will be aggregated into X.npy and Y.csv<br/>   
There will also be a total aggregate of all sessions as X.npy and Y.csv 
___
### Train Steering Model
After preprocessing, open ```train_conv_net.py``` Make sure to specify the SAVE_PATH for the model as well as the hyperparams.<br/> 
Run ```python3 train_conv_net.py``` to train the model.
___
### Train Digit Recognition
To train the digit recognition for monitoring speed run ```python3 train_digit_rec.py```
___
### Train Brake Prediction model
To train the conv net for brake prediction run ```python3 train_brake_net.py```
___
### Testing
Open your game and in gameplay settings set your input as **Keyboard + vJoy Device**.<br/>
If vJoy is not detected then run ```python3 detect_vjoy_ingame.py``` while your game is open and it should ask you to use vJoy as a controller. Like the Xbox controller, make sure the _Controller subtype_ is set to **Gamepad, joystick**<br/>   
In ```test_autopilot.py``` specify the CONV_NET_MODEL directory for your saved model. Also specify if you want to record data from the test. Run ```python3 test_autopilot.py``` if you want to record data from the test, specify the session as an argument in the command line execution. Data will be saved to ```data/roof_cam/raw_autonomous```<br/>   
Once the program is running open the game (if you have 2 monitors it makes it easier to monitor the program while testing) Get your truck onto the highway and up to reasonable speed. Press **B** on your controller or **P** on your keyboard to engage the autopilot. If your button bindings are set up correctly this should also switch to the roof camera. While the system is running you still have control over steering with the keyboard **A** and **D** keys.<br/>   
**LB** and **RB** activate respective lane changes.<br/>
You can disengage the system with the keyboard **W** and **S** keys aswell, this allows for disengagement on human throttle or brake.
___
## Notes
* For data collection and cleaning, removing data of changing lanes and odd outliers drastically improves model performance. This system is essentialy meant to be an advanced lane assist with additional features. So removing data that is not staying in lane is ideal.
* For data collection and testing, using the same truck also improves performance. In my testing I bought the cheapest Peterbilt truck and used it for my data collection and testing. This is because different trucks have different roof heights which affects the height of the roof camera. Alternatively you could collect data from a large enough sample of trucks so that your model can generalize across varying roof camera heights. I attempted this and it did work however results are still better if you use the same truck for all of your training and testing.
* Load on the truck seems to affect the performance of the system if not trained on a robust enough dataset. Essentially since the system doesnt know if the truck is under load or not its predictions do not change accordingly however load may affect how quickly a miscalculation can be corrected for. This effect is miniscule based on testing.
___
## References
* Python vJoy library https://github.com/tidzo/pyvjoy<br/>
* Capture Xbox controller inputs using inputs.py https://github.com/kevinhughes27/TensorKart/blob/master/utils.py<br/>
* End to End CNN for predicting steering wheel commands https://devblogs.nvidia.com/deep-learning-self-driving-cars/<br/>
