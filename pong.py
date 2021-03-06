import numpy as np
import keras
import gym
import os
import h5py

import matplotlib
import scipy

from matplotlib import pyplot as plt
from scipy import misc
from keras.models import Sequential
from keras.layers import Conv2D


from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras import optimizers

img_dim = 40
num_env_variables = img_dim*img_dim*2 # each sate is 2 frames of the pong game
num_env_actions = 6
num_initial_observation = 40
learning_rate =  0.001
apLearning_rate = 0.002
weigths_filename = "Pong-Reduced-v2-weights.h5"
apWeights_filename = "Pong-Reduced-v2-weights-ap.h5"

#range within wich the SmartCrossEntropy action parameters will deviate from
#remembered optimal policy
sce_range = 0.2
b_discount = 0.99
max_memory_len = 30000
starting_explore_prob = 0.03
training_epochs = 5
mini_batch = 256
load_previous_weights = True
observe_and_train = True
save_weights = True
num_games_to_play = 30000


#One hot encoding array
possible_actions = np.arange(0,num_env_actions)
actions_1_hot = np.zeros((num_env_actions,num_env_actions))
actions_1_hot[np.arange(num_env_actions),possible_actions] = 1

#Create testing enviroment
env = gym.make('Pong-v0')
env.reset()

#initialize training matrix with random states and actions
dataX = np.random.random(( 5,num_env_variables+num_env_actions ))
#Only one output for the total score / reward
dataY = np.random.random((5,1))

#initialize training matrix with random states and actions
apdataX = np.random.random(( 5,num_env_variables ))
apdataY = np.random.random((5,num_env_actions))

def custom_error(y_true, y_pred, Qsa):
    cce=0.001*(y_true - y_pred)*Qsa
    return cce


#nitialize the Reward predictor model
model = Sequential()
model.add(Dense(2048, activation='relu', input_dim=dataX.shape[1]))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(dataY.shape[1]))
opt = optimizers.adam(lr=learning_rate)
model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])


#initialize the action predictor model
action_predictor_model = Sequential()
action_predictor_model.add(Dense(256, activation='relu', input_dim=apdataX.shape[1]))
action_predictor_model.add(Dense(32, activation='relu'))
action_predictor_model.add(Dense(apdataY.shape[1]))

opt2 = optimizers.adam(lr=apLearning_rate)

action_predictor_model.compile(loss='mse', optimizer=opt2, metrics=['accuracy'])



#load previous model weights if they exist
if load_previous_weights:
    dir_path = os.path.realpath(".")
    fn = dir_path + "/"+weigths_filename
    print("filepath ", fn)
    if  os.path.isfile(fn):
        print("loading weights")
        model.load_weights(weigths_filename)
    else:
        print("File ",weigths_filename," does not exis. Retraining... ")

#load previous action predictor model weights if they exist
if load_previous_weights:
    dir_path = os.path.realpath(".")
    fn = dir_path + "/"+ apWeights_filename
    print("filepath ", fn)
    if  os.path.isfile(fn):
        print("loading weights")
        action_predictor_model.load_weights(apWeights_filename)
    else:
        print("File ",apWeights_filename," does not exis. Retraining... ")





#Record first 500 in a sequence and add them to the training sequence
total_steps = 0

memorySA = []
memoryS = []
memoryA = []
memoryR = []

mstats = []
num_games_won = 0


#takes a single game frame as input
#preprocesses before feeding into model
def preprocessing(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I = scipy.misc.imresize(I,size=(img_dim,img_dim))
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  #I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  I = I/255
  return I.astype(np.float).ravel() #flattens

def predictTotalRewards(qstate, action):
    qs_a = np.concatenate((qstate,action), axis=0)
    predX = np.zeros(shape=(1,num_env_variables+num_env_actions))
    predX[0] = qs_a

    #print("trying to predict reward at qs_a", predX[0])
    pred = model.predict(predX[0].reshape(1,predX.shape[1]))
    remembered_total_reward = pred[0][0]
    return remembered_total_reward


def GetRememberedOptimalPolicy(qstate):
    predX = np.zeros(shape=(1,num_env_variables))
    predX[0] = qstate

    #print("trying to predict reward at qs_a", predX[0])
    pred = action_predictor_model.predict(predX[0].reshape(1,predX.shape[1]))
    r_remembered_optimal_policy = pred[0]
    return r_remembered_optimal_policy

def SmartCrossEntropy(current_optimal_policy):
    sce = np.zeros(shape=(num_env_actions))
    #print("current_optimal_policy", current_optimal_policy)
    for i in range(num_env_actions):
        sce[i] = current_optimal_policy[i] + sce_range * (np.random.rand(1)*2 - 1)
        if sce[i] > 1:
            sce[i] = 1.0
        if sce[i] < -1:
            sce[i] = -1
    return sce


if observe_and_train:

    #Play the game 500 times
    for game in range(num_games_to_play):
        gameSA = []
        gameS = []
        gameA = []
        gameR = []
        num_points = 0

        previous_state = np.zeros(img_dim*img_dim)
        #Get the Q state
        qs = env.reset()


        #print("qs ", qs)
        for step in range (7000):
            qs = preprocessing(qs)
            sequenceQS = np.concatenate((previous_state,qs),axis=0)
            #sequenceQS = np.abs(qs - previous_state)

            if game < num_initial_observation or game%5==0:
                #take a radmon action
                a = keras.utils.to_categorical(env.action_space.sample(),num_env_actions)[0]
            else:
                prob = np.random.rand(1)
                explore_prob = starting_explore_prob-(starting_explore_prob/num_games_to_play)*game

                #Chose between prediction and chance
                if prob < explore_prob:
                    #take a random action
                    a = keras.utils.to_categorical(env.action_space.sample(),num_env_actions)[0]


                else:
                    predictedRewards = np.zeros(6)
                    for i in range(6):
                        predictedRewards[i] = predictTotalRewards(sequenceQS,
                            keras.utils.to_categorical(i,num_env_actions)[0])
                        #print("predicting ",keras.utils.to_categorical(i,num_env_actions)[0])

                    #print("predictedRewards",predictedRewards)
                    a = np.argmax(predictedRewards)

                    a = keras.utils.to_categorical(a,num_env_actions)[0]
            env.render()
            qs_a = np.concatenate((sequenceQS,a), axis=0)

            #get the target state and reward
            s,r,done,info = env.step(np.argmax(a))
            #record only the first x number of states

            gameSA.append(qs_a)
            #gameS= np.vstack((gameS, sequenceQS))
            gameR.append([r])
            #gameA = np.vstack((gameA, np.array([a])))

            #if step > 1898:
                #done = True

            if r >=1:
                num_games_won +=1
                num_points +=1
            if done:
                #done = True
                if num_points >=20:
                    print("GAME WON ***")


                #Calculate Q values from end to start of game
                #mstats.append(step)
                for i in range(0,len(gameR)):
                    #print("Updating total_reward at game epoch ",(gameY.shape[0]-1) - i)
                    if i==0:
                        #print("reward at the last step ",gameR[(gameR.shape[0]-1)-i][0])
                        gameR[(len(gameR)-1)-i][0] = gameR[(len(gameR)-1)-i][0]
                    else:
                        #print("local error before Bellman", gameR[(gameR.shape[0]-1)-i][0],"Next error ", gameR[(gameR.shape[0]-1)-i+1][0])
                        gameR[(len(gameR)-1)-i][0] = gameR[(len(gameR)-1)-i][0]+b_discount*gameR[(len(gameR)-1)-i+1][0]
                        #print("reward at step",i,"away from the end is",gameR[(gameR.shape[0]-1)-i][0])
                    if i==len(gameR)-1:
                        print("Training Game #",game,"#scores", num_points, " total # scores ", num_games_won,"avg scores per match ",num_games_won/(game+1), "memory ",len(memoryR)," finished with headscore ", gameR[(len(gameR)-1)-i][0])

                #Add experience to memory
                memorySA = memorySA+gameSA
                #memoryS = np.concatenate((memoryS,gameS),axis=0)
                memoryR = memoryR+gameR
                #memoryA = np.concatenate((memoryA,gameA),axis=0)

                #tempGameA = tempGameA[1:]
                #tempGameS = tempGameS[1:]
                #tempGameRR = tempGameRR[1:]
                #tempGameR = tempGameR[1:]
                #tempGameSA = tempGameSA[1:]


                #if memory is full remove first element
                if len(memoryR) >= max_memory_len:
                    memoryR = memoryR[np.alen(gameR):]
                    memorySA = memorySA[np.alen(gameR):]
                    #print("memory full. mem len ", np.alen(memoryX))
                    #for l in range(np.alen(gameR)):
                        #memorySA = np.delete(memorySA, 0, axis=0)
                        #memoryR = np.delete(memoryR, 0, axis=0)
                        #memoryA = np.delete(memoryA, 0, axis=0)
                        #memoryS = np.delete(memoryS, 0, axis=0)


            #Update the states
            previous_state = np.copy(qs)
            qs=s


            #Retrain every X failures after num_initial_observation
            if done and game >= num_initial_observation:
                if game%5 == 0 and game>15:
                    print("Training  game# ", game,"momory size", len(memorySA))

                    #training Reward predictor model
                    model.fit(np.asarray(memorySA),np.asarray(memoryR), batch_size=mini_batch,epochs=training_epochs,verbose=2)

                    #training action predictor model
                    #action_predictor_model.fit(memoryS,memoryA, batch_size=mini_batch, epochs=training_epochs,verbose=0)

            if done and game >= num_initial_observation:
                if save_weights and game%20 == 0:
                    #Save model
                    print("Saving weights")
                    model.save_weights(weigths_filename)
                    action_predictor_model.save_weights(apWeights_filename)

            if done:
                #Game ended - Break
                break


plt.plot(mstats)
plt.show()

if save_weights:
    #Save model
    print("Saving weights")
    model.save_weights(weigths_filename)
    action_predictor_model.save_weights(apWeights_filename)