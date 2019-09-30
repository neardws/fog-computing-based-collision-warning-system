import numpy as np
import pandas as pd
from SimpleHOHMM import HiddenMarkovModelBuilder as Builder
from SimpleHOHMM import HiddenMarkovModel as HMM


def getallstates():
    states = list()
    for i in range(-max, max + 1):
        for j in range(-max, max + 1):
            states.append(str(i) + ',' + str(j))
    print("Number of states is" + str(len(states)))
    print(states)
    return states


def getobs(filename):
    df = pd.read_csv(filename)
    vehicleid = df['vehicleID'].drop_duplicates()
    obs = list()
    for id in vehicleid:
        baselon = 0
        baselat = 0
        tracestate = list()
        trace = df[df['vehicleID'] == id].sort_values(by=['time'])
        longitudes = trace['longitude']
        latitudes = trace['latitude']
        print("id is " + str(id))
        for i in range(len(trace)):
            if i == 0:
                baselon = longitudes.values[i]
                baselat = latitudes.values[i]
            else:
                lon = longitudes.values[i]
                lat = latitudes.values[i]
                state = getstate(lon, lat, baselon, baselat)
                # print(state)
                if state == '0,0':
                    # print("state is 0,0")
                    pass
                else:
                    tracestate.append(state)
                # print(state)
                baselon = lon
                baselat = lat
        if len(tracestate):
            if len(obs) <= 500:
                obs.append(tracestate)
            else:
                break
    print("obs length is" + str(len(obs)))
    with open('tracesstates.txt', 'a', encoding='utf-8') as f:
        f.write(str(obs))
    return obs


def trainmodel():
    builder = Builder()
    builder.set_all_obs(getobs(inputfile))
    builder.set_single_states(getallstates())
    print("start learning")
    hmm = builder.build(highest_order=2, k_smoothing=0.001)
    print(hmm.get_parameters())
    with open('hmm.txt', 'a', encoding='utf-8') as f:
        f.write(str(hmm))
        f.write(str(hmm.get_parameters()))


def main():
    trainmodel()


if __name__ == '__main__':
    main()