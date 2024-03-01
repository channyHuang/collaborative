#include "trainer.h"
#include "policyNet.h"

#include "dataStruct.h"

#include <iostream>

int Cood2Index(int x, int y, int dim) {
    return abs(x) * dim + abs(y);
}

int nTerrainDim = 100;
int nMaxTerrainHeight = 100;
uint32_t nStateDim = 5;

void test(Trainer& trainer) {
    uint32_t nIterateTimes = 10000;
    // test
    Eigen::VectorXf pos = RandomGenerator::getInstance()->generateState(2, nTerrainDim);
    Eigen::VectorXf targetPos = RandomGenerator::getInstance()->generateState(2, nTerrainDim);

    Eigen::VectorXf state(nStateDim);
    state.segment(0, 2) = pos;
    state.segment(2, 2) = RandomGenerator::getInstance()->generateState(2, nTerrainDim);
    state(4) = trainer.getTerrainHeight(pos);

    Eigen::VectorXf action;
    uint32_t i = 0;
    output2File("out.txt", state);
    while (true) {
        // get action by policy
        action = trainer.getAction(state, false);
        state.segment(0, 2) += action;
        targetPos = RandomGenerator::getInstance()->generateVector(2);
        state.segment(2, 2) += targetPos;
        if (state(2) < 0) state(2) = -state(2);
        if (state(3) < 0) state(3) = -state(3);
        state(4) = trainer.getTerrainHeight(state.segment(0, 2));
        //output("step " + std::to_string(i++), state);
        output2File("out.txt", state);
        if (i++ >= nIterateTimes) break;
    }
}

void train(Trainer& trainer) {
    Eigen::VectorXf pos = RandomGenerator::getInstance()->generateState(2, nTerrainDim);
    Eigen::VectorXf targetPos = RandomGenerator::getInstance()->generateState(2, nTerrainDim);

    Eigen::VectorXf state(nStateDim);
    state.segment(0, 2) = pos;
    state.segment(2, 2) = targetPos;
    state(4) = trainer.getTerrainHeight(pos);

    // train
    int getActionByPolicy = 1;
    uint32_t nIterateTimes = 1000000;
    for (uint32_t i = 0; i < nIterateTimes; ++i) {
        if (i % 10000 == 0) std::cout << "train " << i << std::endl;
        // get action by policy
        Eigen::VectorXf action = trainer.getAction(state, getActionByPolicy);
        getActionByPolicy = 1;

        // apply action, calc reward
        Eigen::VectorXf stateNext = state;
        stateNext.segment(0, 2) += action;
        targetPos = RandomGenerator::getInstance()->generateVector(2);
        stateNext.segment(2, 2) += targetPos;
        if (stateNext(2) < 0) stateNext(2) = -stateNext(2);
        if (stateNext(3) < 0) stateNext(3) = -stateNext(3);
        stateNext(4) = trainer.getTerrainHeight(stateNext.segment(0, 2));
        //output("state next", stateNext);
        float reward = 0;
        // add tuple buffer
        MyTuples tuple(i, reward, state, action, stateNext);
        reward = trainer.calcReward(tuple);
        if (reward < -1000.f) {
            //qDebug() << i << ": reward too small";
            getActionByPolicy = 0;
        }
        trainer.addTuple(tuple);
        // feedback network
        trainer.train(tuple, false);
        trainer.train(tuple, true);     

        state = stateNext;
        //output(std::to_string(i), state);
    }

}

int main(int argc, char *argv[])
{
    Trainer trainer(nStateDim, 2);
    trainer.setTerrainDim(nTerrainDim);
    trainer.setTerrainHeight(nMaxTerrainHeight);
    trainer.init();
    std::cout << "start training" << std::endl;
    train(trainer);
    std::cout << "start testing" << std::endl;
    test(trainer);
    std::cout << "end" << std::endl;
    return 0;
}
