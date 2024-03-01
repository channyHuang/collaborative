#include "trainer.h"

Trainer::Trainer(uint32_t stateDim, uint32_t actionDim) {
    vTupleBuffer.resize(nNumOfTupleBuffer);

    nDimOfAction = actionDim;

    vMaxAction.resize(nDimOfAction);
    vMaxAction.fill(1.f);

    vRewardWeights.resize(nDimOfAction);
    vRewardWeights(0) = 0.2f;
    vRewardWeights(1) = 0.8f;

    actor = PolicyNet(3, {stateDim, 8, actionDim});
    float sum = (1 + nNumOfTupleBuffer) * nNumOfTupleBuffer * 0.5f;
    if (isZero(sum)) sum = 1;
    float invsum = 1.f / sum;
    critics.resize(nNumOfTupleBuffer);
    for (uint32_t i = 0; i < nNumOfTupleBuffer; ++i) {
        critics[i] = PolicyNet(3, { stateDim, 8, 1 });
        fGama[i] = (i + 1) * invsum;
	}
}

void Trainer::init() {
    mTerrain = RandomGenerator::getInstance()->generateTerrain(nTerrainDim, nTerrainDim, nMaxTerrainHeight);
    std::ofstream ofs("terrain.txt");
    ofs << nTerrainDim << std::endl;
    for (int i = 0; i < nTerrainDim; ++i) {
        for (int j = 0; j < nTerrainDim; ++j) {
            ofs << mTerrain(i, j) << " ";
        }
        ofs << std::endl;
    }
    ofs.close();
}

MyTuples Trainer::getTuple(uint32_t idx) {
    return vTupleBuffer[idx];
}

void Trainer::addTuple(const MyTuples& tuple) {
    vTupleBuffer[nTupleEndPos] = tuple;
    nTupleEndPos = (nTupleEndPos + 1) % nNumOfTupleBuffer;

    if (nNumOfTuple < nNumOfTupleBuffer) {
        nNumOfTuple++;
    }
}

// policy: 0: random; 1: prob random; 2: policy
Eigen::VectorXf Trainer::getAction(const Eigen::VectorXf& state, int policy) {
    const float epsilon = 0.3f;
    float lambda = RandomGenerator::getInstance()->generateFloat();
    if (policy == 0 || (policy ==1 && lambda < epsilon)) {
        return RandomGenerator::getInstance()->generateAction(nDimOfAction, vMaxAction);
    }
    return actor.getAction(state);
}

float Trainer::calcReward(const MyTuples& tuple) {
    if (!isValidPos(tuple.stateNext.segment(0, 2), nTerrainDim) || !isValidPos(tuple.stateNext.segment(2, 2), nTerrainDim)) return -10000.f;

    Eigen::VectorXf err(2);
    const float gamma = 0.9f;
    Eigen::VectorXf diffAction = (vMaxAction - tuple.action);

    err(0) = std::exp(-gamma * diffAction.norm());
    //err(1) = std::sqrt(nMaxTerrainHeight * nMaxTerrainHeight) + nMaxTerrainHeight - mTerrain((int)tuple.stateNext(0), (int)tuple.stateNext(1)) + mTerrain((int)tuple.state(0), (int)tuple.state(1)) - (tuple.stateNext.segment(2, 2) - tuple.stateNext.segment(0, 2)).norm();
    err(1) = -(tuple.stateNext.segment(2, 2) - tuple.stateNext.segment(0, 2)).norm();

    float reward = vRewardWeights.transpose() * err;
    return reward;
}

void Trainer::train(MyTuples& tuple, bool isActor) {
    if (isActor) {
        float err = (tuple.qNextValue - tuple.qValue);
        actor.train(tuple.state, tuple.action + Eigen::VectorXf::Ones(nDimOfAction) * err);
        return;
    }
    Eigen::VectorXf reward;
    for (uint32_t i = 0; i < critics.size(); ++i) {
        float y = 0.f;
        for (uint32_t j = 0; j < nNumOfTuple; ++j) {
            MyTuples tmpTuple = getTuple(j);
            float reward = calcReward(tmpTuple);

            y += reward + fGama[i] * tmpTuple.qValue;
        }
        y /= critics.size();
        tuple.qValue = y;
        critics[i].trainCritic(tuple);
    }
}

float Trainer::getQValue(MyTuples& tuple) {
    for (uint32_t i = 0; i < critics.size(); ++i) {
        tuple.qValue += critics[i].getValue(tuple.state, tuple.action);
    }
    tuple.qValue /= critics.size();
    return tuple.qValue;
}

float Trainer::getQNextValue(MyTuples& tuple) {
    for (uint32_t i = 0; i < critics.size(); ++i) {
        tuple.qNextValue += critics[i].getValue(tuple.stateNext, tuple.action);
    }
    tuple.qNextValue /= critics.size();
    return tuple.qNextValue;
}

float Trainer::getTerrainHeight(const Eigen::VectorXf& pos) const {
    if ((int)pos(0) < 0 || (int)pos(0) >= nTerrainDim) return nMaxTerrainHeight;
    if ((int)pos(1) < 0 || (int)pos(1) >= nTerrainDim) return nMaxTerrainHeight;
    return mTerrain((int)pos(0), (int)pos(1));
}
