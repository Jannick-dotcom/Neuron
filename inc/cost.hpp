#pragma once
#include <cmath>

class Cost {
public:
    Cost() {}
    ~Cost() {}

    virtual double cost(double output, double target) = 0;
    virtual double costDerivative(double output, double target) = 0;
};

class CrossEntropyCost : public Cost {
public:
    CrossEntropyCost() {}
    ~CrossEntropyCost() {}

    double cost(double output, double target) {
        return -target * log(output) - (1 - target) * log(1 - output);
    }

    double costDerivative(double output, double target) {
        return output - target;
    }
};

class QuadraticCost : public Cost {
public:
    QuadraticCost() {}
    ~QuadraticCost() {}

    double cost(double output, double target) {
        return 0.5 * pow(output - target, 2);
    }

    double costDerivative(double output, double target) {
        return output - target;
    }
};

class ExponentialCost : public Cost {
public:
    ExponentialCost() {}
    ~ExponentialCost() {}

    double cost(double output, double target) {
        return exp(-pow(output - target, 2));
    }

    double costDerivative(double output, double target) {
        return -2 * (output - target) * exp(-pow(output - target, 2));
    }
};