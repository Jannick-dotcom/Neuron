#pragma once

#include <cmath>

class ActivationFunction {
public:
    virtual double operator()(double x) = 0;
    virtual double derivative(double x) = 0;
};

class Linear : public ActivationFunction {
public:
    double operator()(double x) override {
        return x;
    }
    double derivative(double x) override {
        return 1.0;
    }
};

class Sigmoid : public ActivationFunction {
public:
    double operator()(double x) override {
        return 1.0 / (1.0 + exp(-x));
    }
    double derivative(double x) override {
        return (*this)(x) * (1.0 - (*this)(x));
    }
};

class Tanh : public ActivationFunction {
public:
    double operator()(double x) override {
        return tanh(x);
    }
    double derivative(double x) override {
        return 1.0 - (*this)(x) * (*this)(x);
    }
};

class ReLU : public ActivationFunction {
public:
    double operator()(double x) override {
        return x > 0.0 ? x : 0.0;
    }
    double derivative(double x) override {
        return x > 0.0 ? 1.0 : 0.0;
    }
};

class LeakyReLU : public ActivationFunction {
public:
    double operator()(double x) override {
        return x > 0.0 ? x : 0.01 * x;
    }
    double derivative(double x) override {
        return x > 0.0 ? 1.0 : 0.01;
    }
};

class Softmax : public ActivationFunction {
public:
    double operator()(double x) override {
        return exp(x);
    }
    double derivative(double x) override {
        return 1.0;
    }
};